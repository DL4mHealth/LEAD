import torch
import torch.nn as nn
import torch.nn.functional as F


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('bfn,bnv->bfv', (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv1d(c_in, c_out, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x):
        return self.mlp(x)


class GCN(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(GCN, self).__init__()
        self.nconv = nconv()
        c_in = (order + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        return F.relu(h)



class GraphLayer(nn.Module):
    def __init__(self, configs):
        super(GraphLayer, self).__init__()
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.nodedim = configs.nodedim

        self.nodevector_1 = nn.Parameter(torch.randn(self.enc_in, self.nodedim))
        self.nodevector_2 = nn.Parameter(torch.randn(self.nodedim, self.enc_in))

        self.nodevec_gate1 = nn.Sequential(
                nn.Linear(self.d_model + self.nodedim, 1),
                nn.Tanh(),
                nn.ReLU())

        self.nodevec_gate2 = nn.Sequential(
            nn.Linear(self.d_model + self.nodedim, 1),
            nn.Tanh(),
            nn.ReLU())

        self.nodevec_linear1 = nn.Linear(self.d_model, self.nodedim)
        self.nodevec_linear2 = nn.Linear(self.d_model, self.nodedim)

        self.gcn = GCN(self.d_model, self.d_model, self.dropout)


    def forward(self, x):
        B, _, _ = x.size()
        nodevector_1 = self.nodevector_1.view(1, self.enc_in, self.nodedim).repeat(B, 1, 1)
        nodevector_2 = self.nodevector_2.view(1, self.nodedim, self.enc_in).repeat(B, 1, 1)

        x_gate_1 = self.nodevec_gate1(torch.cat([x, nodevector_1], dim=-1))
        x_gate_2 = self.nodevec_gate2(torch.cat([x, nodevector_2.permute(0, 2, 1)], dim=-1))

        x_p1 = x_gate_1 * self.nodevec_linear1(x)
        x_p2 = x_gate_2 * self.nodevec_linear2(x)

        nodevector_1 = nodevector_1 + x_p1
        nodevector_2 = nodevector_2 + x_p2.permute(0, 2, 1)

        adp = F.softmax(F.relu(torch.matmul(nodevector_1, nodevector_2)), dim=-1)

        adjacency_matrix = adp

        adp = [adp]
        x = x.permute(0, 2, 1)
        x = self.gcn(x, adp)
        return x, adjacency_matrix


class MRGNN(nn.Module):
    def __init__(self, configs, res_len):
        super(MRGNN, self).__init__()
        self.resolution_list = list(map(int, configs.resolution_list.split(",")))

        self.res_num = len(self.resolution_list)
        self.res_len = res_len
        self.mr_graphs = nn.ModuleList([GraphLayer(configs) for _ in range(self.res_num)])

    def forward(self, x):
        res = []
        adjacency_matrix_list = []
        for l in range(self.res_num):
            out, adjacency_matrix = self.mr_graphs[l](x[l])
            res.append(out)
            adjacency_matrix_list.append(adjacency_matrix)

        out = torch.stack(res, dim=-1)
        out = torch.mean(out, dim=-1)
        return out, adjacency_matrix_list
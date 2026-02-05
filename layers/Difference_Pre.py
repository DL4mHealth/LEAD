import torch
import torch.nn as nn

class DifferenceDataEmb(nn.Module):
    def __init__(self, res_num, enc_in, d_model):
        super(DifferenceDataEmb, self).__init__()
        self.res_num = res_num
        self.enc_in = enc_in
        self.d_model = d_model
        self.embeddings = nn.ModuleList([nn.Linear(self.enc_in, self.d_model) for _ in range(self.res_num)])

    def forward(self, multi_res_data):
        x_diff_list = []
        x_padding_list = []
        for l in range(self.res_num):
            x = multi_res_data[l].permute(0, 2, 1)
            x_padding = torch.concatenate([x[:, 0:1, :], x], dim=1)
            x_diff = torch.diff(x_padding, dim=1)
            x_diff_emb = self.embeddings[l](x_diff)
            x_diff_list.append(x_diff_emb)
            x_padding_list.append(x_padding)

        return x_diff_list, x_padding_list

class DataRestoration(nn.Module):
    def __init__(self, res_num, enc_in, d_model):
        super(DataRestoration, self).__init__()
        self.res_num = res_num
        self.enc_in = enc_in
        self.d_model = d_model
        self.projections = nn.ModuleList([nn.Linear(self.d_model, self.enc_in) for _ in range(self.res_num)])

    def forward(self, x_diff_list, x_padding_list):
        x_out_list = []
        for l in range(self.res_num):
            x_diff = self.projections[l](x_diff_list[l])
            _x_out = x_diff + x_padding_list[l][:, :-1, :]
            _x_out = _x_out.permute(0, 2, 1)
            x_out_list.append(_x_out)

        return x_out_list

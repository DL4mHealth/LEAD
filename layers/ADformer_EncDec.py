import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff, dropout, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.conv3 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv4 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x_t, x_c, attn_mask=None, tau=None, delta=None):
        new_x_t, new_x_c, attn_t, attn_c = self.attention(x_t, x_c, attn_mask=attn_mask, tau=tau, delta=delta)
        x_t = [_x_t + self.dropout(_nx_t) for _x_t, _nx_t in zip(x_t, new_x_t)]
        x_c = [_x_c + self.dropout(_nx_c) for _x_c, _nx_c in zip(x_c, new_x_c)]

        y_t = x_t = [self.norm1(_x_t) for _x_t in x_t]
        y_t = [self.dropout(self.activation(self.conv1(_y_t.transpose(-1, 1)))) for _y_t in y_t]
        y_t = [self.dropout(self.conv2(_y_t).transpose(-1, 1)) for _y_t in y_t]

        y_c = x_c = [self.norm3(_x_c) for _x_c in x_c]
        y_c = [self.dropout(self.activation(self.conv3(_y_c.transpose(-1, 1)))) for _y_c in y_c]
        y_c = [self.dropout(self.conv4(_y_c).transpose(-1, 1)) for _y_c in y_c]

        return [self.norm2(_x_t + _y_t) for _x_t, _y_t in zip(x_t, y_t)], \
            [self.norm4(_x_c + _y_c) for _x_c, _y_c in zip(x_c, y_c)], attn_t, attn_c


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x_t, x_c, attn_mask=None, tau=None, delta=None):
        # x [[B, L1, D], [B, L2, D], ...]
        attns_t = []
        attns_c = []
        for attn_layer in self.attn_layers:
            x_t, x_c, attn_t, attn_c = attn_layer(x_t, x_c, attn_mask=attn_mask, tau=tau, delta=delta)
            attns_t.append(attn_t)
            attns_c.append(attn_c)

        """# concat all the outputs
        if x_t:
            x_t = torch.cat(x_t, dim=1)  # (batch_size, patch_num_1 + patch_num_2 + ... , d_model)
        else:
            x_t = None
        if x_c:
            x_c = torch.cat(x_c, dim=1)  # (batch_size, enc_in_1 + enc_in_2 + ... , d_model)
        else:
            x_c = None"""
        # only concat the routers. router is the last patch/channel of each element in the list

        """if x_t:
            x_t = torch.cat([x[:, -1, :].unsqueeze(1) for x in x_t], dim=1)   # (batch_size, len(patch_len_list), d_model)
        else:
            x_t = None
        if x_c:
            x_c = torch.cat([x[:, -1, :].unsqueeze(1) for x in x_c], dim=1)  # (batch_size, len(up_dim_list), d_model)
        else:
            x_c = None

        if self.norm is not None:
            x_t = self.norm(x_t) if x_t is not None else None
            x_c = self.norm(x_c) if x_c is not None else None"""

        if x_t:
            x_t = x_t
        else:
            x_t = None
        if x_c:
            x_c = x_c
        else:
            x_c = None

        return x_t, x_c, attns_t, attns_c

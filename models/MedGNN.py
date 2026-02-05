import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import MultiResolutionData, FrequencyEmbedding
from layers.MedGNN_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FormerLayer, DifferenceFormerlayer
from layers.Multi_Resolution_GNN import MRGNN
from layers.Difference_Pre import DifferenceDataEmb, DataRestoration


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.n_heads = configs.n_heads
        self.e_layers = configs.e_layers
        self.dropout = configs.dropout
        self.output_attention = configs.output_attention
        self.activation = configs.activation
        self.resolution_list = list(map(int, configs.resolution_list.split(",")))


        self.res_num = len(self.resolution_list)
        self.stride_list = self.resolution_list
        self.res_len = [int(self.seq_len//res)+1 for res in self.resolution_list]
        self.augmentations = configs.augmentations.split(",")

        # step1: multi_resolution_data
        self.multi_res_data = MultiResolutionData(self.enc_in, self.resolution_list, self.stride_list)

        # step2.1: frequency convolution network
        self.freq_embedding = FrequencyEmbedding(self.d_model, self.res_len, self.augmentations)

        # step2.2: difference attention network
        self.diff_data_emb = DifferenceDataEmb(self.res_num, self.enc_in, self.d_model)
        self.difference_attention = Encoder(
            [
                EncoderLayer(
                    DifferenceFormerlayer(
                        self.enc_in,
                        self.res_num,
                        self.d_model,
                        self.n_heads,
                        self.dropout,
                        self.output_attention
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        self.data_restoration = DataRestoration(self.res_num, self.enc_in, self.d_model)
        self.embeddings = nn.ModuleList([nn.Linear(res_len, self.d_model) for res_len in self.res_len])

        # step 3: transformer
        self.encoder = Encoder(
            [
                EncoderLayer(
                    FormerLayer(
                        len(self.resolution_list),
                        configs.d_model,
                        configs.n_heads,
                        configs.dropout,
                        configs.output_attention
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

        # step 4: multi-resolution GNN
        self.mrgnn = MRGNN(configs, self.res_len)

        # step 5: projection
        self.projection = nn.Linear(self.d_model * self.enc_in, configs.num_class)


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, fs=None, mask=None):
        B, T, C = x_enc.shape

        # step1: multi_resolution_data
        multi_res_data = self.multi_res_data(x_enc)

        # step2.1: frequency convolution network
        enc_out_1 = self.freq_embedding(multi_res_data)

        # step2.2: difference attention network
        x_diff_emb, x_padding = self.diff_data_emb(multi_res_data)
        x_diff_enc, attns = self.difference_attention(x_diff_emb, attn_mask=None)
        enc_out_2 = self.data_restoration(x_diff_enc, x_padding)
        enc_out_2 = [self.embeddings[l](enc_out_2[l]) for l in range(self.res_num)]

        # step 3: transformer
        data_enc = [enc_out_1[l] + enc_out_2[l] for l in range(self.res_num)]
        enc_out, attns = self.encoder(data_enc, attn_mask=None)

        # step 4: multi-resolution GNN
        output, adjacency_matrix_list = self.mrgnn(enc_out)

        # step 5: projection
        output = output.reshape(B, -1)
        output = self.projection(output)

        return output

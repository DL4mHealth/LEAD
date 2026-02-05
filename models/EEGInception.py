import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Conv_Blocks import InceptionBlock, SpatialBlock
import numpy as np


class Model(nn.Module):

    def __init__(self, configs, n_blocks=3, channels=(96, 192, 384),
                 kernel_sizes=(8, 16, 32), depth_multiplier=2, bottleneck_channels=32):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in

        assert len(channels) == n_blocks

        self.permute_in = lambda x: x.permute(0, 2, 1)
        self.permute_back = lambda x: x.permute(0, 2, 1)

        # Stack of Inception + Spatial blocks
        blocks = []
        in_ch = self.enc_in
        for out_ch in channels:
            blocks.append(
                InceptionBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_sizes=kernel_sizes,
                    bottleneck_channels=bottleneck_channels,
                    activation=nn.ReLU(inplace=True),
                    dropout=configs.dropout,
                )
            )
            blocks.append(
                SpatialBlock(
                    in_channels=out_ch,
                    depth_multiplier=depth_multiplier,
                    activation=nn.ReLU(inplace=True),
                )
            )
            in_ch = out_ch
        self.feature_extractor = nn.Sequential(*blocks)

        self.global_pool = nn.AdaptiveAvgPool1d(1)  # → [B, C, 1]

        if self.task_name == "supervised":
            self.classifier = nn.Sequential(
                nn.Flatten(),  # [B, C]
                nn.Dropout(configs.dropout),
                nn.Linear(in_ch, configs.num_class)
            )

    def supervised(self, x_enc, x_mark_enc):  # (batch_size, timestamps, enc_in)
        # [B, C, T]
        x = self.permute_in(x_enc)
        x = self.feature_extractor(x)

        # GAP + FC
        x = self.global_pool(x)        # [B, C, 1]
        output = self.classifier(x)

        return output                     # [B, num_class]

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, fs=None, mask=None):
        if self.task_name == 'supervised':
            dec_out = self.supervised(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        else:
            raise ValueError("Task name not recognized or not implemented within the EEGInception Model")

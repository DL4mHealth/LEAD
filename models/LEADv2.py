import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.LEAD_EncDec import (
    EncoderV2,
    EncoderLayerV2,
)
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import LEADv2Embedding
from layers.SelfAttention_Family import LEADLayerV2
import numpy as np
import random
from layers.Augmentation import get_augmentation
from einops import rearrange, repeat


def compute_patch_num(seq_len, patch_len, stride):
    """Compute the exact number of patches after right padding and unfold."""
    L, P, S = seq_len, patch_len, stride
    if L < P:
        pad_right = P - L
    else:
        rem = (L - P) % S
        pad_right = 0 if rem == 0 else (S - rem)
    Lp = L + pad_right
    patch_num = (Lp - P) // S + 1
    return patch_num


class Model(nn.Module):
    """
    LEADv2
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.output_attention = configs.output_attention
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.enc_in = configs.enc_in
        self.patch_num = compute_patch_num(
            configs.seq_len,
            configs.patch_len,
            configs.stride,
        )
        channel_names = configs.channel_names.split(",")
        if len(channel_names) != configs.enc_in:
            raise ValueError("channel_names length does not match enc_in")
        augmentations = configs.augmentations.split(",")
        # augmentations are necessary for contrastive pretraining
        if augmentations == ["none"] and "pretrain" in self.task_name:
            augmentations = ["patch0.2", "mask0.2", "channel0.2"]
        # Embedding
        self.enc_embedding = LEADv2Embedding(
            configs.enc_in,
            configs.seq_len,
            configs.d_model,
            configs.patch_len,
            configs.stride,
            configs.dropout,
            channel_names,
            configs.montage_name,
            augmentations,
        )
        # Encoder
        self.encoder = EncoderV2(
            [
                EncoderLayerV2(
                    LEADLayerV2(
                        configs.d_model,
                        configs.enc_in,
                        configs.n_heads,
                        self.patch_num,
                        configs.dropout,
                        configs.output_attention,
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

        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.input_dim = self.enc_in * configs.d_model
        if self.task_name == "supervised" or self.task_name == "finetune":
            self.classifier = nn.Linear(
                self.input_dim,
                configs.num_class
            )
        elif self.task_name == "pretrain_lead":
            # use for LEAD pretraining framework
            self.projection_head = nn.Sequential(
                nn.Linear(
                    self.input_dim,
                    configs.d_model * 2
                ),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(configs.d_model * 2, configs.d_model)
            )

    def supervised(self, x_enc, x_mark_enc, fs):
        # Embedding
        enc_out = self.enc_embedding(x_enc, fs)
        enc_out, attns_t, attns_c = self.encoder(enc_out, attn_mask=None)
        B, C_P, D = enc_out.shape
        C, P = self.enc_in, int(C_P // self.enc_in)
        enc_out = rearrange(enc_out, 'b (c n) d -> b c n d', b=B, n=P, c=C)
        # take cls token
        enc_out = enc_out[:, :, -1, :]

        # Output
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, (enc_in * 1) * d_model)
        output = self.classifier(output)  # (batch_size, num_classes)
        return output

    def pretrain_lead(self, x_enc, x_mark_enc, fs):
        # Embedding
        enc_out = self.enc_embedding(x_enc, fs)
        enc_out, attns_t, attns_c = self.encoder(enc_out, attn_mask=None)
        B, C_P, D = enc_out.shape
        C, P = self.enc_in, int(C_P // self.enc_in)
        enc_out = rearrange(enc_out, 'b (c n) d -> b c n d', b=B, n=P, c=C)
        # take cls token
        enc_out = enc_out[:, :, -1, :]

        # Output
        output = self.act(enc_out)
        output = self.dropout(output)
        reprs_h = output.squeeze(2)  # (batch_size, enc_in, d_model)
        reprs_h = torch.mean(reprs_h, dim=1)  # pooling among cls tokens for linear probing, (batch_size, d_model)
        output = output.reshape(output.shape[0], -1)  # (batch_size, (enc_in * 1) * d_model)
        reprs_z = self.projection_head(output)  # (batch_size, d_model)
        return reprs_h, reprs_z

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, fs=None, mask=None):
        if self.task_name == "supervised" or self.task_name == "finetune":
            return self.supervised(x_enc, x_mark_enc, fs)
        elif self.task_name == "pretrain_lead":
            return self.pretrain_lead(x_enc, x_mark_enc, fs)
        else:
            raise ValueError("Task name not recognized or not implemented within the LEADv2 model")

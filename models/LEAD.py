"""import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.LEAD_EncDec import (
    Encoder,
    EncoderLayer,
)
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import LEADEmbedding
from layers.SelfAttention_Family import LEADLayer
import numpy as np
import random
from layers.Augmentation import get_augmentation


class Model(nn.Module):
    '''
    Brainformer model (use global CLS for classification)
    '''
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.patch_len = configs.patch_len
        self.seq_len = configs.seq_len
        self.stride = configs.patch_len
        self.P = int((self.seq_len - self.patch_len) / self.stride + 2)   # must match embedding's patch_num
        self.d_model = configs.d_model
        self.enc_in = configs.enc_in
        self.use_freq_embedding = configs.use_freq_embedding
        self.use_moe = configs.use_moe

        # Embedding: must return (tokens, gcls)
        augmentations = configs.augmentations.split(",")
        # augmentations are necessary for contrastive pretraining
        if augmentations == ["none"] and "pretrain" in self.task_name:
            augmentations = ["mask0.2", "channel0.2"]

        self.enc_embedding = LEADEmbedding(
            configs.enc_in,
            configs.seq_len,
            configs.d_model,
            configs.patch_len,
            self.stride,
            augmentations,
        )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    LEADLayer(
                        configs.d_model,
                        configs.enc_in,
                        configs.n_heads,
                        self.P,
                        configs.dropout,
                        configs.output_attention,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    use_moe=configs.use_moe,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

        # Head: classify from global CLS only
        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        if self.task_name in ("supervised", "finetune"):
            self.projection = nn.Linear(configs.d_model * (configs.enc_in + self.P), configs.num_class)
            # self.projection = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)
        elif self.task_name in "pretrain_lead":
            # projection head, use for LEAD or MOCO pretraining framework, drop when using downstream
            input_dim = configs.d_model * (configs.enc_in + self.P)
            self.projection_head = nn.Sequential(
                nn.Linear(
                    input_dim,
                    input_dim * 2
                ),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(input_dim * 2, configs.d_model)
            )

    def supervised(self, x_enc, x_mark_enc, fs):
        B, T, C = x_enc.size()  # [B, T, C]
        tokens = self.enc_embedding(x_enc, fs)  # [B, C*P+1, D]
        tokens, channel_cls, patch_cls, aux_total = self.encoder(tokens, attn_mask=None)
        tokens = torch.cat([channel_cls, patch_cls], dim=1)
        tokens = tokens.reshape(B, -1)
        tokens = self.dropout(tokens)
        logits = self.projection(tokens)
        return logits, aux_total

    def pretrain_lead(self, x_enc, x_mark_enc, fs):
        B, T, C = x_enc.size()  # [B, T, C]
        tokens = self.enc_embedding(x_enc, fs)  # [B, C+P, D]
        tokens, channel_cls, patch_cls, aux_total = self.encoder(tokens, attn_mask=None)
        tokens = torch.cat([channel_cls, patch_cls], dim=1)
        tokens = tokens.reshape(B, -1)  # [B, D*(C+P)]
        repr_z = self.projection_head(tokens)  # [B, D]

        return tokens, repr_z, aux_total

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, fs=None, mask=None):
        if self.task_name in ("supervised", "finetune"):
            return self.supervised(x_enc, x_mark_enc, fs)  # [B, num_class]
        elif self.task_name in "pretrain_lead":
            return self.pretrain_lead(x_enc, x_mark_enc, fs=fs)
        else:
            raise ValueError("Task name not recognized or not implemented within the Transformer model")"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.LEAD_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import LEADLayer
from layers.Embed import LEADEmbedding
import numpy as np


class Model(nn.Module):
    """
    Model class
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.output_attention = configs.output_attention
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.cross_patch_len = configs.cross_patch_len
        self.scaled_channel_num = configs.scaled_channel_num
        self.stride = configs.cross_patch_len
        augmentations = configs.augmentations.split(",")
        # augmentations are necessary for contrastive pretraining
        if augmentations == ["none"] and "pretrain" in self.task_name:
            augmentations = ["flip", "frequency", "jitter", "mask", "channel", "drop"]

        self.enc_embedding = LEADEmbedding(
            configs.enc_in,
            configs.seq_len,
            configs.d_model,
            configs.cross_patch_len,
            configs.scaled_channel_num,
            self.stride,
            configs.dropout,
            augmentations,
        )
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    LEADLayer(
                        configs.d_model,
                        configs.n_heads,
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
        # Decoder
        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        if self.task_name == "supervised" or self.task_name == "finetune":
            # use for final classification
            self.classifier = nn.Linear(
                configs.d_model * 2,
                configs.num_class,
            )
        elif self.task_name == "pretrain_lead":
            # use for LEAD or MOCO pretraining framework
            input_dim = configs.d_model * 2
            self.projection_head = nn.Sequential(
                nn.Linear(
                    input_dim,
                    input_dim * 2
                ),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(input_dim * 2, configs.d_model)
            )

    def supervised(self, x_enc, x_mark_enc, fs):
        # Embedding
        enc_out_t, enc_out_s = self.enc_embedding(x_enc, fs)
        enc_out_t, enc_out_s, attns_t, attns_s = self.encoder(enc_out_t, enc_out_s, attn_mask=None)
        enc_out_t = enc_out_t[:, -1, :].unsqueeze(1)  # gcls from tokens
        enc_out_s = enc_out_s[:, -1, :].unsqueeze(1)  # gcls from channels
        # cat the gcls from tokens and channels
        enc_out = torch.cat([enc_out_t, enc_out_s], dim=1)

        # Output
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, 2 * d_model)
        output = self.classifier(output)  # (batch_size, num_classes)
        return output

    def pretrain(self, x_enc, x_mark_enc, fs):  # x_enc (batch_size, seq_length, enc_in)
        # Embedding
        enc_out_t, enc_out_s = self.enc_embedding(x_enc, fs)
        enc_out_t, enc_out_s, attns_t, attns_s = self.encoder(enc_out_t, enc_out_s, attn_mask=None)
        enc_out_t = enc_out_t[:, -1, :].unsqueeze(1)  # gcls from tokens
        enc_out_s = enc_out_s[:, -1, :].unsqueeze(1)  # gcls from channels
        # cat the gcls from tokens and channels
        enc_out = torch.cat([enc_out_t, enc_out_s], dim=1)

        # Output
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)

        repr_out = self.projection_head(output)  # (batch_size, repr_len)
        return output, repr_out  # first for downstream tasks, second for encoding representation

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, fs=None, mask=None):
        if self.task_name == "supervised" or self.task_name == "finetune":
            return self.supervised(x_enc, x_mark_enc, fs)
        elif self.task_name == "pretrain_lead":
            return self.pretrain(x_enc, x_mark_enc, fs)
        else:
            raise ValueError("Task name not recognized or not implemented within the LEAD model")

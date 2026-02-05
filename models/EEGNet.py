import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Conv_Blocks import TemporalSpatialConv
import numpy as np


class Model(nn.Module):

    def __init__(self, configs, f1=32, d=2, kernel_size=128):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.encoder = TemporalSpatialConv(channels=configs.enc_in, dropout=configs.dropout)

        # Decoder
        if self.task_name == 'supervised':
            self.projection = nn.Linear(128, configs.num_class)

    def supervised(self, x_enc, x_mark_enc):  # (batch_size, timestamps, enc_in)
        # conv encoder
        output = self.encoder(x_enc)

        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, fs=None, mask=None):
        if self.task_name == 'supervised':
            dec_out = self.supervised(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        else:
            raise ValueError("Task name not recognized or not implemented within the EEGNet Model")

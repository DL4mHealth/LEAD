import torch
import torch.nn as nn
from layers.Manual_Feature import feature_extractor


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.sampling_rate = list(map(int, configs.sampling_rate_list.split(",")))[0]

        self.encoder = feature_extractor

        if self.task_name == 'supervised':
            self.projection = nn.Linear(configs.enc_in*31, configs.num_class)

    def supervised(self, x_enc, x_mark_enc):
        enc_out = self.encoder(x_enc, fs=self.sampling_rate)  # (batch_size, features, enc_in)
        enc_out = enc_out.reshape(enc_out.shape[0], -1)  # (batch_size, features * enc_in)

        output = self.projection(enc_out)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, fs=None, mask=None):
        if self.task_name == "supervised":
            dec_out = self.supervised(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        else:
            raise ValueError("Task name not recognized or not implemented within the ManualFeature model")

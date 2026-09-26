import torch
import torch.nn as nn
from layers.Manual_Feature import feature_extractor


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        sampling_rate_arg = str(getattr(configs, "sampling_rate_list", "all")).strip().lower()
        if sampling_rate_arg in {"", "all", "none"}:
            self.sampling_rate = None
        else:
            self.sampling_rate = int(sampling_rate_arg.split(",")[0])

        self.encoder = feature_extractor

        if self.task_name == 'supervised':
            self.projection = nn.Linear(configs.enc_in*31, configs.num_class)

    def supervised(self, x_enc, label_id=None):
        # Manual spectral features depend on the physical sampling rate. Use
        # label_id[:, 2] so mixed-rate downstream batches remain correct.
        if label_id is None:
            if self.sampling_rate is None:
                raise ValueError(
                    "ManualFeature requires label_id[:, 2] when --sampling_rate_list=all."
                )
            enc_out = self.encoder(x_enc, fs=self.sampling_rate)
        else:
            fs_values = label_id[:, 2].long().to(x_enc.device)
            feature_parts = []
            index_parts = []
            for fs in torch.unique(fs_values, sorted=True):
                indices = torch.nonzero(fs_values == fs, as_tuple=False).squeeze(1)
                feature_parts.append(self.encoder(x_enc.index_select(0, indices), fs=int(fs.item())))
                index_parts.append(indices)
            merged_features = torch.cat(feature_parts, dim=0)
            merged_indices = torch.cat(index_parts, dim=0)
            restore_order = torch.argsort(merged_indices)
            enc_out = merged_features.index_select(0, restore_order)

        enc_out = enc_out.reshape(enc_out.shape[0], -1)
        return self.projection(enc_out)

    def forward(self, x_enc, label_id=None, mask=None, **kwargs):
        if self.task_name == "supervised":
            dec_out = self.supervised(x_enc, label_id=label_id)
            return dec_out  # [B, N]
        else:
            raise ValueError("Task name not recognized or not implemented within the ManualFeature model")

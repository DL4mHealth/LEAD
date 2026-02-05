import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math

from layers.Criss_Cross_Transformer import TransformerEncoderLayer, TransformerEncoder


class CBraMod(nn.Module):
    def __init__(self, in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12,
                    nhead=8):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_dim, out_dim, d_model, seq_len)
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True, norm_first=True,
            activation=F.gelu
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=n_layer, enable_nested_tensor=False)
        self.proj_out = nn.Sequential(
            # nn.Linear(d_model, d_model*2),
            # nn.GELU(),
            # nn.Linear(d_model*2, d_model),
            # nn.GELU(),
            nn.Linear(d_model, out_dim),
        )
        self.apply(_weights_init)

    def forward(self, x, mask=None):
        patch_emb = self.patch_embedding(x, mask)
        feats = self.encoder(patch_emb)

        out = self.proj_out(feats)

        return out

class PatchEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim, d_model, seq_len):
        super().__init__()
        self.d_model = d_model
        self.positional_encoding = nn.Sequential(
            nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(19, 7), stride=(1, 1), padding=(9, 3),
                      groups=d_model),
        )
        self.mask_encoding = nn.Parameter(torch.zeros(in_dim), requires_grad=False)
        # self.mask_encoding = nn.Parameter(torch.randn(in_dim), requires_grad=True)

        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24)),
            nn.GroupNorm(5, 25),
            nn.GELU(),

            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),

            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
        )
        self.spectral_proj = nn.Sequential(
            nn.Linear(101, d_model),
            nn.Dropout(0.1),
            # nn.LayerNorm(d_model, eps=1e-5),
        )
        # self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        # self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        # self.proj_in = nn.Sequential(
        #     nn.Linear(in_dim, d_model, bias=False),
        # )


    def forward(self, x, mask=None):
        bz, ch_num, patch_num, patch_size = x.shape
        if mask == None:
            mask_x = x
        else:
            mask_x = x.clone()
            mask_x[mask == 1] = self.mask_encoding

        mask_x = mask_x.contiguous().view(bz, 1, ch_num * patch_num, patch_size)
        patch_emb = self.proj_in(mask_x)
        patch_emb = patch_emb.permute(0, 2, 1, 3).contiguous().view(bz, ch_num, patch_num, self.d_model)

        mask_x = mask_x.contiguous().view(bz*ch_num*patch_num, patch_size)
        spectral = torch.fft.rfft(mask_x, dim=-1, norm='forward')
        spectral = torch.abs(spectral).contiguous().view(bz, ch_num, patch_num, 101)
        spectral_emb = self.spectral_proj(spectral)
        # print(patch_emb[5, 5, 5, :])
        # print(spectral_emb[5, 5, 5, :])
        patch_emb = patch_emb + spectral_emb

        positional_embedding = self.positional_encoding(patch_emb.permute(0, 3, 1, 2))
        positional_embedding = positional_embedding.permute(0, 2, 3, 1)

        patch_emb = patch_emb + positional_embedding

        return patch_emb


def _weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)



"""if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12,
                    nhead=8).to(device)
    model.load_state_dict(torch.load('../checkpoints/CBraMod/pretrain_cbramod/CBraMod/pretrained_weights.pth',
                                     map_location=device))
    a = torch.randn((128, 19, 1.5, 200)).cuda()
    b = model(a)
    print(a.shape, b.shape)"""


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.channel_mapping = nn.Conv1d(
            in_channels=configs.enc_in,
            out_channels=19,  # default 19 in CBraMod checkpoint
            kernel_size=1,
            padding='same',
            bias=False,
        )
        self.backbone = CBraMod(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30,
            n_layer=12, nhead=8
        )
        self.task_name = configs.task_name
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_path = configs.checkpoints_path
        # since the author of CBraMod does not use swa, we directly load the model here for simplicity
        if os.path.exists(model_path) and configs.is_training == 1 and configs.task_name == "supervised":
            missing_keys, unexpected_keys = self.backbone.load_state_dict(torch.load(model_path, map_location=device))
            print('Missing keys:', missing_keys)
            print('Unexpected keys:', unexpected_keys)
            print(f"Loading model successful at {model_path}")
        self.backbone.proj_out = nn.Identity()
        self.duration = math.ceil(configs.seq_len / 200)  # duration in seconds
        self.classifier = nn.Sequential(
            nn.Linear(int(19*self.duration*200), 4*200),
            nn.ELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(4*200, 200),
            nn.ELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(200, configs.num_class)
        )

    def pad_to_multiple(self, x: torch.Tensor, multiple: int = 200):
        """
        Zero-pad sequence length (dim=1) so that it's a multiple of `multiple`.
        Input shape: (batch_size, seq_len, feature_dim)
        """
        seq_len = x.size(1)
        remainder = seq_len % multiple
        if remainder != 0:
            pad_len = multiple - remainder
            x = F.pad(x, (0, 0, 0, pad_len))  # pad along seq_length dim
        return x

    def supervised(self, x_enc, x_mark_enc):  # x_enc (batch_size, seq_length, enc_in)
        # padding and channel mapping for loading CBraMod weights
        x_enc = self.pad_to_multiple(x_enc, multiple=200)  # pad to multiple of 200
        x_enc = self.channel_mapping(x_enc.permute(0, 2, 1))  # map to 19 channels (batch_size, 19, seq_length)
        x_enc = x_enc.view(x_enc.size(0), x_enc.size(1), self.duration, 200)
        batch_size, enc_in, duration, patch_length = x_enc.shape
        feats = self.backbone(x_enc)
        feats = feats.contiguous().view(batch_size, enc_in * duration * patch_length)
        out = self.classifier(feats)
        return out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, fs=None, mask=None):
        if self.task_name == "supervised" or self.task_name == "finetune":
            output = self.supervised(x_enc, x_mark_enc)
            return output
        else:
            raise ValueError("Task name not recognized or not implemented within the LEAD model")

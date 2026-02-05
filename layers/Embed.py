import copy
import math
import random
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.nn.utils import weight_norm

from layers.Augmentation import get_augmentation
from data_provider.uea import bandpass_filter_func
from utils.tools import get_eeg_coords_from_montage


SAMPLING_RATE_TO_ID = {200: 0, 100: 1, 50: 2}
NUM_SAMPLING_RATES = len(SAMPLING_RATE_TO_ID)  # = 3


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class Electrode3DEmbedding(nn.Module):
    """
    3D electrode embedding that supports any d_model.
    Automatically splits dimensions among x, y, z axes:
    d_x + d_y + d_z == d_model.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        # Automatically allocate dimensions for x, y, z
        self.d_x = d_model // 3
        self.d_y = d_model // 3
        self.d_z = d_model - self.d_x - self.d_y  # handle remainder automatically

        assert self.d_x > 0 and self.d_y > 0 and self.d_z > 0, \
            "Each axis must have at least 1 dimension."

    def _encode_axis(self, pos: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Sinusoidal encoding for one axis.
        pos: (C,)
        dim: number of embedding dimensions for this axis
        return: (C, dim)
        """
        C = pos.shape[0]
        device = pos.device
        pos = pos.unsqueeze(1)  # (C,1)

        # j indexes even dimensions
        j = torch.arange(0, dim, 2, device=device).float()  # (dim/2,)
        div_term = torch.exp(-math.log(10000.0) * j / dim)  # (dim/2,)

        angle = pos * div_term  # (C, dim/2)

        emb = torch.zeros(C, dim, device=device)
        emb[:, 0::2] = torch.sin(angle)
        emb[:, 1::2] = torch.cos(angle)

        return emb

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords: (C, 3) float electrode coordinates
        return: (C, d_model)
        """
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]

        pe_x = self._encode_axis(x, self.d_x)  # (C, d_x)
        pe_y = self._encode_axis(y, self.d_y)  # (C, d_y)
        pe_z = self._encode_axis(z, self.d_z)  # (C, d_z)

        # Concatenate per-axis embedding
        pe = torch.cat([pe_x, pe_y, pe_z], dim=-1)  # (C, d_model)
        return pe


class TokenEmbedding(nn.Module):  # (batch_size, seq_len, enc_in)
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type="fixed", freq="h"):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == "fixed" else nn.Embedding
        if freq == "t":
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = (
            self.minute_embed(x[:, :, 4]) if hasattr(self, "minute_embed") else 0.0
        )
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type="timeF", freq="h"):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != "timeF"
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = (
                self.value_embedding(x)
                + self.temporal_embedding(x_mark)
                + self.position_embedding(x)
            )
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)  # c_in is seq_length here
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)  # (batch_size, enc_in, seq_length)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)  # (batch_size, enc_in, d_model)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != "timeF"
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars


class ShallowNetEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout):
        super().__init__()

        self.shallow_net = nn.Sequential(
            nn.Conv2d(1, d_model, (1, 25), (1, 1)),
            nn.Conv2d(d_model, d_model, (c_in, 1), (1, 1)),
            nn.BatchNorm2d(d_model),
            nn.ELU(),
            nn.AvgPool2d((1, 8), (1, 4)),
            nn.Dropout(dropout),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(d_model, d_model, (1, 1), stride=(1, 1)),
        )

    def forward(self, x):  # (batch_size, seq_len, enc_in)
        x = x.permute(0, 2, 1).unsqueeze(1)  # Shape becomes (B, 1, C, T)
        x = self.shallow_net(x)
        x = self.projection(x)
        # Rearrange the output to match the Transformer input format (B, patch_num, d_model)
        x = rearrange(x, 'b d h w -> b (h w) d')
        return x


class EEG2RepEmbedding(nn.Module):
    def __init__(self, c_in, d_model, pooling_size):
        super().__init__()

        k = 7
        # Embedding Layer -----------------------------------------------------------
        self.depthwise_conv = nn.Conv2d(in_channels=1, out_channels=d_model, kernel_size=(c_in, 1))
        self.spatial_padding = nn.ReflectionPad2d((int(np.floor((k - 1) / 2)), int(np.ceil((k - 1) / 2)), 0, 0))
        self.spatialwise_conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, k))
        self.spatialwise_conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, k))
        self.SiLU = nn.SiLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(1, pooling_size), stride=(1, pooling_size))

    def forward(self, x):
        x = x.permute(0, 2, 1).unsqueeze(1)  # Shape becomes (B, 1, C, T)
        x = self.depthwise_conv(x)  # (B, d_model, 1 , T)
        x = x.transpose(1, 2)  # (B, 1, d_model, T)
        x = self.spatial_padding(x)
        x = self.spatialwise_conv1(x)  # (B, 1, d_model, T)
        x = self.SiLU(x)
        x = self.maxpool(x)  # (B, 1, d_model, T // pooling_size)
        x = self.spatial_padding(x)
        x = self.spatialwise_conv2(x)
        x = x.squeeze(1)  # (B, d_model, T // pooling_size)
        x = x.transpose(1, 2)  # (B, T // pooling_size, d_model)
        x = self.SiLU(x)

        return x


class CrossChannelTokenEmbedding(nn.Module):  # (batch_size, 1, enc_in, seq_len)
    def __init__(self, c_in, l_patch, d_model, stride=None):
        super().__init__()
        if stride is None:
            stride = l_patch
        self.tokenConv = nn.Conv2d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=(c_in, l_patch),
            stride=(1, stride),
            padding=0,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        x = self.tokenConv(x)
        return x  # (batch_size, d_model, 1, patch_num)


class UpDimensionChannelEmbedding(nn.Module):  # B x C x T
    def __init__(self, c_in, t_in, u_dim, d_model):
        super().__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.u_dim = u_dim
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=u_dim,
            kernel_size=3,
            padding=padding,
            bias=False,
        )
        self.fc = nn.Linear(t_in, d_model)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        x = self.tokenConv(x)  # B x u_dim x T
        x = self.fc(x)  # B x u_dim x d_model
        return x


class TokenChannelEmbedding(nn.Module):
    def __init__(
        self,
        enc_in,
        seq_len,
        d_model,
        patch_len_list,
        up_dim_list,
        stride_list,
        dropout,
        augmentation=["none"],
    ):
        super().__init__()
        self.patch_len_list = patch_len_list
        self.up_dim_list = up_dim_list
        self.stride_list = stride_list
        self.enc_in = enc_in
        self.paddings = [nn.ReplicationPad1d((0, stride)) for stride in stride_list]

        linear_layers_t = [
            CrossChannelTokenEmbedding(
                c_in=enc_in,
                l_patch=patch_len,
                d_model=d_model,
            )
            for patch_len in patch_len_list
        ]
        linear_layers_c = [
            UpDimensionChannelEmbedding(
                c_in=enc_in,
                t_in=seq_len,
                u_dim=u_dim,
                d_model=d_model,
            )  # c_in is seq_length here
            for u_dim in up_dim_list
        ]
        self.value_embeddings_t = nn.ModuleList(linear_layers_t)
        self.value_embeddings_c = nn.ModuleList(linear_layers_c)
        self.position_embedding_t = PositionalEmbedding(d_model=d_model)
        self.position_embedding_c = PositionalEmbedding(d_model=seq_len)
        self.dropout = nn.Dropout(dropout)
        self.augmentation = nn.ModuleList(
            [get_augmentation(aug) for aug in augmentation]
        )

        self.learnable_embeddings_t = nn.ParameterList(
            [nn.Parameter(torch.randn(1, d_model)) for _ in self.patch_len_list]
        )
        self.learnable_embeddings_c = nn.ParameterList(
            [nn.Parameter(torch.randn(1, d_model)) for _ in self.up_dim_list]
        )

    def forward(self, x):  # (batch_size, seq_len, enc_in)
        x = x.permute(0, 2, 1)  # (batch_size, enc_in, seq_len)

        x_list_t = []
        x_list_c = []
        for padding, value_embedding_t in zip(self.paddings, self.value_embeddings_t):
            x_copy = x.clone()
            # per granularity augmentation
            aug_idx = random.randint(0, len(self.augmentation) - 1)
            x_new_t = self.augmentation[aug_idx](x_copy)
            # temporal dimension
            x_new_t = padding(x_new_t).unsqueeze(1)  # (batch_size, 1, enc_in, seq_len+stride)
            x_new_t = value_embedding_t(x_new_t)  # (batch_size, d_model, 1, patch_num)
            x_new_t = x_new_t.squeeze(2).transpose(1, 2)  # (batch_size, patch_num, d_model)
            x_list_t.append(x_new_t)

        for value_embedding_c in self.value_embeddings_c:
            x_copy = x.clone()
            # per granularity augmentation
            aug_idx = random.randint(0, len(self.augmentation) - 1)
            x_new_c = self.augmentation[aug_idx](x_copy)
            # add positional embedding to tag each channel
            x_new_c = x_new_c + self.position_embedding_c(x_new_c)
            # channel dimension
            x_new_c = value_embedding_c(x_new_c)  # (batch_size, enc_in, d_model)
            x_list_c.append(x_new_c)

        x_t = [
            x + cxt + self.position_embedding_t(x)
            for x, cxt in zip(x_list_t, self.learnable_embeddings_t)
        ]  # (batch_size, patch_num_1, d_model), (batch_size, patch_num_2, d_model), ...
        x_c = [
            x + cxt
            for x, cxt in zip(x_list_c, self.learnable_embeddings_c)
        ]  # (batch_size, enc_in, d_model), (batch_size, enc_in, d_model), ...
        return x_t, x_c


class LEADEmbedding(nn.Module):
    def __init__(
        self,
        enc_in,
        seq_len,
        d_model,
        cross_patch_len,
        scaled_channel_num,
        stride,
        dropout,
        augmentation=["none"],
    ):
        super().__init__()
        self.cross_patch_len = cross_patch_len
        self.scaled_channel_num = scaled_channel_num
        self.stride = stride
        self.enc_in = enc_in
        self.padding = nn.ReplicationPad1d((0, stride))

        self.temporal_embedding = CrossChannelTokenEmbedding(
            c_in=enc_in,
            l_patch=cross_patch_len,
            d_model=d_model,
        )

        self.spatial_embedding = UpDimensionChannelEmbedding(
            c_in=enc_in,
            t_in=seq_len,
            u_dim=scaled_channel_num,
            d_model=d_model,
        )

        self.position_embedding_t = PositionalEmbedding(d_model=d_model)
        self.position_embedding_s = PositionalEmbedding(d_model=seq_len)
        self.dropout = nn.Dropout(dropout)
        self.augmentation = nn.ModuleList(
            [get_augmentation(aug) for aug in augmentation]
        )

        # sampling rate embedding for multi-scale training
        self.fs_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.fs_ln = nn.LayerNorm(d_model)

        self.temporal_cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.spatial_cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, x, fs: int = None):  # (batch_size, seq_len, enc_in)
        x = x.permute(0, 2, 1)  # (batch_size, enc_in, seq_len)

        # temporal dimension embedding
        x_copy = x.clone()
        # per granularity augmentation
        aug_idx = random.randint(0, len(self.augmentation) - 1)
        x_new_t = self.augmentation[aug_idx](x_copy)
        # temporal dimension
        x_new_t = self.padding(x_new_t).unsqueeze(1)  # (batch_size, 1, enc_in, seq_len+stride)
        x_new_t = self.temporal_embedding(x_new_t)  # (batch_size, d_model, 1, patch_num)
        x_new_t = x_new_t.squeeze(2).transpose(1, 2)  # (batch_size, patch_num, d_model)
        # concat cls token
        cls_tokens_t = self.temporal_cls.expand(x_new_t.size(0), 1, x_new_t.size(-1))  # (B,1,D)
        x_new_t = torch.cat([x_new_t, cls_tokens_t], dim=1)  # (batch_size, patch_num+1, d_model)

        # spatial dimension embedding
        x_copy = x.clone()
        # per granularity augmentation
        aug_idx = random.randint(0, len(self.augmentation) - 1)
        x_new_s = self.augmentation[aug_idx](x_copy)
        # add positional embedding to tag each channel
        x_new_s = x_new_s + self.position_embedding_s(x_new_s)
        # channel dimension
        x_new_s = self.spatial_embedding(x_new_s)  # (batch_size, scaled_channel_num, d_model)
        # concat cls token
        cls_tokens_s = self.spatial_cls.expand(x_new_s.size(0), 1, x_new_s.size(-1))  # (B,1,D)
        x_new_s = torch.cat([x_new_s, cls_tokens_s], dim=1)  # (batch_size, scaled_channel_num+1, d_model)

        # ----- sampling rate embedding -----
        if fs is not None:
            if fs.dim() == 1:
                fs = fs.unsqueeze(-1)  # [B, 1]
            fs = torch.tensor(fs, device=x.device, dtype=x.dtype)
            fs_norm = (fs.log() - 4.0) / 2.0  # rough normalization for stability (log Hz)
            fs_emb = self.fs_mlp(fs_norm)  # [B, D]
            fs_emb = self.fs_ln(fs_emb)
            # broadcast to all tokens
            x_new_t = x_new_t + fs_emb.unsqueeze(1)
            x_new_s = x_new_s + fs_emb.unsqueeze(1)

        return x_new_t, x_new_s


class LEADv2Embedding(nn.Module):
    def __init__(
        self,
        enc_in,
        seq_len,
        d_model,
        patch_len,
        stride,
        dropout,
        channel_names,
        montage_name,
        augmentation=["none"],
    ):
        super().__init__()
        self.enc_in = enc_in
        self.seq_len = seq_len
        self.d_model = d_model
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.coords = get_eeg_coords_from_montage(channel_names, montage_name=montage_name)

        # Linear projection for patch embedding
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        nn.init.xavier_uniform_(self.value_embedding.weight)
        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, 1024, d_model) * 0.02)
        # Channel Embedding
        # self.channel_token = nn.Parameter(torch.randn(1, enc_in, 1) * 0.02)
        self.channel_embedding = Electrode3DEmbedding(d_model)
        # Data augmentation modules
        self.augmentation = nn.ModuleList([get_augmentation(aug, patch_len) for aug in augmentation])
        # sampling rate embedding for multi-scale training
        self.sr_embedding = nn.Embedding(NUM_SAMPLING_RATES, d_model)
        # initialize sampling rate embedding to zero to avoid instability at the beginning
        nn.init.zeros_(self.sr_embedding.weight)

        self.dropout = nn.Dropout(dropout)

    def _pad_to_stride(self, x):
        """Pad the input so that unfolding covers the sequence evenly."""
        L = x.size(-1)
        if L < self.patch_len:
            pad_right = self.patch_len - L
        else:
            remainder = (L - self.patch_len) % self.stride
            pad_right = 0 if remainder == 0 else (self.stride - remainder)
        return F.pad(x, (0, pad_right), mode='replicate')

    def forward(self, x, fs=None):
        """Forward pass: (B, seq_len, enc_in) -> (B, enc_in * patch_num, d_model)."""
        # Change to (B, C, T)
        x = x.permute(0, 2, 1).contiguous()

        # Apply augmentation only during training
        if self.training and len(self.augmentation) > 0:
            aug_idx = torch.randint(0, len(self.augmentation), (1,), device=x.device).item()
            x = self.augmentation[aug_idx](x)

        # Dynamic padding
        x = self._pad_to_stride(x)
        # Unfold patches: (B, C, N, patch_len)
        x = x.unfold(-1, self.patch_len, self.stride)
        B, C, N, _ = x.shape
        # Linear projection: ((B*C), N, D)
        x = rearrange(x, 'b c n l -> (b c) n l')
        x = self.value_embedding(x)   # (B*C, N, D)
        x = x + self.pos_embedding[:, :N, :]  # Add positional embedding
        # reshape to (B, C, N, D)
        x = rearrange(x, '(b c) n d -> b c n d', b=B, c=C)
        # 3-D coordinate embedding
        ch_embed = self.channel_embedding(torch.tensor(self.coords, dtype=torch.float32, device=x.device))  # (C, D)
        # reshape for broadcast → (1, C, 1, D)
        ch_embed = ch_embed.view(1, C, 1, self.d_model)
        x = x + ch_embed  # (B, C, N, D)
        # flatten to (B, C*N, D)
        x = rearrange(x, 'b c n d -> b (c n) d')

        # sampling rate embedding
        if fs is not None:
            fs_ids = torch.tensor([SAMPLING_RATE_TO_ID[int(f)] for f in fs], device=x.device)
            sr_embed = self.sr_embedding(fs_ids)   # (B, D)
            sr_embed = sr_embed.unsqueeze(1)       # (B, 1, D)
            x = x + sr_embed                       # (B, C*N, D)

        return self.dropout(x)


class MultiResolutionData(nn.Module):
    def __init__(self, enc_in, resolution_list, stride_list):
        super().__init__()
        self.paddings = nn.ModuleList([nn.ReplicationPad1d((0, stride)) for stride in stride_list])

        self.multi_res = nn.ModuleList([
            nn.Conv1d(
                in_channels=enc_in,
                out_channels=enc_in,
                kernel_size=res,
                stride=res,
                padding=0,
                padding_mode='circular')
            for res in resolution_list
        ])

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x_list = []
        for l in range(len(self.multi_res)):
            out = self.paddings[l](x)
            out = self.multi_res[l](out)
            x_list.append(out)
        return x_list


class FrequencyEmbedding(nn.Module):
    def __init__(self, d_model, res_len, augmentation=["none"]):
        super().__init__()
        self.d_model = d_model
        self.embeddings = nn.ModuleList([
            nn.Linear(int(res/2)+1, int(self.d_model/2)+1).to(torch.cfloat)
            for res in res_len
        ])

        self.augmentation = nn.ModuleList(
            [get_augmentation(aug) for aug in augmentation]
        )

    def forward(self, x_list):
        x_out = []
        for l in range(len(x_list)):
            x = torch.fft.rfft(x_list[l], dim=-1)
            out = self.embeddings[l](x)
            out = torch.fft.irfft(out, dim=-1, n=self.d_model)

            aug_idx = random.randint(0, len(self.augmentation) - 1)
            out = self.augmentation[aug_idx](out)
            x_out.append(out)

        return x_out

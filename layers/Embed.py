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


class PositionalEmbedding(nn.Module):
    """Fixed sinusoidal positional embedding.

    This implementation supports both even and odd ``d_model`` and is used by
    LEAD when temporal/channel positional embeddings are configured as fixed.
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        d_model = int(d_model)
        max_len = int(max_len)
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)

        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-(math.log(10000.0) / d_model))
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return self.pe[:, : x.size(1)].to(device=x.device, dtype=x.dtype)

    def get(self, length, device=None, dtype=None):
        if length > self.pe.shape[1]:
            raise ValueError(
                f"Requested {length} fixed positions, but max_len={self.pe.shape[1]}."
            )
        out = self.pe[:, :length]
        if device is not None or dtype is not None:
            out = out.to(device=device or out.device, dtype=dtype or out.dtype)
        return out


class Electrode3DEmbedding(nn.Module):
    """Legacy LEADv2 3D electrode embedding.

    Raw montage xyz coordinates are encoded independently with sinusoidal
    functions and concatenated. No learnable projection or normalization is
    applied, matching the original LEADv2 model used by the legacy checkpoint.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = int(d_model)
        self.d_x = self.d_model // 3
        self.d_y = self.d_model // 3
        self.d_z = self.d_model - self.d_x - self.d_y
        if min(self.d_x, self.d_y, self.d_z) <= 0:
            raise ValueError("d_model must be at least 3 for 3D electrode embedding.")

    @staticmethod
    def _encode_axis(pos: torch.Tensor, dim: int) -> torch.Tensor:
        C = pos.shape[0]
        device, dtype = pos.device, pos.dtype
        pos = pos.unsqueeze(1)
        num_freqs = (dim + 1) // 2
        j = torch.arange(num_freqs, device=device, dtype=dtype)
        div_term = torch.exp(-math.log(10000.0) * (2 * j) / max(dim, 1))
        angle = pos * div_term
        emb = torch.zeros(C, dim, device=device, dtype=dtype)
        emb[:, 0::2] = torch.sin(angle[:, :emb[:, 0::2].shape[1]])
        emb[:, 1::2] = torch.cos(angle[:, :emb[:, 1::2].shape[1]])
        return emb

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        pe_x = self._encode_axis(coords[:, 0], self.d_x)
        pe_y = self._encode_axis(coords[:, 1], self.d_y)
        pe_z = self._encode_axis(coords[:, 2], self.d_z)
        return torch.cat([pe_x, pe_y, pe_z], dim=-1)


class EEGDeformerConv2dWithConstraint(nn.Conv2d):
    """Conv2d layer with a DataParallel-safe max-norm weight constraint.

    Do not modify ``self.weight`` in-place inside ``forward``. Under
    ``torch.nn.DataParallel``, broadcast parameters can be views tracked by
    autograd, and in-place renorm can trigger:

        RuntimeError: Output ... of BroadcastBackward0 is a view and its base
        or another view of its base has been modified inplace.

    Instead, build a normalized temporary weight tensor and pass it to
    ``F.conv2d``. Gradients still flow back to ``self.weight`` while avoiding
    the in-place parameter update.
    """

    def __init__(self, *args, max_norm=1.0, do_weight_norm=True, **kwargs):
        self.max_norm = max_norm
        self.do_weight_norm = do_weight_norm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if self.do_weight_norm:
            weight = torch.renorm(
                self.weight,
                p=2,
                dim=0,
                maxnorm=self.max_norm,
            )
        else:
            weight = self.weight

        return F.conv2d(
            x,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class EEGDeformerEmbedding(nn.Module):
    """
    Shallow convolutional feature encoder used by EEG-Deformer.

    Input:
        x: [B, T, C]

    Output:
        tokens: [B, D, F]
            D = d_model convolutional feature tokens / filters
            F = temporal feature dimension after temporal/spatial convolution and pooling
    """

    def __init__(self, num_channels, num_time, temporal_kernel, d_model, dropout=0.1):
        super().__init__()
        temporal_kernel = int(temporal_kernel)
        if temporal_kernel % 2 == 0:
            temporal_kernel += 1

        self.num_channels = int(num_channels)
        self.num_time = int(num_time)
        self.temporal_kernel = temporal_kernel
        self.num_kernel = int(d_model)
        self.feature_dim = max(1, self.num_time // 2)

        self.cnn_encoder = nn.Sequential(
            EEGDeformerConv2dWithConstraint(
                1,
                self.num_kernel,
                kernel_size=(1, self.temporal_kernel),
                padding=(0, self.temporal_kernel // 2),
                max_norm=2.0,
                bias=True,
            ),
            EEGDeformerConv2dWithConstraint(
                self.num_kernel,
                self.num_kernel,
                kernel_size=(self.num_channels, 1),
                padding=0,
                max_norm=2.0,
                bias=True,
            ),
            nn.BatchNorm2d(self.num_kernel),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(dropout),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_kernel, self.feature_dim) * 0.02)

    def forward(self, x):
        # [B, T, C] -> [B, 1, C, T]
        x = x.permute(0, 2, 1).unsqueeze(1).contiguous()
        x = self.cnn_encoder(x)  # [B, K, 1, F]
        x = x.squeeze(2)         # [B, K, F]

        # For rare odd-length inputs, MaxPool2d may produce floor(T/2). Keep the
        # positional embedding aligned with the actual runtime feature length.
        feature_len = x.shape[-1]
        if feature_len <= self.pos_embedding.shape[-1]:
            pos = self.pos_embedding[:, :, :feature_len]
        else:
            pos = F.interpolate(
                self.pos_embedding,
                size=feature_len,
                mode="linear",
                align_corners=False,
            )
        return x + pos




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
    """Dataset-aware LEAD patch embedding with configurable positional encodings.

    Temporal position:
      - ``fixed``: sinusoidal encoding over patch indices.
      - ``learnable``: learned patch-position table.

    Channel position:
      - ``fixed``: sinusoidal encoding over runtime channel indices.
      - ``learnable``: learned channel-position table.
      - ``3d``: dataset-aware 3D electrode coordinate embedding.

    For ``3d`` channel position, channel names and montage are routed by
    ``dataset_id``. Coordinate buffers are runtime metadata and intentionally
    excluded from checkpoints.
    """

    def __init__(
        self,
        d_model,
        patch_len,
        stride,
        dropout,
        channel_names_by_id,
        montage_by_id,
        sampling_rate_to_id=None,
        use_sampling_embedding=False,
        augmentation=("none",),
        max_patch_positions=1024,
        temporal_pos_type="learnable",
        channel_pos_type="3D",
        max_channel_positions=368,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.max_patch_positions = int(max_patch_positions)
        self.max_channel_positions = int(max_channel_positions)
        self.temporal_pos_type = str(temporal_pos_type).lower()
        raw_channel_pos_type = str(channel_pos_type)
        self.channel_pos_type = "3D" if raw_channel_pos_type.lower() == "3d" else raw_channel_pos_type.lower()
        self.use_sampling_embedding = bool(use_sampling_embedding)

        if self.temporal_pos_type not in {"fixed", "learnable"}:
            raise ValueError(
                "temporal_pos_type must be one of {'fixed', 'learnable'}, got "
                f"{temporal_pos_type!r}."
            )
        if self.channel_pos_type not in {"fixed", "learnable", "3D"}:
            raise ValueError(
                "channel_pos_type must be one of {'fixed', 'learnable', '3D'}, got "
                f"{channel_pos_type!r}."
            )
        if self.max_patch_positions <= 0:
            raise ValueError("max_patch_positions must be positive.")
        if self.max_channel_positions <= 0:
            raise ValueError("max_channel_positions must be positive.")

        # Sampling-rate embeddings are optional. When enabled, parameters are
        # keyed by the physical Hz value rather than a dataset-local integer ID,
        # which keeps checkpoint transfer stable across different rate sets.
        sampling_rate_to_id = dict(sampling_rate_to_id or {})
        self.sampling_rates = sorted(int(k) for k in sampling_rate_to_id.keys())
        if self.use_sampling_embedding and not self.sampling_rates:
            raise ValueError(
                "use_sampling_embedding=True requires at least one sampling rate."
            )

        self.value_embedding = nn.Linear(self.patch_len, self.d_model, bias=False)
        nn.init.xavier_uniform_(self.value_embedding.weight)

        if self.temporal_pos_type == "learnable":
            # Keep the historical parameter name ``pos_embedding`` so existing
            # LEAD checkpoints remain loadable when using the default learnable
            # temporal positional embedding.
            self.pos_embedding = nn.Parameter(
                torch.randn(1, self.max_patch_positions, self.d_model) * 0.02
            )

        self.coord_buffer_names = {}
        if self.channel_pos_type == "3D":
            self.channel_embedding = Electrode3DEmbedding(self.d_model)
            if not channel_names_by_id:
                raise ValueError(
                    "channel_pos_type='3D' requires channel_names_by_id from dataset meta.json."
                )
            for dataset_id, channel_names in channel_names_by_id.items():
                dataset_id = int(dataset_id)
                channel_names = list(channel_names or [])
                montage_name = montage_by_id.get(dataset_id, None)
                coords = get_eeg_coords_from_montage(channel_names, montage_name)
                buffer_name = f"electrode_coords_dataset_{dataset_id}"
                self.register_buffer(
                    buffer_name,
                    torch.tensor(coords, dtype=torch.float32),
                    persistent=False,
                )
                self.coord_buffer_names[dataset_id] = buffer_name
        elif self.channel_pos_type == "learnable":
            self.channel_pos_embedding = nn.Parameter(
                torch.randn(1, self.max_channel_positions, self.d_model) * 0.02
            )

        self.augmentation = nn.ModuleList(
            [get_augmentation(aug, self.patch_len) for aug in augmentation]
        )
        if self.use_sampling_embedding:
            self.sr_embeddings = nn.ParameterDict({
                f"hz_{rate}": nn.Parameter(torch.zeros(self.d_model))
                for rate in self.sampling_rates
            })
        else:
            self.sr_embeddings = None
        self.dropout = nn.Dropout(dropout)

    def _pad_to_stride(self, x):
        L = x.size(-1)
        if L < self.patch_len:
            pad_right = self.patch_len - L
        else:
            remainder = (L - self.patch_len) % self.stride
            pad_right = 0 if remainder == 0 else self.stride - remainder
        return F.pad(x, (0, pad_right), mode="replicate")

    def _sampling_rate_vectors(self, fs, device, dtype):
        if not self.use_sampling_embedding or self.sr_embeddings is None:
            raise RuntimeError(
                "Sampling-rate vectors were requested while sampling embedding is disabled."
            )
        fs_values = [int(v) for v in fs.detach().cpu().tolist()]
        missing = sorted({v for v in fs_values if f"hz_{v}" not in self.sr_embeddings})
        if missing:
            raise ValueError(
                f"Sampling rates {missing} are not present in the metadata-derived "
                f"rate set {self.sampling_rates}."
            )
        return torch.stack([
            self.sr_embeddings[f"hz_{value}"].to(device=device, dtype=dtype)
            for value in fs_values
        ], dim=0)

    def _fixed_sinusoidal_position(self, length, device, dtype):
        """Build standard sin/cos positions dynamically with no learned parameters."""
        position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device, dtype=dtype)
            * (-(math.log(10000.0) / self.d_model))
        )
        pe = torch.zeros(length, self.d_model, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        if self.d_model > 1:
            pe[:, 1::2] = torch.cos(
                position * div_term[: pe[:, 1::2].shape[1]]
            )
        return pe.unsqueeze(0)

    def _temporal_position(self, N, device, dtype):
        if self.temporal_pos_type == "learnable":
            if N > self.max_patch_positions:
                raise ValueError(
                    f"Runtime patch count N={N} exceeds LEAD max_patch_positions="
                    f"{self.max_patch_positions}. Increase --max_patch_positions or use a shorter sequence."
                )
            return self.pos_embedding[:, :N, :].to(device=device, dtype=dtype)
        return self._fixed_sinusoidal_position(N, device, dtype)

    def _channel_position(self, C, current_dataset_id, device, dtype):
        if self.channel_pos_type == "3D":
            if current_dataset_id not in self.coord_buffer_names:
                raise KeyError(
                    f"No electrode metadata registered for dataset_id={current_dataset_id}. "
                    f"Available IDs: {sorted(self.coord_buffer_names)}"
                )
            coords = getattr(self, self.coord_buffer_names[current_dataset_id]).to(
                device=device, dtype=dtype
            )
            if coords.shape[0] != C:
                raise ValueError(
                    f"Dataset_id={current_dataset_id}: batch has C={C}, but metadata provides "
                    f"{coords.shape[0]} channel coordinates."
                )
            return self.channel_embedding(coords).unsqueeze(0)  # [1, C, D]

        if self.channel_pos_type == "learnable":
            if C > self.max_channel_positions:
                raise ValueError(
                    f"Runtime channel count C={C} exceeds LEAD max_channel_positions="
                    f"{self.max_channel_positions}. Increase --max_channel_positions."
                )
            return self.channel_pos_embedding[:, :C, :].to(device=device, dtype=dtype)
        return self._fixed_sinusoidal_position(C, device, dtype)

    def forward(self, x, dataset_id, fs=None, apply_augmentation=True):
        """
        Args:
            x: [B, T, C]
            dataset_id: [B], compatible datasets may coexist in one batch
            fs: [B] sampling frequency
            apply_augmentation: whether to apply one configured augmentation

        Returns:
            tokens: [B, C*N, D] in channel-major patch order
        """
        dataset_id_long = dataset_id.long()
        unique_dataset_ids = torch.unique(dataset_id_long.detach()).cpu().tolist()

        x = x.permute(0, 2, 1).contiguous()  # [B, C, T]
        # Augmentation is controlled explicitly by the caller. This is intentionally
        # independent of module.train()/eval() so probe training can augment inputs
        # while keeping the frozen backbone in eval mode.
        if apply_augmentation and len(self.augmentation) > 0:
            aug_idx = torch.randint(0, len(self.augmentation), (1,), device=x.device).item()
            x = self.augmentation[aug_idx](x)

        x = self._pad_to_stride(x)
        x = x.unfold(-1, self.patch_len, self.stride)  # [B, C, N, patch_len]
        B, C, N, _ = x.shape

        x = rearrange(x, "b c n l -> (b c) n l")
        x = self.value_embedding(x)
        x = x + self._temporal_position(N, x.device, x.dtype)
        x = rearrange(x, "(b c) n d -> b c n d", b=B, c=C)

        if self.channel_pos_type == "3D":
            # Route 3D coordinates per dataset even when multiple compatible
            # datasets share a batch. This preserves the original dataset_id
            # metadata instead of silently reusing one dataset's coordinates.
            channel_pe = torch.zeros(B, C, self.d_model, device=x.device, dtype=x.dtype)
            for dataset_value in unique_dataset_ids:
                current_dataset_id = int(dataset_value)
                pe = self._channel_position(
                    C, current_dataset_id, x.device, x.dtype
                ).squeeze(0)
                mask = (dataset_id_long == current_dataset_id).to(x.dtype).view(B, 1, 1)
                channel_pe = channel_pe + mask * pe.unsqueeze(0)
            x = x + channel_pe.unsqueeze(2)  # [B, C, 1, D]
        else:
            # Fixed/learnable channel embeddings depend only on channel index.
            channel_pe = self._channel_position(C, 0, x.device, x.dtype)
            x = x + channel_pe.unsqueeze(2)  # [1, C, 1, D]

        x = rearrange(x, "b c n d -> b (c n) d")

        if self.use_sampling_embedding:
            if fs is None:
                raise ValueError(
                    "Sampling frequency labels are required when --use_sampling_embedding is enabled."
                )
            fs_embed = self._sampling_rate_vectors(fs, x.device, x.dtype)
            x = x + fs_embed.unsqueeze(1)

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

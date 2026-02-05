from collections import defaultdict
from torch.nn import functional as Fun
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import copy
from typing import Optional, Any, Union, Callable
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


class CSBrain_TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, bias: bool = True,
                 area_config: dict = {}, sorted_indices: list = []):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first

        self.inter_region_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                                       bias=bias, batch_first=batch_first)

        self.inter_window_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                                       bias=bias, batch_first=batch_first)

        self.global_fc = nn.Linear(d_model, d_model, bias=bias)

        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        if isinstance(activation, str):
            activation = getattr(F, activation, F.relu)
        self.activation = activation

        self.area_config = area_config
        self.mask_builder = None
        self.region_attn_mask = None
        self.region_indices_dict = None

        if area_config is not None:
            total_channels = sum(len(range(info['slice'].start or 0, info['slice'].stop, info['slice'].step or 1))
                                 if isinstance(info['slice'], slice) else len(info['slice'])
                                 for info in area_config.values())

            self.mask_builder = RegionAttentionMaskBuilder(total_channels, area_config)
            self.region_attn_mask = self.mask_builder.get_mask()
            self.region_indices_dict = self.mask_builder.get_region_indices()

    def forward(
        self,
        src: torch.Tensor,
        area_config: Optional[dict] = None,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = src
        x = x + self._inter_window_attention(self.norm1(x), src_mask, src_key_padding_mask)

        if self.region_attn_mask is None and area_config is not None:
            x = x + self._inter_region_attention_dynamic(self.norm2(x), area_config, src_mask, src_key_padding_mask)
        else:
            x = x + self._inter_region_attention_static(self.norm2(x), src_mask, src_key_padding_mask)

        x = x + self._ff_block(self.norm3(x))
        return x

    def _inter_region_attention_static(self, x: torch.Tensor,
                                       attn_mask: Optional[torch.Tensor] = None,
                                       key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.region_attn_mask is None or self.region_indices_dict is None:
            raise ValueError("no initialized region attention mask or region indices dictionary")

        batch, chans, T, F = x.shape

        x_reshaped = x.permute(0, 2, 1, 3)
        x_flat = x_reshaped.reshape(batch * T, chans, F)

        region_global_features = {}
        for region_name, region_indices in self.region_indices_dict.items():
            region_x = x[:, region_indices, :, :]
            region_global = region_x.mean(dim=1, keepdim=True)
            region_global_features[region_name] = region_global

        global_features = torch.zeros_like(x_flat)

        for region_name, region_indices in self.region_indices_dict.items():
            region_global = region_global_features[region_name]
            region_global = region_global.permute(0, 2, 1, 3)
            region_global = region_global.reshape(batch * T, 1, F)

            for idx in region_indices:
                global_features[:, idx:idx + 1, :] = region_global

        global_features = self.global_fc(global_features)
        x_enhanced = x_flat + global_features

        region_attn_mask = self.region_attn_mask.to(x.device)

        attn_output = self.inter_region_attn(
            x_enhanced, x_enhanced, x_enhanced,
            attn_mask=region_attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )[0]

        attn_output = attn_output.reshape(batch, T, chans, F).permute(0, 2, 1, 3)

        return self.dropout1(attn_output)

    def _inter_window_attention(self, x: torch.Tensor,
                                attn_mask: Optional[torch.Tensor] = None,
                                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, chans, T, Fea = x.shape
        window_size = min(T, 5)

        num_windows = T // window_size
        original_T = T

        if T % window_size != 0:
            pad_length = window_size - (T % window_size)
            x = F.pad(x, (0, 0, 0, pad_length))
            T = T + pad_length
            num_windows = T // window_size

        x = x.view(batch, chans, num_windows, window_size, Fea)

        x = x.permute(0, 3, 1, 2, 4)
        x = x.reshape(batch * window_size * chans, num_windows, Fea)

        temporal_attn_mask = None
        if attn_mask is not None:
            if isinstance(attn_mask, torch.Tensor) and attn_mask.dim() == 2:
                temporal_attn_mask = torch.triu(
                    torch.ones(num_windows, num_windows, device=x.device) * float('-inf'),
                    diagonal=1
                )

        x = self.inter_window_attn(
            x, x, x,
            attn_mask=temporal_attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )[0]

        x = x.reshape(batch, window_size, chans, num_windows, Fea)
        x = x.permute(0, 2, 3, 1, 4)

        x = x.reshape(batch, chans, T, Fea)
        if T != original_T:
            x = x[:, :, :original_T, :]

        return self.dropout2(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, F = x.shape
        x_reshaped = x.permute(0, 2, 1, 3).reshape(B * T, C, F)

        x_ff = self.linear2(self.dropout(self.activation(self.linear1(x_reshaped))))

        x_ff = x_ff.reshape(B, T, C, F).permute(0, 2, 1, 3)

        return self.dropout3(x_ff)


class RegionAttentionMaskBuilder:
    def __init__(self, num_channels: int, area_config: dict, device=None):
        self.num_channels = num_channels
        self.area_config = area_config
        self.device = device

        self.region_indices_dict = self._process_region_indices()

        self.attention_mask = self._build_attention_mask()

    def _process_region_indices(self):
        region_indices_dict = {}

        for region_name, region_info in self.area_config.items():
            region_slice = region_info['slice']
            if isinstance(region_slice, slice):
                start = region_slice.start or 0
                stop = region_slice.stop
                step = region_slice.step or 1
                region_indices = list(range(start, stop, step))
            else:
                region_indices = list(region_slice)

            region_indices_dict[region_name] = region_indices

        return region_indices_dict

    def _build_attention_mask(self):
        device = self.device if self.device is not None else torch.device('cpu')
        region_attn_mask = torch.ones(self.num_channels, self.num_channels, device=device) * float('-inf')

        num_groups = max(len(indices) for indices in self.region_indices_dict.values())

        groups = [[] for _ in range(num_groups)]

        for g in range(num_groups):
            for region_name, region_indices in self.region_indices_dict.items():
                n_electrodes = len(region_indices)
                if n_electrodes == 0:
                    continue

                electrode_idx = region_indices[g % n_electrodes]
                groups[g].append(electrode_idx)

        for g, group_electrodes in enumerate(groups):
            for idx1 in group_electrodes:
                for idx2 in group_electrodes:
                    region_attn_mask[idx1, idx2] = 0

        return region_attn_mask

    def get_mask(self):
        return self.attention_mask

    def get_region_indices(self):
        return self.region_indices_dict


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_seq_len(
        src: Tensor,
        batch_first: bool
) -> Optional[int]:

    if src.is_nested:
        return None
    else:
        src_size = src.size()
        if len(src_size) == 2:
            return src_size[0]
        else:
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]


def _detect_is_causal_mask(
        mask: Optional[Tensor],
        is_causal: Optional[bool] = None,
        size: Optional[int] = None,
) -> bool:
    make_causal = (is_causal is True)

    if is_causal is None and mask is not None:
        sz = size if size is not None else mask.size(-2)
        causal_comparison = _generate_square_subsequent_mask(
            sz, device=mask.device, dtype=mask.dtype)

        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all())
        else:
            make_causal = False

    return make_causal


def _generate_square_subsequent_mask(
        sz: int,
        device: torch.device = torch.device(torch._C._get_default_device()),
        dtype: torch.dtype = torch.get_default_dtype(),
) -> Tensor:
    return torch.triu(
        torch.full((sz, sz), float('-inf'), dtype=dtype, device=device),
        diagonal=1,
    )


def cast_tuple(val, length=1):
    return val if isinstance(val, tuple) else ((val,) * length)


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None, enable_nested_tensor=True, mask_check=True):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
            self,
            src: Tensor,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: Optional[bool] = None) -> Tensor:

        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class CSBrain_TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None, enable_nested_tensor=True, mask_check=True):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
            self,
            src: Tensor,
            area_config: dict,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: Optional[bool] = None) -> Tensor:

        output = src  # [128, 19, 30, 200]
        for mod in self.layers:
            output = mod(output, area_config, src_mask=mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TemEmbedEEGLayer(nn.Module):
    def __init__(
            self,
            dim_in,
            dim_out,
            kernel_sizes,
            stride=1
    ):
        super().__init__()
        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)

        dim_scales = [int(dim_out / (2 ** i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=dim_in, out_channels=dim_scale, kernel_size=(kt, 1),
                      stride=(stride, 1), padding=((kt - 1) // 2, 0))
            for (kt,), dim_scale in zip(kernel_sizes, dim_scales)
        ])

    def forward(self, x):
        batch, chans, time, d_model = x.shape

        x = x.view(batch * chans, d_model, time, 1)

        fmaps = [conv(x) for conv in self.convs]

        assert all(f.shape[2] == time for f in fmaps), "Time dimension mismatch after convolutions!"

        x = torch.cat(fmaps, dim=1)

        x = x.view(batch, chans, time, -1)

        return x


class BrainEmbedEEGLayer(nn.Module):
    def __init__(self, dim_in=200, dim_out=200, total_regions=5):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.total_regions = total_regions

        kernel_sizes = [1, 3, 5]

        dim_scales = [dim_out // (2 ** (i + 1)) for i in range(len(kernel_sizes) - 1)]
        dim_scales.append(dim_out - sum(dim_scales))

        self.region_blocks = nn.ModuleDict({
            f"region_{i}": nn.ModuleList([
                nn.Conv2d(
                    in_channels=dim_in,
                    out_channels=dim_scale,
                    kernel_size=(k, 1),
                    padding=(0, 0),
                    groups=1
                ) for k, dim_scale in zip(kernel_sizes, dim_scales)
            ])
            for i in range(total_regions)
        })

    def forward(self, x, area_config):
        batch, chans, T, F = x.shape
        device = x.device

        output = torch.zeros((batch, chans, T, self.dim_out), device=device)

        for region_key, region_info in area_config.items():
            if region_key not in self.region_blocks:
                continue

            channel_slice = region_info['slice']
            n_electrodes = region_info['channels']

            x_region = x[:, channel_slice, :, :]

            x_trans = x_region.permute(0, 2, 1, 3).reshape(-1, n_electrodes, F)
            x_trans = x_trans.permute(0, 2, 1).unsqueeze(-1)

            fmap_outputs = []
            for conv, k in zip(self.region_blocks[region_key], [1, 3, 5]):
                pad_size = (k - 1) // 2

                if n_electrodes == 1:
                    x_padded = Fun.pad(x_trans, (0, 0, pad_size, pad_size), mode='constant', value=0)
                else:
                    x_padded = Fun.pad(x_trans, (0, 0, pad_size, pad_size), mode='circular')

                fmap_outputs.append(conv(x_padded))

            fmap_cat = torch.cat(fmap_outputs, dim=1)
            fmap_out = fmap_cat.squeeze(-1).permute(0, 2, 1).reshape(batch, T, n_electrodes, self.dim_out)
            fmap_out = fmap_out.permute(0, 2, 1, 3)

            output[:, channel_slice, :, :] = fmap_out

        return output


class BrainAreaConv(nn.Module):
    def __init__(self, area_config):
        super().__init__()
        self.conv_layers = nn.ModuleDict({
            name: nn.Conv2d(
                in_channels=cfg['channels'],
                out_channels=cfg['channels'],
                kernel_size=(1, 1),
                padding=(0, 0)
            ) for name, cfg in area_config.items()
        })
        self.area_config = area_config

    def forward(self, x):
        outputs = []
        for name, cfg in self.area_config.items():
            x_area = x[:, cfg['slice'], :, :]
            conv = self.conv_layers[name]
            outputs.append(conv(x_area))
        return torch.cat(outputs, dim=1)


# transformer classes

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


def FeedForward(dim, mult=4, dropout=0.):
    return nn.Sequential(
        LayerNorm(dim),
        nn.Conv2d(dim, dim * mult, 1),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Conv2d(dim * mult, dim, 1)
    )


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return Fun.relu
    elif activation == "gelu":
        return Fun.gelu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_seq_len(
        src: Tensor,
        batch_first: bool
) -> Optional[int]:
    if src.is_nested:
        return None
    else:
        src_size = src.size()
        if len(src_size) == 2:
            # unbatched: S, E
            return src_size[0]
        else:
            # batched: B, S, E if batch_first else S, B, E
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]


def _detect_is_causal_mask(
        mask: Optional[Tensor],
        is_causal: Optional[bool] = None,
        size: Optional[int] = None,
) -> bool:
    """Return whether the given attention mask is causal.

    Warning:
    If ``is_causal`` is not ``None``, its value will be returned as is.  If a
    user supplies an incorrect ``is_causal`` hint,

    ``is_causal=False`` when the mask is in fact a causal attention.mask
       may lead to reduced performance relative to what would be achievable
       with ``is_causal=True``;
    ``is_causal=True`` when the mask is in fact not a causal attention.mask
       may lead to incorrect and unpredictable execution - in some scenarios,
       a causal mask may be applied based on the hint, in other execution
       scenarios the specified mask may be used.  The choice may not appear
       to be deterministic, in that a number of factors like alignment,
       hardware SKU, etc influence the decision whether to use a mask or
       rely on the hint.
    ``size`` if not None, check whether the mask is a causal mask of the provided size
       Otherwise, checks for any causal mask.
    """
    # Prevent type refinement
    make_causal = (is_causal is True)

    if is_causal is None and mask is not None:
        sz = size if size is not None else mask.size(-2)
        causal_comparison = _generate_square_subsequent_mask(
            sz, device=mask.device, dtype=mask.dtype)

        # Do not use `torch.equal` so we handle batched masks by
        # broadcasting the comparison.
        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all())
        else:
            make_causal = False

    return make_causal


def _generate_square_subsequent_mask(
        sz: int,
        device: torch.device = torch.device(torch._C._get_default_device()),  # torch.device('cpu'),
        dtype: torch.dtype = torch.get_default_dtype(),
) -> Tensor:
    r"""Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    return torch.triu(
        torch.full((sz, sz), float('-inf'), dtype=dtype, device=device),
        diagonal=1,
    )




class CSBrain(nn.Module):
    def __init__(self, in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12,
                 nhead=8, TemEmbed_kernel_sizes=[(1,), (3,), (5,)], brain_regions=[], sorted_indices=[]):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_dim, out_dim, d_model, seq_len)

        self.TemEmbed_kernel_sizes = TemEmbed_kernel_sizes
        kernel_sizes = self.TemEmbed_kernel_sizes
        self.TemEmbedEEGLayer = TemEmbedEEGLayer(dim_in=in_dim, dim_out=out_dim, kernel_sizes=kernel_sizes, stride=1)

        self.brain_regions = brain_regions
        self.area_config = generate_area_config(sorted(brain_regions))
        self.BrainEmbedEEGLayer = BrainEmbedEEGLayer(dim_in=in_dim, dim_out=out_dim)
        self.sorted_indices = sorted_indices

        encoder_layer = CSBrain_TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, area_config=self.area_config, sorted_indices=self.sorted_indices, batch_first=True,
            activation=F.gelu
        )
        self.encoder = CSBrain_TransformerEncoder(encoder_layer, num_layers=n_layer, enable_nested_tensor=False)

        self.proj_out = nn.Sequential(
            nn.Linear(d_model, out_dim),
        )
        self.apply(_weights_init)

        self.features_by_layer = []
        self.input_features = []

    def forward(self, x, mask=None):
        x = x[:, self.sorted_indices, :, :]

        patch_emb = self.patch_embedding(x, mask)

        for layer_idx in range(self.encoder.num_layers):
            patch_emb = self.TemEmbedEEGLayer(patch_emb) + patch_emb
            patch_emb = self.BrainEmbedEEGLayer(patch_emb, self.area_config) + patch_emb

            patch_emb = self.encoder.layers[layer_idx](patch_emb, self.area_config)

        out = self.proj_out(patch_emb)

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
            nn.Linear(d_model // 2 + 1, d_model),
            nn.Dropout(0.1),
        )

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

        mask_x = mask_x.contiguous().view(bz * ch_num * patch_num, patch_size)
        spectral = torch.fft.rfft(mask_x, dim=-1, norm='forward')
        spectral = torch.abs(spectral).contiguous().view(bz, ch_num, patch_num, mask_x.shape[1] // 2 + 1)
        spectral_emb = self.spectral_proj(spectral)
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


def generate_area_config(brain_regions):
    region_to_channels = defaultdict(list)
    for channel_idx, region in enumerate(brain_regions):
        region_to_channels[region].append(channel_idx)

    area_config = {}
    for region, channels in region_to_channels.items():
        area_config[f'region_{region}'] = {
            'channels': len(channels),
            'slice': slice(channels[0], channels[-1] + 1)
        }
    return area_config


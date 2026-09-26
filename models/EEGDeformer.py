import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import EEGDeformerEmbedding
from layers.SelfAttention_Family import FullAttention, AttentionLayer


class EEGDeformerFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class EEGDeformerHCTLayer(nn.Module):
    """
    One Hierarchical Coarse-to-Fine Transformer (HCT) layer.

    The coarse branch halves the temporal feature dimension and applies
    self-attention across convolutional feature tokens. The fine branch applies
    temporal convolution plus pooling. A purified dense feature is extracted
    from the fine branch and later concatenated to the final embedding.
    """

    def __init__(
        self,
        input_dim,
        num_kernel,
        n_heads,
        mlp_dim,
        temporal_kernel,
        dropout=0.1,
        factor=1,
        output_attention=False,
    ):
        super().__init__()
        if input_dim < 2:
            raise ValueError("EEGDeformerHCTLayer requires input_dim >= 2.")

        self.output_dim = max(1, input_dim // 2)
        heads = max(1, min(int(n_heads), self.output_dim))

        self.coarse_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.attention = AttentionLayer(
            FullAttention(
                False,
                factor=factor,
                attention_dropout=dropout,
                output_attention=output_attention,
            ),
            self.output_dim,
            heads,
        )
        self.feed_forward = EEGDeformerFeedForward(
            self.output_dim,
            hidden_dim=max(int(mlp_dim), self.output_dim),
            dropout=dropout,
        )

        temporal_kernel = int(temporal_kernel)
        if temporal_kernel % 2 == 0:
            temporal_kernel += 1
        self.fine_branch = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(
                in_channels=num_kernel,
                out_channels=num_kernel,
                kernel_size=temporal_kernel,
                padding=temporal_kernel // 2,
            ),
            nn.BatchNorm1d(num_kernel),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

    @staticmethod
    def _purify_info(x):
        # Dense Information Purification: log power over the temporal dimension.
        return torch.log(torch.mean(x.pow(2), dim=-1).clamp_min(1e-6))

    def forward(self, x):
        # x: [B, K, F]
        coarse = self.coarse_pool(x)  # [B, K, F/2]
        attn_out, _ = self.attention(coarse, coarse, coarse, attn_mask=None)
        coarse = coarse + attn_out

        fine = self.fine_branch(x)  # [B, K, F/2]
        dense_info = self._purify_info(fine)  # [B, K]

        x = self.feed_forward(coarse) + fine
        return x, dense_info


class EEGDeformerHCT(nn.Module):
    def __init__(
        self,
        feature_dim,
        num_kernel,
        depth,
        n_heads,
        mlp_dim,
        temporal_kernel,
        dropout=0.1,
        factor=1,
        output_attention=False,
    ):
        super().__init__()
        max_depth = int(math.floor(math.log2(max(feature_dim, 1))))
        self.depth = max(0, min(int(depth), max_depth))

        layers = []
        current_dim = int(feature_dim)
        for _ in range(self.depth):
            layer = EEGDeformerHCTLayer(
                input_dim=current_dim,
                num_kernel=num_kernel,
                n_heads=n_heads,
                mlp_dim=mlp_dim,
                temporal_kernel=temporal_kernel,
                dropout=dropout,
                factor=factor,
                output_attention=output_attention,
            )
            layers.append(layer)
            current_dim = layer.output_dim

        self.layers = nn.ModuleList(layers)
        self.output_dim = current_dim

    def forward(self, x):
        dense_features = []
        for layer in self.layers:
            x, dense_info = layer(x)
            dense_features.append(dense_info)

        flat_hct = x.reshape(x.shape[0], -1)
        if dense_features:
            dense = torch.cat(dense_features, dim=-1)
            return torch.cat([flat_hct, dense], dim=-1)
        return flat_hct


class Model(nn.Module):
    """
    EEG-Deformer baseline adapted to the LEAD training interface.

    Input shape follows the project convention:
        x_enc: [B, T, C]

    Forward signature follows LEAD:
        forward(x_enc, label_id=None, ...)
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.num_class = configs.num_class
        self.num_kernel = int(configs.d_model)
        self.temporal_kernel = self._default_temporal_kernel(configs)

        self.embedding = EEGDeformerEmbedding(
            num_channels=self.enc_in,
            num_time=self.seq_len,
            temporal_kernel=self.temporal_kernel,
            d_model=self.num_kernel,
            dropout=configs.dropout,
        )

        self.hct = EEGDeformerHCT(
            feature_dim=self.embedding.feature_dim,
            num_kernel=self.num_kernel,
            depth=configs.e_layers,
            n_heads=configs.n_heads,
            mlp_dim=configs.d_ff,
            temporal_kernel=self.temporal_kernel,
            dropout=configs.dropout,
            factor=configs.factor,
            output_attention=configs.output_attention,
        )

        out_size = self.num_kernel * self.hct.output_dim + self.num_kernel * self.hct.depth
        self.classifier = nn.Sequential(
            nn.Dropout(configs.dropout),
            nn.Linear(out_size, configs.num_class),
        )

    @staticmethod
    def _default_temporal_kernel(configs):
        """Choose a small EEG-style temporal kernel from existing args.

        We do not add an EEGDeformer-specific parser. For 200 Hz EEG, this
        gives 21 samples, close to a 100 ms temporal filter.
        """
        sampling_rate = int(getattr(configs, "sampling_rate", 200))
        kernel = max(3, int(round(0.1 * sampling_rate)))
        if kernel % 2 == 0:
            kernel += 1
        return kernel

    def supervised(self, x_enc, label_id=None):
        x = self.embedding(x_enc)
        x = self.hct(x)
        return self.classifier(x)

    def forward(self, x_enc, label_id=None, mask=None, **kwargs):
        if self.task_name == "supervised":
            return self.supervised(x_enc, label_id=label_id)
        raise ValueError("Task name not recognized or not implemented within the EEGDeformer model")

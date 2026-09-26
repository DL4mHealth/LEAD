import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from layers.LEAD_EncDec import Encoder, EncoderLayer
from layers.Embed import LEADEmbedding
from layers.SelfAttention_Family import LEADLayer


class Model(nn.Module):
    """LEAD with dataset-aware electrode embedding and dynamic C/P shapes."""

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.output_attention = configs.output_attention
        self.patch_len = int(configs.patch_len)
        self.stride = int(configs.stride)
        self.enc_in = int(configs.enc_in)  # downstream dataset only
        self.d_model = int(configs.d_model)

        self.contrastive_token_ratio = float(
            getattr(configs, "contrastive_token_ratio", 0.25)
        )
        if not (0.0 < self.contrastive_token_ratio <= 1.0):
            raise ValueError(
                f"contrastive_token_ratio must be in (0, 1], got "
                f"{self.contrastive_token_ratio}."
            )

        channel_names_by_id = dict(getattr(configs, "channel_names_by_id", {}))
        montage_by_id = dict(getattr(configs, "montage_by_id", {}))
        self.use_sampling_embedding = bool(
            getattr(configs, "use_sampling_embedding", False)
        )
        sampling_rate_to_id = dict(getattr(configs, "sampling_rate_to_id", {}))
        if not channel_names_by_id:
            raise ValueError(
                "LEAD requires channel_names_by_id from dataset meta.json. "
                "Load PRETRAIN/TRAIN data before constructing the model."
            )
        if self.use_sampling_embedding and not sampling_rate_to_id:
            raise ValueError(
                "--use_sampling_embedding requires sampling_rate_to_id derived "
                "from dataset metadata."
            )

        augmentations = [x.strip() for x in configs.augmentations.split(",") if x.strip()]
        if not augmentations:
            augmentations = ["none"]
        if augmentations == ["none"] and self.task_name == "pretrain":
            augmentations = ["patch0.2", "mask0.2", "channel0.2"]

        self.enc_embedding = LEADEmbedding(
            d_model=configs.d_model,
            patch_len=configs.patch_len,
            stride=configs.stride,
            dropout=configs.dropout,
            channel_names_by_id=channel_names_by_id,
            montage_by_id=montage_by_id,
            sampling_rate_to_id=sampling_rate_to_id,
            use_sampling_embedding=self.use_sampling_embedding,
            augmentation=augmentations,
            max_patch_positions=getattr(configs, "max_patch_positions", 1024),
            temporal_pos_type=getattr(configs, "temporal_pos_type", "learnable"),
            channel_pos_type=getattr(configs, "channel_pos_type", "3D"),
            max_channel_positions=getattr(configs, "max_channel_positions", 368),
        )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    LEADLayer(
                        d_model=configs.d_model,
                        n_heads=configs.n_heads,
                        dropout=configs.dropout,
                        output_attention=configs.output_attention,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )

        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)

        if self.task_name in ["supervised", "finetune", "probe"]:
            # Downstream is intentionally restricted to one dataset, so the
            # classifier may remain dataset-specific while the backbone is C-agnostic.
            self.classifier = nn.Linear(self.enc_in * self.d_model, configs.num_class)
        elif self.task_name == "pretrain":
            # h is a channel-order/number invariant D-dimensional representation.
            # The projection head maps h -> z for contrastive learning, so one
            # shared head works for arbitrary C_keep and heterogeneous montages.
            self.projection_head = nn.Sequential(
                nn.Linear(self.d_model, self.d_model * 2),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(self.d_model * 2, self.d_model),
            )
        else:
            raise ValueError(
                "LEAD task_name must be one of: supervised, pretrain, finetune, probe."
            )

    @staticmethod
    def _parse_label_id(label_id):
        if label_id is None or label_id.ndim != 2 or label_id.shape[1] < 4:
            raise ValueError(
                "LEAD expects label_id with columns "
                "[disease_id, subject_id, sampling_frequency, dataset_id]."
            )
        fs = label_id[:, 2]
        dataset_id = label_id[:, 3]
        return fs, dataset_id

    def _embed(self, x_enc, label_id, apply_augmentation):
        fs, dataset_id = self._parse_label_id(label_id)
        tokens = self.enc_embedding(
            x_enc,
            dataset_id=dataset_id,
            fs=fs,
            apply_augmentation=apply_augmentation,
        )
        C = int(x_enc.shape[-1])
        if tokens.shape[1] % C != 0:
            raise ValueError(
                f"Token length {tokens.shape[1]} is not divisible by runtime C={C}."
            )
        P = int(tokens.shape[1] // C)
        return tokens, C, P

    def _encode_full(self, x_enc, label_id, apply_augmentation=False):
        tokens, C, P = self._embed(x_enc, label_id, apply_augmentation)
        enc_out, attns_t, attns_c = self.encoder(
            tokens,
            num_channels=C,
            num_patches=P,
            attn_mask=None,
        )
        return enc_out, C, P, attns_t, attns_c

    @staticmethod
    def _sample_structured_grid_indices(batch_size, C, P, keep_ratio, device):
        """Sample a rectangular C_keep x P_keep sub-grid.

        LEAD factorizes attention over a complete channel x patch grid, so
        arbitrary token subsampling would destroy the required structure.

        Both the selected channels and selected temporal patches are shared
        across the entire input-compatible batch. This keeps every sample aligned to
        the same spatial-temporal sub-grid, making subject-level contrast easier:
        positive trials differ in EEG content/augmentation, but not in which
        electrodes or temporal locations are observed. The two augmented views
        also reuse this exact same shared sub-grid.
        """
        if keep_ratio >= 1.0:
            channel_ids = torch.arange(C, device=device).unsqueeze(0).repeat(batch_size, 1)
            patch_ids = torch.arange(P, device=device).unsqueeze(0).repeat(batch_size, 1)
            return channel_ids, patch_ids

        axis_ratio = math.sqrt(keep_ratio)
        c_keep = min(C, max(1, int(round(C * axis_ratio))))
        p_keep = min(P, max(1, int(round(P * axis_ratio))))

        # One channel subset for the whole batch so channel-wise features remain
        # aligned across samples. Each batch is guaranteed to contain only
        # datasets with the same channel names and ordering.
        channel_ids_1d = torch.randperm(C, device=device)[:c_keep]
        channel_ids_1d = torch.sort(channel_ids_1d).values
        channel_ids = channel_ids_1d.unsqueeze(0).repeat(batch_size, 1)

        # One temporal subset for the whole batch as well. Positional information
        # was added before subsampling, so selected patches retain their true time
        # indices even though all samples now observe the same temporal locations.
        patch_ids_1d = torch.randperm(P, device=device)[:p_keep]
        patch_ids_1d = torch.sort(patch_ids_1d).values
        patch_ids = patch_ids_1d.unsqueeze(0).repeat(batch_size, 1)
        return channel_ids, patch_ids

    @staticmethod
    def _gather_structured_grid(tokens, C, P, channel_ids, patch_ids):
        B, L, D = tokens.shape
        if L != C * P:
            raise ValueError(f"Expected L=C*P, got L={L}, C={C}, P={P}.")

        grid = tokens.view(B, C, P, D)
        c_keep = channel_ids.shape[1]
        p_keep = patch_ids.shape[1]

        channel_index = channel_ids[:, :, None, None].expand(B, c_keep, P, D)
        grid = torch.gather(grid, dim=1, index=channel_index)
        patch_index = patch_ids[:, None, :, None].expand(B, c_keep, p_keep, D)
        grid = torch.gather(grid, dim=2, index=patch_index)
        return grid.reshape(B, c_keep * p_keep, D), c_keep, p_keep

    @staticmethod
    def _last_token_per_channel(enc_out, C, P):
        """Return the final temporal token for every channel: [B, C, D]."""
        return rearrange(enc_out, "b (c n) d -> b c n d", c=C, n=P)[:, :, -1, :]

    def _build_pretrain_representations(self, enc_out, C, P):
        """Build channel-invariant pretraining representations h and z.

        h: last token/channel -> GELU -> Dropout -> mean over channels, [B, D].
        z: projection_head(h), [B, D].
        """
        channel_features = self._last_token_per_channel(enc_out, C, P)
        channel_features = self.dropout(self.act(channel_features))

        # h is used for pretraining linear probing and is invariant to C/order.
        h = torch.mean(channel_features, dim=1)

        # z is used only by the contrastive objective.
        z = self.projection_head(h)
        return h, z

    def _contrastive_view_pair(self, x_enc, label_id):
        """Build two augmented views and return old-style (h, z) pairs.

        Structured channel/patch subsampling is retained from the current LEAD.
        After encoding, each view follows a channel-invariant h -> z path:
        last token/channel -> GELU -> Dropout -> channel mean pooling gives h,
        and the shared projection head maps h to z for contrastive learning.
        """
        tokens1, C, P = self._embed(x_enc, label_id, apply_augmentation=True)
        B = tokens1.shape[0]
        channel_ids, patch_ids = self._sample_structured_grid_indices(
            batch_size=B,
            C=C,
            P=P,
            keep_ratio=self.contrastive_token_ratio,
            device=tokens1.device,
        )
        tokens1, C_keep, P_keep = self._gather_structured_grid(
            tokens1, C, P, channel_ids, patch_ids
        )

        # The second augmentation uses the exact same spatial/temporal sub-grid
        # for each sample.
        tokens2, C2, P2 = self._embed(x_enc, label_id, apply_augmentation=True)
        if C2 != C or P2 != P:
            raise RuntimeError("The two contrastive views produced different token grids.")
        tokens2, _, _ = self._gather_structured_grid(
            tokens2, C, P, channel_ids, patch_ids
        )

        enc1, _, _ = self.encoder(
            tokens1,
            num_channels=C_keep,
            num_patches=P_keep,
            attn_mask=None,
        )
        enc2, _, _ = self.encoder(
            tokens2,
            num_channels=C_keep,
            num_patches=P_keep,
            attn_mask=None,
        )

        h1, z1 = self._build_pretrain_representations(enc1, C_keep, P_keep)
        h2, z2 = self._build_pretrain_representations(enc2, C_keep, P_keep)
        return h1, z1, h2, z2

    def encode(self, x_enc, label_id):
        """Return h for pretraining linear probing: [B, D].
        """
        enc_out, C, P, _, _ = self._encode_full(
            x_enc, label_id, apply_augmentation=False
        )
        output = self._last_token_per_channel(enc_out, C, P)
        output = self.dropout(self.act(output))
        return torch.mean(output, dim=1)

    def supervised(self, x_enc, label_id):
        # Restore the original LEAD downstream augmentation behavior:
        #   - supervised / finetune: augment only while the model is training;
        #   - probe: the backbone stays in eval mode, but input augmentation is
        #     enabled while the classifier is training.
        if self.task_name == "probe":
            apply_augmentation = self.classifier.training
        else:
            apply_augmentation = self.training

        enc_out, C, P, _, _ = self._encode_full(
            x_enc, label_id, apply_augmentation=apply_augmentation
        )
        if C != self.enc_in:
            raise ValueError(
                f"Downstream classifier was initialized for C={self.enc_in}, "
                f"but the current batch has C={C}. TRAIN/VAL/TEST must use the same dataset."
            )

        # Preserve the original LEAD downstream representation: last temporal
        # patch from every channel (the old comment called this a CLS token, but
        # no explicit CLS token is appended by LEADEmbedding).
        output = self._last_token_per_channel(enc_out, C, P)
        output = self.dropout(self.act(output)).reshape(output.shape[0], -1)
        return self.classifier(output)

    def forward(self, x_enc, label_id=None, mask=None, **kwargs):
        if self.task_name in ["supervised", "finetune", "probe"]:
            return self.supervised(x_enc, label_id)
        if self.task_name == "pretrain":
            h1, z1, h2, z2 = self._contrastive_view_pair(x_enc, label_id)
            return {"h1": h1, "z1": z1, "h2": h2, "z2": z2}
        raise ValueError(f"Unsupported task_name={self.task_name}")

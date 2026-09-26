"""
REVE baseline for the LEAD training interface.

This wrapper follows the same supervised-learning style used by large pretrained
baselines such as BIOT, LaBraM, and CBraMod: it loads the pretrained REVE-Base
backbone, builds electrode positions from MNE standard montages, and trains a
lightweight classifier on top.

REVE expects:
    eeg_data:  [B, C, T], sampled at 200 Hz
    positions: [B, C, 3]

The project convention is:
    x_enc: [B, T, C]
"""

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


class ReVeClassifierHead(nn.Module):
    """Small downstream classifier used on top of pooled REVE features."""

    def __init__(self, in_dim: int, num_class: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_class),
        )

    def forward(self, x):
        return self.net(x)


class Model(nn.Module):
    """
    REVE-Base supervised baseline adapted to LEAD.

    The pretrained backbone is loaded through Hugging Face Transformers remote
    code. Electrode positions are NOT loaded from brain-bzh/reve-positions.
    Instead, they are constructed automatically from MNE standard montages using
    the channel names provided by the LEAD dataset loader.
    """

    REVE_BASE_ID = "brain-bzh/reve-base"

    # MNE montage names to try when args.montage_by_id is unavailable or does
    # not cover all requested channels. Keep standard_1020 / standard_1005 first
    # because most ERP datasets use 10-20 / 10-10 / 10-05 naming.
    DEFAULT_MNE_MONTAGES = [
        "standard_1020",
        "standard_1005",
        "standard_1010",
        "biosemi16",
        "biosemi32",
        "biosemi64",
        "biosemi128",
        "biosemi160",
        "GSN-HydroCel-32",
        "GSN-HydroCel-64_1.0",
        "GSN-HydroCel-65_1.0",
        "GSN-HydroCel-128",
        "GSN-HydroCel-129",
        "GSN-HydroCel-256",
        "easycap-M1",
        "easycap-M10",
        "mgh60",
        "mgh70",
    ]

    CHANNEL_ALIASES = {
        # Old 10-20 temporal names to modern equivalents.
        "T3": "T7",
        "T4": "T8",
        "T5": "P7",
        "T6": "P8",
        # Common mastoid aliases.
        "A1": "M1",
        "A2": "M2",
        # Common capitalization variants are also handled automatically, but
        # keep the most common frontal pole spellings explicit.
        "FP1": "Fp1",
        "FP2": "Fp2",
        "FPZ": "Fpz",
        "AFZ": "AFz",
        "FCZ": "FCz",
        "CZ": "Cz",
        "CPZ": "CPz",
        "PZ": "Pz",
        "POZ": "POz",
        "OZ": "Oz",
        "IZ": "Iz",
    }

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.num_class = configs.num_class
        self.dropout = getattr(configs, "dropout", 0.1)

        if int(getattr(configs, "sampling_rate", 200)) != 200:
            print(
                "Warning: REVE was released for EEG sampled at 200 Hz, "
                f"but args.sampling_rate={getattr(configs, 'sampling_rate', None)}."
            )

        model_path = self._resolve_reve_model_path(configs)
        self.backbone = self._load_hf_model(model_path, model_name="REVE backbone")

        self.channel_names = self._resolve_channel_names(configs)
        requested_montage = self._resolve_requested_montage(configs)
        positions, resolved_names, montage_name = self._build_mne_positions(
            self.channel_names,
            requested_montage=requested_montage,
        )
        self.channel_names = resolved_names
        self.mne_montage_name = montage_name
        self.register_buffer("positions", positions, persistent=False)

        feature_dim = self._infer_feature_dim(configs)
        self.classifier = ReVeClassifierHead(
            in_dim=feature_dim,
            num_class=configs.num_class,
            dropout=self.dropout,
        )
        print(
            f"Initialized REVE baseline with backbone={model_path}, "
            f"MNE montage={self.mne_montage_name}, feature_dim={feature_dim}, "
            f"channels={len(self.channel_names)}."
        )

    @classmethod
    def _resolve_reve_model_path(cls, configs) -> str:
        """Use local/HF checkpoints_path if it looks like REVE, otherwise default to reve-base."""
        path = str(getattr(configs, "checkpoints_path", "") or "").strip()
        if path.startswith("hf:"):
            return path[3:]
        if path.startswith("brain-bzh/reve"):
            return path
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json")):
            return path
        return cls.REVE_BASE_ID

    @staticmethod
    def _load_hf_model(path: str, model_name: str):
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError(
                "REVE baseline requires the `transformers` package. "
                "Please install transformers or use another baseline."
            ) from exc

        try:
            return AutoModel.from_pretrained(path, trust_remote_code=True)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load {model_name} from {path}. "
                "If this is the Hugging Face model, make sure the environment is logged in "
                "with `huggingface-cli login` or `hf auth login` and that you have accepted "
                "access to brain-bzh/reve-base. You can also pass a local directory through "
                "--checkpoints_path if it contains the downloaded REVE files."
            ) from exc

    @staticmethod
    def _resolve_channel_names(configs) -> List[str]:
        channel_names_by_id = getattr(configs, "channel_names_by_id", None) or {}
        if isinstance(channel_names_by_id, dict) and len(channel_names_by_id) > 0:
            # Downstream TRAIN/VAL/TEST use a single dataset, so use the first entry.
            first_key = sorted(channel_names_by_id.keys(), key=lambda x: int(x))[0]
            names = list(channel_names_by_id[first_key])
        else:
            names = []

        if len(names) == 0:
            raise ValueError(
                "REVE requires channel names so electrode positions can be retrieved from MNE. "
                "Expected args.channel_names_by_id to be populated by the dataset loader."
            )
        if len(names) != int(getattr(configs, "enc_in", len(names))):
            print(
                f"Warning: REVE received {len(names)} channel names but enc_in={configs.enc_in}. "
                "The model will use the provided channel-name list."
            )
        return names

    @classmethod
    def _resolve_requested_montage(cls, configs) -> Optional[str]:
        montage_by_id = getattr(configs, "montage_by_id", None) or {}
        montage = None
        if isinstance(montage_by_id, dict) and len(montage_by_id) > 0:
            first_key = sorted(montage_by_id.keys(), key=lambda x: int(x))[0]
            montage = montage_by_id[first_key]
        if montage is None:
            montage = getattr(configs, "montage", None)
        return cls._normalize_montage_name(montage)

    @staticmethod
    def _normalize_montage_name(montage: Optional[str]) -> Optional[str]:
        if montage is None:
            return None
        name = str(montage).strip()
        if len(name) == 0 or name.lower() in {"none", "null", "unknown"}:
            return None

        key = name.lower().replace("-", "_").replace(" ", "")
        alias = {
            "standard1020": "standard_1020",
            "standard_1020": "standard_1020",
            "1020": "standard_1020",
            "10_20": "standard_1020",
            "standard1010": "standard_1010",
            "standard_1010": "standard_1010",
            "1010": "standard_1010",
            "10_10": "standard_1010",
            "standard1005": "standard_1005",
            "standard_1005": "standard_1005",
            "1005": "standard_1005",
            "10_05": "standard_1005",
            "standard_1005_cap385": "standard_1005",
            "standard_1020_cap19": "standard_1020",
            "standard_1020_cap21": "standard_1020",
            "standard_1020_cap60": "standard_1020",
            "standard_1020_cap64": "standard_1020",
            "gsn65": "GSN-HydroCel-65_1.0",
            "gsn_65": "GSN-HydroCel-65_1.0",
            "gsn_hydrocel_65": "GSN-HydroCel-65_1.0",
            "gsn_hydrocel_65_1.0": "GSN-HydroCel-65_1.0",
            "gsn64": "GSN-HydroCel-64_1.0",
            "gsn_64": "GSN-HydroCel-64_1.0",
            "gsn_hydrocel_64": "GSN-HydroCel-64_1.0",
            "gsn_hydrocel_64_1.0": "GSN-HydroCel-64_1.0",
            "biosemi16": "biosemi16",
            "biosemi32": "biosemi32",
            "biosemi64": "biosemi64",
            "biosemi128": "biosemi128",
            "biosemi160": "biosemi160",
            "easycapm1": "easycap-M1",
            "easycap_m1": "easycap-M1",
            "easycapm10": "easycap-M10",
            "easycap_m10": "easycap-M10",
        }
        return alias.get(key, name)

    @staticmethod
    def _clean_channel_name(name: str) -> str:
        name = str(name).strip()
        # Common prefixes/suffixes introduced by EEG file formats.
        for prefix in ["EEG ", "EEG-", "EEG_", "POL ", "POL-"]:
            if name.upper().startswith(prefix.upper()):
                name = name[len(prefix):]
        for suffix in ["-REF", "_REF", " REF", "-LE", "_LE", "-AVG", "_AVG"]:
            if name.upper().endswith(suffix):
                name = name[: -len(suffix)]
        return name.strip().strip(".")

    @classmethod
    def _channel_candidates(cls, name: str) -> List[str]:
        clean = cls._clean_channel_name(name)
        candidates = [clean]

        upper = clean.upper()
        if upper in cls.CHANNEL_ALIASES:
            candidates.append(cls.CHANNEL_ALIASES[upper])

        # Handle common case variants such as fp1 -> Fp1, fz -> Fz, afz -> AFz.
        candidates.extend([
            clean.upper(),
            clean.lower(),
            clean.capitalize(),
            clean[:1].upper() + clean[1:].lower() if clean else clean,
        ])
        if len(clean) >= 2:
            candidates.append(clean[:-1].upper() + clean[-1].lower())  # FPZ -> FPz / AFZ -> AFz
            candidates.append(clean[0].upper() + clean[1:-1].lower() + clean[-1])

        # Preserve order while removing duplicates.
        deduped = []
        seen = set()
        for cand in candidates:
            if cand and cand not in seen:
                seen.add(cand)
                deduped.append(cand)
        return deduped

    @classmethod
    def _try_resolve_channels_in_montage(
        cls,
        channel_names: Sequence[str],
        ch_pos: Dict[str, object],
    ) -> Tuple[Optional[List[str]], List[str]]:
        exact = {str(name): str(name) for name in ch_pos.keys()}
        lower_map = {str(name).lower(): str(name) for name in ch_pos.keys()}

        resolved = []
        missing = []
        for raw_name in channel_names:
            matched = None
            for cand in cls._channel_candidates(str(raw_name)):
                if cand in exact:
                    matched = exact[cand]
                    break
                lower = cand.lower()
                if lower in lower_map:
                    matched = lower_map[lower]
                    break
            if matched is None:
                missing.append(str(raw_name))
            else:
                resolved.append(matched)

        if missing:
            return None, missing
        return resolved, []

    @classmethod
    def _build_mne_positions(
        cls,
        channel_names: Sequence[str],
        requested_montage: Optional[str] = None,
    ) -> Tuple[torch.Tensor, List[str], str]:
        try:
            import numpy as np
            import mne
        except ImportError as exc:
            raise ImportError(
                "REVE with automatic MNE positions requires `mne` and `numpy`. "
                "Please install them with `pip install mne numpy`."
            ) from exc

        montage_candidates = []
        if requested_montage is not None:
            montage_candidates.append(requested_montage)
        for montage_name in cls.DEFAULT_MNE_MONTAGES:
            if montage_name not in montage_candidates:
                montage_candidates.append(montage_name)

        tried = []
        last_missing = None
        last_available_preview = None
        for montage_name in montage_candidates:
            try:
                montage = mne.channels.make_standard_montage(montage_name)
            except Exception:
                tried.append(f"{montage_name}: unavailable")
                continue

            ch_pos = montage.get_positions().get("ch_pos", {})
            resolved_names, missing = cls._try_resolve_channels_in_montage(channel_names, ch_pos)
            if resolved_names is None:
                tried.append(f"{montage_name}: missing {missing[:8]}")
                last_missing = missing
                last_available_preview = list(ch_pos.keys())[:20]
                continue

            positions_np = np.stack([ch_pos[name] for name in resolved_names], axis=0).astype("float32")
            positions = torch.from_numpy(positions_np)
            return positions, resolved_names, montage_name

        raise ValueError(
            "Unable to build REVE electrode positions from MNE for the requested channels.\n"
            f"Original channel names: {list(channel_names)}\n"
            f"Requested montage: {requested_montage}\n"
            f"Tried montages: {tried}\n"
            f"Last missing channels: {last_missing}\n"
            f"Example available channels from last tried montage: {last_available_preview}\n"
            "Please check dataset channel names or set args.montage_by_id to a valid MNE montage name."
        )

    def _build_positions(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        pos = self.positions.to(device=device, dtype=dtype)
        if pos.shape[0] != len(self.channel_names):
            raise ValueError(
                f"MNE returned {pos.shape[0]} positions for {len(self.channel_names)} channels. "
                f"Check channel names: {self.channel_names}."
            )
        return pos.unsqueeze(0).expand(batch_size, -1, -1)

    @staticmethod
    def _first_tensor(output: Any) -> torch.Tensor:
        if torch.is_tensor(output):
            return output
        if hasattr(output, "pooler_output") and torch.is_tensor(output.pooler_output):
            return output.pooler_output
        if hasattr(output, "last_hidden_state") and torch.is_tensor(output.last_hidden_state):
            return output.last_hidden_state
        if isinstance(output, dict):
            for key in ["pooler_output", "last_hidden_state", "features", "embeddings", "hidden_states"]:
                value = output.get(key, None)
                if torch.is_tensor(value):
                    return value
                if isinstance(value, (list, tuple)) and value and torch.is_tensor(value[-1]):
                    return value[-1]
            for value in output.values():
                if torch.is_tensor(value):
                    return value
        if isinstance(output, (list, tuple)):
            for value in output:
                if torch.is_tensor(value):
                    return value
                if hasattr(value, "last_hidden_state") and torch.is_tensor(value.last_hidden_state):
                    return value.last_hidden_state
        raise TypeError(f"Unsupported REVE output type: {type(output)}")

    @staticmethod
    def _pool_features(features: torch.Tensor) -> torch.Tensor:
        """Return a [B, D] tensor from tensor outputs of different possible ranks."""
        if features.ndim == 2:
            return features
        if features.ndim < 2:
            raise ValueError(f"Expected batched features from REVE, got shape {features.shape}.")
        # Keep batch and final embedding dimension; pool all intermediate token/time/channel axes.
        return features.reshape(features.shape[0], -1, features.shape[-1]).mean(dim=1)

    def _extract_features(self, x_enc):
        # Project convention: [B, T, C]. REVE convention: [B, C, T].
        eeg_data = x_enc.permute(0, 2, 1).contiguous()
        positions = self._build_positions(
            batch_size=eeg_data.shape[0],
            device=eeg_data.device,
            dtype=eeg_data.dtype,
        )
        output = self.backbone(eeg_data, positions)
        features = self._first_tensor(output)
        return self._pool_features(features)

    def _feature_dim_from_config(self) -> Optional[int]:
        config = getattr(self.backbone, "config", None)
        if config is None:
            return None
        for name in [
            "hidden_size",
            "d_model",
            "embed_dim",
            "embedding_dim",
            "encoder_embed_dim",
            "dim",
        ]:
            value = getattr(config, name, None)
            if isinstance(value, int) and value > 0:
                return int(value)
        return None

    def _infer_feature_dim(self, configs) -> int:
        # Prefer a real dry run because remote-code model outputs may be pooled or token-level.
        was_training = self.backbone.training
        self.backbone.eval()
        try:
            with torch.no_grad():
                dummy_len = max(16, int(getattr(configs, "seq_len", 200)))
                dummy = torch.zeros(1, dummy_len, len(self.channel_names), dtype=torch.float32)
                features = self._extract_features(dummy)
                feature_dim = int(features.shape[-1])
        except Exception as exc:
            feature_dim = self._feature_dim_from_config()
            if feature_dim is None:
                raise RuntimeError(
                    "Unable to infer REVE feature dimension from a dry run or from the HF config."
                ) from exc
        finally:
            self.backbone.train(was_training)
        return feature_dim

    def supervised(self, x_enc, label_id=None):
        features = self._extract_features(x_enc)
        return self.classifier(features)

    def forward(self, x_enc, label_id=None, mask=None, **kwargs):
        if self.task_name == "supervised":
            return self.supervised(x_enc, label_id=label_id)
        raise ValueError("Task name not recognized or not implemented within the ReVe model")

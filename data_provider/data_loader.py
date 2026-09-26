"""
Multi-dataset loader for heterogeneous EEG datasets with different sequence
lengths, channel counts, montages, and channel names.

Main guarantees:
    1. X from different datasets is never concatenated.
    2. PRETRAIN may contain multiple datasets with heterogeneous T/C.
    3. PRETRAIN samples may share a mini-batch only when sequence length,
       channel count, montage, channel names, and channel order are identical.
    4. TRAIN/VAL/TEST always use exactly one downstream dataset.
    5. Dataset IDs are stable across runs/splits and are used to route metadata
       such as channel names and montage to LEAD.

Label format exposed by this wrapper:
    [disease_id, subject_id, sampling_frequency, dataset_id]
"""

import os
from typing import Dict, Iterator, List
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from data_provider.dataset_loader.adftd_loader import ADFTDLoader
from data_provider.dataset_loader.cnbpm_loader import CNBPMLoader
from data_provider.dataset_loader.cognision_rs_loader import CognisionRSLoader
from data_provider.dataset_loader.apava_loader import APAVALoader
from data_provider.dataset_loader.adfsu_loader import ADFSULoader
from data_provider.dataset_loader.adsz_loader import ADSZLoader
from data_provider.dataset_loader.caueeg_loader import CAUEEGLoader
from data_provider.dataset_loader.ad_auditory_loader import ADAuditoryLoader
from data_provider.dataset_loader.baca_rs_loader import BACARSLoader
from data_provider.dataset_loader.brainlat_loader import BrainLatLoader
from data_provider.dataset_loader.depression_loader import DepressionLoader
from data_provider.dataset_loader.fepcr_loader import FEPCRLoader
from data_provider.dataset_loader.mcef_rs_loader import MCEFRSLoader
from data_provider.dataset_loader.p_adic_loader import PADICLoader
from data_provider.dataset_loader.pd_rs_loader import PDRSLoader
from data_provider.dataset_loader.pearl_neuro_loader import PEARLNeuroLoader
from data_provider.dataset_loader.srm_rs_loader import SRMRSLoader
from data_provider.dataset_loader.tdbrain_loader import TBDRAINLoader
from data_provider.dataset_loader.tueg_loader import TUEGLoader


# Dataset folder name -> loader.
data_folder_dict = {
    # Downstream datasets
    "ADFTD": ADFTDLoader,
    "ADFTD-RS": ADFTDLoader,
    "ADFTD-PS": ADFTDLoader,
    "CNBPM": CNBPMLoader,
    "Cognision-RS": CognisionRSLoader,
    "APAVA": APAVALoader,
    "ADFSU": ADFSULoader,
    "ADSZ": ADSZLoader,

    # Pretraining datasets
    "AD-Auditory": ADAuditoryLoader,
    "BACA-RS": BACARSLoader,
    "BrainLat": BrainLatLoader,
    "Depression": DepressionLoader,
    "FEPCR": FEPCRLoader,
    "MCEF-RS": MCEFRSLoader,
    "P-ADIC": PADICLoader,
    "PD-RS": PDRSLoader,
    "PEARL-Neuro": PEARLNeuroLoader,
    "SRM-RS": SRMRSLoader,
    "TDBrain": TBDRAINLoader,
    "TUEP": TUEGLoader,
    "TUEG": TUEGLoader,
    "CAUEEG": CAUEEGLoader,
}


# Stable IDs are required because model-side coordinate buffers are keyed by
# dataset_id. Never derive dataset IDs from the order of a comma-separated list.
global_dataset_id_by_name = {
    dataset_name: dataset_idx + 1
    for dataset_idx, dataset_name in enumerate(data_folder_dict.keys())
}

warnings.filterwarnings("ignore")


class MultiDataset(Dataset):
    """Index-based wrapper for one or more heterogeneous EEG datasets."""

    def __init__(self, args, root_path, flag=None):
        self.args = args
        self.root_path = root_path
        self.flag = self._normalize_flag(flag)
        self.no_normalize = getattr(args, "no_normalize", False)

        data_folder_list = self._get_dataset_list(args, self.flag)
        print(f"Loading {self.flag} samples from EEG datasets...")
        print(f"Datasets used: {data_folder_list}")

        self.datasets = []
        self.dataset_names: List[str] = []
        self.dataset_ids: List[int] = []
        self.dataset_id_to_name: Dict[int, str] = {}
        self.global_to_local: List[tuple] = []
        self.indices_by_dataset: Dict[int, List[int]] = {}

        self.dataset_metadata_by_id: Dict[int, dict] = {}
        self.channel_names_by_id: Dict[int, List[str]] = {}
        self.montage_by_id: Dict[int, str] = {}
        self.sampling_rates_by_id: Dict[int, List[int]] = {}

        # PRETRAIN compatibility groups. Two datasets are allowed to share a
        # mini-batch only when their input geometry and channel metadata match
        # exactly. The original dataset_id is always preserved.
        self.compatibility_group_by_dataset_idx: Dict[int, int] = {}
        self.compatibility_group_by_dataset_id: Dict[int, int] = {}
        self.compatibility_group_key_by_id: Dict[int, tuple] = {}
        self.indices_by_compatibility_group: Dict[int, List[int]] = {}
        self.dataset_indices_by_compatibility_group: Dict[int, List[int]] = {}
        compatibility_key_to_group_id: Dict[tuple, int] = {}

        all_y = []
        global_index = 0
        max_seq_len = 0
        max_num_channels = 0

        for dataset_idx, dataset_name in enumerate(data_folder_list):
            if dataset_name not in data_folder_dict:
                raise ValueError(
                    f"Dataset '{dataset_name}' is not found in data_folder_dict. "
                    f"Available datasets: {list(data_folder_dict.keys())}"
                )

            Data = data_folder_dict[dataset_name]
            dataset = Data(
                root_path=os.path.join(root_path, dataset_name),
                args=args,
                flag=self.flag,
            )

            if not hasattr(dataset, "X") or not hasattr(dataset, "y"):
                raise AttributeError(
                    f"The loader for {dataset_name} must provide dataset.X and dataset.y."
                )
            if dataset.y.ndim != 2 or dataset.y.shape[1] != 3:
                raise ValueError(
                    f"{dataset_name}.y must have three columns before wrapping: "
                    f"[disease_id, subject_id, sampling_frequency]. Got {dataset.y.shape}."
                )

            dataset_id = global_dataset_id_by_name[dataset_name]
            y = np.asarray(dataset.y, dtype=np.float32).copy()
            dataset_id_col = np.full((y.shape[0], 1), dataset_id, dtype=y.dtype)
            y = np.concatenate([y[:, :3], dataset_id_col], axis=1)
            dataset.y = y

            n_samples = len(dataset)
            dataset_global_indices = list(range(global_index, global_index + n_samples))
            self.indices_by_dataset[dataset_idx] = dataset_global_indices
            self.global_to_local.extend((dataset_idx, local_idx) for local_idx in range(n_samples))
            global_index += n_samples

            self.datasets.append(dataset)
            self.dataset_names.append(dataset_name)
            self.dataset_ids.append(dataset_id)
            self.dataset_id_to_name[dataset_id] = dataset_name
            all_y.append(y)

            metadata = dict(getattr(dataset, "dataset_metadata", {}) or {})
            channels = list(getattr(dataset, "channels", metadata.get("channels", [])) or [])
            montage = getattr(dataset, "montage", metadata.get("montage", None))
            sampling_rates = list(
                getattr(dataset, "sample_rate_list", metadata.get("sample_rate_list", [])) or []
            )
            if not sampling_rates:
                sampling_rates = sorted(np.unique(y[:, 2].astype(int)).tolist())

            self.dataset_metadata_by_id[dataset_id] = metadata
            self.channel_names_by_id[dataset_id] = channels
            self.montage_by_id[dataset_id] = montage
            self.sampling_rates_by_id[dataset_id] = [int(x) for x in sampling_rates]

            seq_len = int(dataset.X.shape[1])
            num_channels = int(dataset.X.shape[2])
            max_seq_len = max(max_seq_len, seq_len)
            max_num_channels = max(max_num_channels, num_channels)

            if channels and len(channels) != num_channels:
                raise ValueError(
                    f"{dataset_name}: X has C={num_channels}, but meta CHANNELS has "
                    f"{len(channels)} entries."
                )

            compatibility_key = self._make_compatibility_key(
                seq_len=seq_len,
                num_channels=num_channels,
                montage=montage,
                channels=channels,
                dataset_id=dataset_id,
            )
            if compatibility_key not in compatibility_key_to_group_id:
                compatibility_key_to_group_id[compatibility_key] = len(compatibility_key_to_group_id)
            compatibility_group_id = compatibility_key_to_group_id[compatibility_key]
            self.compatibility_group_by_dataset_idx[dataset_idx] = compatibility_group_id
            self.compatibility_group_by_dataset_id[dataset_id] = compatibility_group_id
            self.compatibility_group_key_by_id[compatibility_group_id] = compatibility_key
            self.indices_by_compatibility_group.setdefault(compatibility_group_id, []).extend(
                dataset_global_indices
            )
            self.dataset_indices_by_compatibility_group.setdefault(
                compatibility_group_id, []
            ).append(dataset_idx)

            print(
                f"[dataset_id={dataset_id}, compatibility_group={compatibility_group_id}] "
                f"{dataset_name}: trials={n_samples}, T={seq_len}, C={num_channels}, "
                f"montage={montage}, sampling_rates={self.sampling_rates_by_id[dataset_id]}"
            )

        if not self.datasets:
            raise ValueError("No dataset was loaded. Check the dataset arguments.")

        self.y = np.concatenate(all_y, axis=0)
        self.X = None  # heterogeneous X must never be concatenated

        # Make metadata reachable from configs before model construction.
        self.args.global_dataset_id_by_name = global_dataset_id_by_name
        self.args.dataset_id_to_name = dict(self.dataset_id_to_name)
        self.args.dataset_metadata_by_id = dict(self.dataset_metadata_by_id)
        self.args.channel_names_by_id = dict(self.channel_names_by_id)
        self.args.montage_by_id = dict(self.montage_by_id)
        self.args.sampling_rates_by_id = dict(self.sampling_rates_by_id)
        self.args.compatibility_group_by_dataset_id = dict(self.compatibility_group_by_dataset_id)

        if self.flag in ["TRAIN", "VAL", "TEST"] and len(self.datasets) == 1:
            self.seq_len = int(self.datasets[0].X.shape[1])
            self.max_seq_len = self.seq_len
            self.enc_in = int(self.datasets[0].X.shape[2])
            self.num_channels = self.enc_in
            self.num_class = int(len(np.unique(self.y[:, 0])))
        else:
            self.max_seq_len = max_seq_len
            self.max_num_channels = max_num_channels

        print(self.summary())
        print()

    @staticmethod
    def _normalize_flag(flag):
        if flag in ["TRAIN", "train"]:
            return "TRAIN"
        if flag in ["VAL", "val", "valid"]:
            return "VAL"
        if flag in ["TEST", "test"]:
            return "TEST"
        if flag in ["PRETRAIN", "pretrain"]:
            return "PRETRAIN"
        raise ValueError("flag must be PRETRAIN, TRAIN, VAL, or TEST")

    @staticmethod
    def _split_dataset_names(dataset_string: str) -> List[str]:
        return [x.strip() for x in dataset_string.split(",") if x.strip()]

    @staticmethod
    def _make_compatibility_key(seq_len, num_channels, montage, channels, dataset_id):
        """Return the exact input-geometry key used for PRETRAIN batch mixing.

        Same sequence length, montage, channel names, and channel order are
        required. If channel metadata is missing, fall back to a dataset-unique
        key so unrelated datasets can never be mixed accidentally.
        """
        channel_signature = tuple(str(ch).strip() for ch in (channels or []))
        if montage is None or not channel_signature:
            return ("dataset_only", int(dataset_id), int(seq_len), int(num_channels))
        montage_signature = str(montage).strip().lower()
        return (
            "compatible",
            int(seq_len),
            int(num_channels),
            montage_signature,
            channel_signature,
        )

    def _get_dataset_list(self, args, flag) -> List[str]:
        if flag == "PRETRAIN":
            dataset_list = self._split_dataset_names(args.pretraining_datasets)
        elif flag in ["TRAIN", "VAL", "TEST"]:
            dataset_list = self._split_dataset_names(args.training_dataset)
            if len(dataset_list) != 1:
                raise ValueError(
                    f"For {flag}, exactly one training dataset is allowed. "
                    f"Got {dataset_list}. Only --pretraining_datasets may contain multiple datasets."
                )
        else:
            raise ValueError("flag must be PRETRAIN, TRAIN, VAL, or TEST")

        if not dataset_list:
            raise ValueError(f"No dataset was provided for flag={flag}.")
        return dataset_list

    def __getitem__(self, index):
        dataset_idx, local_idx = self.global_to_local[index]
        x, y = self.datasets[dataset_idx][local_idx]

        # The child loader owns only the first three columns. Add stable dataset_id
        # here instead of mutating child-loader semantics.
        dataset_id = self.dataset_ids[dataset_idx]
        y = torch.cat([y[:3], y.new_tensor([float(dataset_id)])], dim=0)
        return x, y

    def __len__(self):
        return len(self.global_to_local)

    def get_dataset_metadata(self, dataset_id: int) -> dict:
        return self.dataset_metadata_by_id[int(dataset_id)]

    def get_batch_metadata(self, label_id) -> dict:
        if torch.is_tensor(label_id):
            dataset_ids = torch.unique(label_id[:, 3]).detach().cpu().numpy().tolist()
        else:
            dataset_ids = np.unique(np.asarray(label_id)[:, 3]).tolist()
        if len(dataset_ids) != 1:
            raise ValueError(f"Expected one dataset_id per batch, got {dataset_ids}.")
        return self.get_dataset_metadata(int(dataset_ids[0]))

    def summary(self) -> str:
        unique_subjects = 0
        for dataset in self.datasets:
            unique_subjects += len(np.unique(np.asarray(dataset.y)[:, 1]))

        lines = [
            f"Dataset(flag={self.flag}), total trials: {len(self)}, "
            f"total subjects: {unique_subjects}, total datasets: {len(self.datasets)}."
        ]
        for dataset_idx, dataset in enumerate(self.datasets):
            dataset_id = self.dataset_ids[dataset_idx]
            compatibility_group_id = self.compatibility_group_by_dataset_idx[dataset_idx]
            lines.append(
                f"  dataset_id={dataset_id}, compatibility_group={compatibility_group_id}, "
                f"name={self.dataset_names[dataset_idx]}, trials={len(dataset)}, "
                f"seq_len={dataset.X.shape[1]}, channels={dataset.X.shape[2]}, "
                f"montage={self.montage_by_id.get(dataset_id)}"
            )
        return "\n".join(lines)


class DatasetHomogeneousBatchSampler(Sampler[List[int]]):
    """Yield input-compatible PRETRAIN mini-batches with memmap-friendly access.

    The historical class name is kept for compatibility, but batches are no
    longer restricted to one dataset. Datasets sharing the same compatibility
    group (same T, C, montage, channel names, and channel order) may be mixed in
    a batch. Sampling is always proportional: every sample appears once per
    epoch before ``drop_last`` is applied.
    """

    def __init__(
        self,
        dataset: MultiDataset,
        batch_size: int,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: int = 42,
        sequential_first_epoch: bool = True,
        batch_block_size: int = 8,
        batch_inner_group_size: int = 16,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if batch_block_size <= 0:
            raise ValueError("batch_block_size must be positive.")
        if batch_inner_group_size <= 0:
            raise ValueError("batch_inner_group_size must be positive.")

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.sequential_first_epoch = bool(sequential_first_epoch)
        self.batch_block_size = int(batch_block_size)
        self.batch_inner_group_size = int(batch_inner_group_size)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _num_batches_from_n(self, n: int) -> int:
        if self.drop_last:
            return n // self.batch_size
        return int(np.ceil(n / self.batch_size))

    def _make_contiguous_batches(self, indices: np.ndarray) -> List[List[int]]:
        """Split one compatibility group's ordered indices into mini-batches."""
        n = len(indices)
        usable_n = (n // self.batch_size) * self.batch_size if self.drop_last else n
        batches: List[List[int]] = []
        for start in range(0, usable_n, self.batch_size):
            batch = indices[start:start + self.batch_size].tolist()
            if len(batch) == self.batch_size or (batch and not self.drop_last):
                batches.append(batch)
        return batches

    def _make_batch_blocks(self, batches: List[List[int]]) -> List[List[List[int]]]:
        return [
            batches[start:start + self.batch_block_size]
            for start in range(0, len(batches), self.batch_block_size)
        ]

    @staticmethod
    def _flatten_blocks(blocks: List[List[List[int]]]) -> List[List[int]]:
        return [batch for block in blocks for batch in block]

    def _shuffle_groups_inside_batch(
        self,
        batch: List[int],
        rng: np.random.Generator,
    ) -> List[int]:
        groups = [
            batch[start:start + self.batch_inner_group_size]
            for start in range(0, len(batch), self.batch_inner_group_size)
        ]
        rng.shuffle(groups)
        return [index for group in groups for index in group]

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        use_sequential_epoch = self.sequential_first_epoch and self.epoch == 0
        all_blocks: List[List[List[int]]] = []

        # Indices from datasets with identical input geometry are concatenated
        # before batching. This naturally combines dataset tails while never
        # mixing incompatible T/C/channel layouts.
        for compatibility_group_id in sorted(self.dataset.indices_by_compatibility_group):
            indices = np.asarray(
                self.dataset.indices_by_compatibility_group[compatibility_group_id],
                dtype=np.int64,
            )
            batches = self._make_contiguous_batches(indices)
            all_blocks.extend(self._make_batch_blocks(batches))

        # Preserve the previous memmap-friendly first epoch. Later epochs shuffle
        # local batch blocks globally, while every batch remains compatibility-
        # homogeneous.
        if self.shuffle and not use_sequential_epoch:
            rng.shuffle(all_blocks)

        all_batches = self._flatten_blocks(all_blocks)
        if self.shuffle and not use_sequential_epoch:
            all_batches = [
                self._shuffle_groups_inside_batch(batch, rng) for batch in all_batches
            ]

        for batch in all_batches:
            yield batch

    def __len__(self) -> int:
        return sum(
            self._num_batches_from_n(len(indices))
            for indices in self.dataset.indices_by_compatibility_group.values()
        )


class SubjectRateGroupedDatasetBatchSampler(Sampler[List[int]]):
    """Old-style index group shuffling within input-compatibility groups.

    The original grouping logic is preserved:
        1. Group samples by ``(dataset_id, subject_id, sampling_rate)``.
        2. Shuffle samples within each group.
        3. Split each group into chunks of at most ``group_size`` samples.
        4. Pool and shuffle all chunks from datasets that share the same input
           compatibility group.
        5. Flatten chunks, split by ``batch_size``, then shuffle sample order
           inside each mini-batch.

    Including ``dataset_id`` in the group key prevents equal local subject IDs
    from different datasets from being treated as the same subject. Sampling is
    always proportional; no dataset balancing or resampling is performed.
    """

    def __init__(
        self,
        dataset: MultiDataset,
        batch_size: int,
        group_size: int = 2,
        drop_last: bool = True,
        shuffle: bool = False,
        seed: int = 42,
        sequential_first_epoch: bool = False,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if group_size <= 0:
            raise ValueError("group_size must be positive.")
        if group_size > batch_size:
            raise ValueError("group_size cannot exceed batch_size.")

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.group_size = int(group_size)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.sequential_first_epoch = bool(sequential_first_epoch)
        self.epoch = 0

        # Cache global indices by compatibility group and by the original LEAD
        # grouping key, extended with dataset_id to avoid cross-dataset subject
        # collisions.
        self.indices_by_compatibility_group_group: Dict[int, Dict[tuple, List[int]]] = {}
        for dataset_idx, global_indices in dataset.indices_by_dataset.items():
            child = dataset.datasets[dataset_idx]
            dataset_id = int(dataset.dataset_ids[dataset_idx])
            compatibility_group_id = dataset.compatibility_group_by_dataset_idx[dataset_idx]
            group_map = self.indices_by_compatibility_group_group.setdefault(
                compatibility_group_id, {}
            )
            for local_idx, global_idx in enumerate(global_indices):
                subject_id = int(child.y[local_idx, 1])
                sampling_rate = int(child.y[local_idx, 2])
                group_key = (dataset_id, subject_id, sampling_rate)
                group_map.setdefault(group_key, []).append(global_idx)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _make_grouped_order(
        self,
        compatibility_group_id: int,
        rng: np.random.Generator,
        do_shuffle: bool,
    ) -> List[int]:
        """Build one compatibility group's old-style group-shuffled order."""
        group_map = self.indices_by_compatibility_group_group[compatibility_group_id]
        chunks: List[List[int]] = []

        for group_indices in group_map.values():
            indices = np.asarray(group_indices, dtype=np.int64).copy()
            if do_shuffle:
                rng.shuffle(indices)
            chunks.extend(
                indices[start:start + self.group_size].tolist()
                for start in range(0, len(indices), self.group_size)
            )

        if do_shuffle:
            rng.shuffle(chunks)
        return [index for chunk in chunks for index in chunk]

    def _order_to_batches(
        self,
        order: List[int],
        rng: np.random.Generator,
        do_shuffle: bool,
    ) -> List[List[int]]:
        batches: List[List[int]] = []
        n = len(order)
        usable_n = (n // self.batch_size) * self.batch_size if self.drop_last else n

        for start in range(0, usable_n, self.batch_size):
            batch = list(order[start:start + self.batch_size])
            if not batch:
                continue
            if do_shuffle:
                rng.shuffle(batch)
            batches.append(batch)
        return batches

    def _compatibility_group_batches(
        self,
        compatibility_group_id: int,
        rng: np.random.Generator,
        do_shuffle: bool,
    ) -> List[List[int]]:
        order = self._make_grouped_order(compatibility_group_id, rng, do_shuffle)
        return self._order_to_batches(order, rng, do_shuffle)

    def __iter__(self) -> Iterator[List[int]]:
        # Rebuild a fresh group-shuffled order every DataLoader iteration, as in
        # the original LEAD sampler. Explicit set_epoch() remains supported.
        current_epoch = self.epoch
        self.epoch += 1
        rng = np.random.default_rng(self.seed + current_epoch)
        do_shuffle = self.shuffle and not (
            self.sequential_first_epoch and current_epoch == 0
        )

        all_batches: List[List[int]] = []
        for compatibility_group_id in sorted(self.indices_by_compatibility_group_group):
            all_batches.extend(
                self._compatibility_group_batches(
                    compatibility_group_id, rng, do_shuffle
                )
            )

        # Finished batches from different compatibility groups can be shuffled
        # globally; samples are never mixed across incompatible groups.
        if do_shuffle:
            rng.shuffle(all_batches)

        for batch in all_batches:
            yield batch

    def __len__(self) -> int:
        total = 0
        for group_map in self.indices_by_compatibility_group_group.values():
            n = sum(len(indices) for indices in group_map.values())
            if self.drop_last:
                total += n // self.batch_size
            else:
                total += int(np.ceil(n / self.batch_size))
        return total

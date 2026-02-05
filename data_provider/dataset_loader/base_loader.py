import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from data_provider.uea import normalize_batch_ts


def build_subject_label_table_from_y(y_mem: np.memmap) -> np.ndarray:
    """
    Build subject-level label table from segment-level y_mem.
    y_mem: shape (N_segment, 3) => [disease_label, subject_id, fs]
    Returns:
        data_list: shape (N_subject, 2) => [disease_label, subject_id]
    """
    subj_ids = y_mem[:, 1].astype(int)
    labels = y_mem[:, 0].astype(int)

    unique_sids = np.unique(subj_ids)
    rows = []

    for sid in unique_sids:
        # find all segments of this subject
        idx = np.where(subj_ids == sid)[0]
        lab = int(labels[idx[0]])  # assume disease label is consistent per subject
        rows.append([lab, sid])

    data_list = np.asarray(rows, dtype=int)
    return data_list


class BaseLoader(Dataset):
    """
    Base class for memmap-based EEG datasets.

    Common behavior:
      - Load meta.json, X.dat, y.dat
      - Build subject-level table from y_mem
      - Use dataset-specific _get_id_lists() for subject split
      - Use dataset-specific _postprocess_labels() for label remapping
      - Provide __getitem__/__len__, self.X, self.y, self.max_seq_len
    """

    def __init__(self, args, root_path, flag=None):
        super().__init__()
        self.args = args
        self.root_path = root_path
        self.flag = flag
        self.no_normalize = args.no_normalize

        # ------ load meta.json ------
        meta_path = os.path.join(root_path, 'meta.json')
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"meta.json not found at {meta_path}")

        with open(meta_path, 'r') as f:
            meta = json.load(f)

        self.N = int(meta["N"])
        self.T = int(meta["T"])
        self.C = int(meta["C"])
        self.X_path = os.path.join(root_path, 'X.dat')
        self.y_path = os.path.join(root_path, 'y.dat')

        # ------ open memmaps ------
        self.X_mem = np.memmap(self.X_path, dtype=np.float32, mode='r',
                               shape=(self.N, self.T, self.C))
        y_mem = np.memmap(self.y_path, dtype=np.float32, mode='r',
                          shape=(self.N, 3))

        # For compatibility with MultiDatasetsLoader
        self.X = self.X_mem

        # ------ build subject-level label table ------
        data_list = build_subject_label_table_from_y(y_mem)

        # ------ dataset-specific subject split ------
        # subclass need to implement _get_id_lists，Use data_list to get all_ids / train_ids / val_ids / test_ids
        a, b = args.ratio_a, args.ratio_b
        self.all_ids, self.train_ids, self.val_ids, self.test_ids = \
            self._get_id_lists(args, data_list, a, b)

        # ------ choose subject ids according to flag ------
        if flag == 'TRAIN':
            ids = self.train_ids
            print('Training ids:', ids)
            print('Number of training subjects:', len(ids))
        elif flag == 'VAL':
            ids = self.val_ids
            print('Validation ids:', ids)
            print('Number of validation subjects:', len(ids))
        elif flag == 'TEST':
            ids = self.test_ids
            print('Test ids:', ids)
            print('Number of test subjects:', len(ids))
        elif flag == 'PRETRAIN':
            ids = self.all_ids
            print('All ids:', ids)
            print('Number of all subjects:', len(ids))
        else:
            raise ValueError('Invalid flag. Please use TRAIN, VAL, TEST, or PRETRAIN.')

        ids = np.asarray(ids, dtype=int)

        # ------ select segments whose subject_id is in ids ------
        subj_ids_all = y_mem[:, 1].astype(int)
        mask = np.isin(subj_ids_all, ids)
        self.indices = np.where(mask)[0].astype(int)

        # keep y subset in memory
        self.y = np.asarray(y_mem[self.indices])   # shape: (N_sel, 3)

        # ------ sampling rate filtering ------
        sampling_rate_list = list(map(int, args.sampling_rate_list.split(",")))
        sampling_mask = np.isin(self.y[:, 2], sampling_rate_list)
        if sampling_mask.sum() == 0:
            print("Unique sampling rate in data:", np.unique(self.y[:, 2]))
            print("Target sampling_rate_list:", sampling_rate_list)
            raise RuntimeError("No matching sampling rates found.")
        self.indices = self.indices[sampling_mask]
        self.y = self.y[sampling_mask]

        # ------ dataset-specific label post-processing ------
        # subclass make modification based on classify_choice to further process self.y and self.indices
        self._postprocess_labels(args, flag)

        # ------ max sequence length ------
        self.max_seq_len = self.T

    # --------------- Two interfaces for subclass ---------------

    def _get_id_lists(self, args, data_list: np.ndarray, a: float, b: float):
        """
        Dataset-specific subject split logic.
        Must return (all_ids, train_ids, val_ids, test_ids).
        data_list: (N_subject, 2) => [disease_label, subject_id]
        """
        raise NotImplementedError

    def _postprocess_labels(self, args, flag: str):
        """
        Dataset-specific label post-processing logic (e.g., classify_choice mapping).
        Default: do nothing.
        """
        # default: no-op
        return

    # --------------- PyTorch Dataset API ---------------

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        real_idx = int(self.indices[index])

        x_np = self.X_mem[real_idx]  # (T, C)
        y_np = self.y[index]         # (3,)

        if not self.no_normalize:
            # normalize_batch_ts expects shape (B, T, C)
            x_np = normalize_batch_ts(x_np[np.newaxis, ...])[0]

        x = torch.from_numpy(np.asarray(x_np, dtype=np.float32))
        y = torch.from_numpy(np.asarray(y_np, dtype=np.float32))

        return x, y

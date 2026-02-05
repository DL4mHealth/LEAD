import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from data_provider.uea import (
    normalize_batch_ts,
    bandpass_filter_func,
)
import warnings
import random
import json
from data_provider.dataset_loader.base_loader import BaseLoader

warnings.filterwarnings('ignore')


def get_id_list_pearl_neuro(args, data_list: np.ndarray, a=0.6, b=0.8):
    '''
    Loads subject IDs for all, training, validation, and test sets for PEARL-Neuro data
    Args:
        args: arguments
        label_path: directory of label.npy file
        a: ratio of ids in training set
        b: ratio of ids in training and validation set
    Returns:
        all_ids: list of all IDs
        train_ids: list of IDs for training set
        val_ids: list of IDs for validation set
        test_ids: list of IDs for test set
    '''
    # random shuffle to break the potential influence of human named ID order,
    # e.g., put all healthy subjects first or put subjects with more samples first, etc.
    # (which could cause data imbalance in training, validation, and test sets)
    all_ids = list(data_list[:, 1])
    if args.cross_val == 'fixed':  # fixed split
        random.seed(42)
    elif args.cross_val == 'mccv':  # Monte Carlo cross-validation
        random.seed(args.seed)
    elif args.cross_val == '5-fold':
        pass
    else:
        raise ValueError('Invalid cross_val. Please use fixed or mccv.')
    random.shuffle(all_ids)
    train_ids = all_ids[:int(a * len(all_ids))]
    val_ids = all_ids[int(a * len(all_ids)):int(b * len(all_ids))]
    test_ids = all_ids[int(b * len(all_ids)):]

    return sorted(all_ids), sorted(train_ids), sorted(val_ids), sorted(test_ids)


class PEARLNeuroLoader(BaseLoader):
    def _get_id_lists(self, args, data_list: np.ndarray, a: float, b: float):
        # reuse existing splitting logic
        return get_id_list_pearl_neuro(args, data_list, a, b)

    def _postprocess_labels(self, args, flag: str):
        """
        Classification choice processing (same as original ADFTDLoader).
        disease label is in self.y[:, 0]
        """
        if flag != 'PRETRAIN':
            if args.classify_choice == 'ad_vs_hc':
                pass
            elif args.classify_choice == 'ad_vs_nonad':
                pass
            elif args.classify_choice == 'hc_vs_abnormal':
                pass
            elif args.classify_choice == 'multi_class':
                pass

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


def get_id_list_apava(args, data_list: np.ndarray, a=0.6, b=0.8):
    '''
    Loads subject IDs for all, training, validation, and test sets for APAVA data
    As APAVA dataset has predefined splits, we follow the predefined splits here.
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
    all_ids = list(data_list[:, 1])  # all subjects
    hc_list = list(data_list[np.where(data_list[:, 0] == 0)][:, 1])  # healthy IDs
    ad_list = list(data_list[np.where(data_list[:, 0] == 1)][:, 1])  # Alzheimer's disease IDs
    if args.cross_val == 'fixed' or args.cross_val == 'mccv':  # fixed split
        if args.cross_val == 'fixed':
            val_ids = [15, 16, 19, 20]  # 15, 19 are AD; 16, 20 are HC
            test_ids = [1, 2, 17, 18]  # 1, 17 are AD; 2, 18 are HC
            train_ids = [int(i) for i in all_ids if i not in val_ids + test_ids]
        else:
            raise NotImplementedError('MCCV not implemented yet for APAVA dataset.')

        return sorted(all_ids), sorted(train_ids), sorted(val_ids), sorted(test_ids)

    elif args.cross_val == 'loso':  # leave-one-subject-out
        all_ids = list(data_list[:, 1])  # all subjects, including subjects with other labels beyond AD and HC
        hc_ad_list = sorted(hc_list + ad_list)  # all subjects with AD and HC labels
        # take subject ID with index (args.seed-41) % len(all_ids) as test set, random seed start from 41
        test_ids = [hc_ad_list[(args.seed - 41) % len(hc_ad_list)]]
        train_ids = [id for id in hc_ad_list if id not in test_ids]
        # randomly take 10% of the training set as validation set
        random.seed(args.seed)
        random.shuffle(train_ids)
        val_ids = train_ids

        return sorted(all_ids), sorted(train_ids), sorted(val_ids), sorted(test_ids)
    else:
        raise ValueError('Invalid cross_val. Please use fixed or loso.')


class APAVALoader(BaseLoader):
    def _get_id_lists(self, args, data_list: np.ndarray, a: float, b: float):
        # reuse existing splitting logic
        return get_id_list_apava(args, data_list, a, b)

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

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


"""def get_id_list_adftd(args, label_path, a=0.6, b=0.8):
    '''
    Loads subject IDs for all, training, validation, and test sets for ADFTD data
    Use healthy, frontotemporal dementia, and Alzheimer's disease subjects
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
    data_list = np.load(label_path)
    all_ids = list(data_list[:, 1])  # all subjects
    hc_list = list(data_list[np.where(data_list[:, 0] == 0)][:, 1])  # healthy IDs
    ad_list = list(data_list[np.where(data_list[:, 0] == 1)][:, 1])  # Alzheimer's disease IDs
    ftd_list = list(data_list[np.where(data_list[:, 0] == 2)][:, 1])  # frontotemporal dementia IDs
    if args.cross_val == 'fixed' or args.cross_val == 'mccv':  # fixed split or Monte Carlo cross-validation
        if args.cross_val == 'fixed':
            random.seed(42)  # fixed seed for fixed split
        else:
            random.seed(args.seed)  # random seed for Monte Carlo cross-validation

        random.shuffle(hc_list)
        random.shuffle(ad_list)
        random.shuffle(ftd_list)

        train_ids = hc_list[:int(a * len(hc_list))] + ad_list[:int(a * len(ad_list))] + ftd_list[
                                                                                        :int(a * len(ftd_list))]
        val_ids = (hc_list[int(a * len(hc_list)):int(b * len(hc_list))] +
                   ad_list[int(a * len(ad_list)):int(b * len(ad_list))] +
                   ftd_list[int(a * len(ftd_list)):int(b * len(ftd_list))])
        test_ids = hc_list[int(b * len(hc_list)):] + ad_list[int(b * len(ad_list)):] + ftd_list[int(b * len(ftd_list)):]

        return sorted(all_ids), sorted(train_ids), sorted(val_ids), sorted(test_ids)

    elif args.cross_val == '5-fold':  # 5-fold cross-validation
        random.seed(42)
        # split data into 5 folds
        all_ids = list(data_list[:, 1])  # all subjects, including subjects with other labels beyond AD and HC
        random.shuffle(hc_list)
        random.shuffle(ad_list)
        random.shuffle(ftd_list)
        fold_size_hc = len(hc_list) / 5
        fold_size_ad = len(ad_list) / 5
        fold_size_ftd = len(ftd_list) / 5
        seed = (args.seed-41) % 5
        # take 1-fold as test set
        test_ids = (hc_list[int(seed * fold_size_hc):int((seed + 1) * fold_size_hc)] +
                    ad_list[int(seed * fold_size_ad):int((seed + 1) * fold_size_ad)] +
                    ftd_list[int(seed * fold_size_ftd):int((seed + 1) * fold_size_ftd)])
        # take another 1-fold as validation set
        if seed == 4:  # take the first fold as validation set when seed is 4
            val_ids = (hc_list[:int(fold_size_hc)] + ad_list[:int(fold_size_ad)]) + ftd_list[:int(fold_size_ftd)]
        else:
            val_ids = (hc_list[int((seed + 1) * fold_size_hc):int((seed + 2) * fold_size_hc)] +
                       ad_list[int((seed + 1) * fold_size_ad):int((seed + 2) * fold_size_ad)] +
                       ftd_list[int((seed + 1) * fold_size_ftd):int((seed + 2) * fold_size_ftd)])
        # take the remaining ids not in test and validation sets as training set
        train_ids = [id for id in all_ids if id not in test_ids and id not in val_ids]

        return sorted(all_ids), sorted(train_ids), sorted(val_ids), sorted(test_ids)

    elif args.cross_val == 'loso':  # leave-one-subject-out cross-validation
        if args.classify_choice == 'ad_vs_hc':
            hc_ad_list = sorted(hc_list + ad_list)  # all subjects with AD and HC labels
            # take subject ID with index (args.seed-41) % len(all_ids) as test set, random seed start from 41
            test_ids = [hc_ad_list[(args.seed - 41) % len(hc_ad_list)]]
            train_ids = [id for id in hc_ad_list if id not in test_ids]
        else:
            # take subject ID with index (args.seed-41) % len(all_ids) as test set, random seed start from 41
            all_ids = sorted(all_ids)
            test_ids = [all_ids[(args.seed - 41) % len(all_ids)]]
            train_ids = [id for id in all_ids if id not in test_ids]
        # randomly take 10% of the training set as validation set
        random.seed(args.seed)
        random.shuffle(train_ids)
        val_ids = train_ids

        return sorted(all_ids), sorted(train_ids), sorted(val_ids), sorted(test_ids)
    else:
        raise ValueError('Invalid cross_val. Please use fixed, mccv, or loso.')


class ADFTDLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.no_normalize = args.no_normalize
        self.root_path = root_path
        self.data_path = os.path.join(root_path, 'Feature/')
        self.label_path = os.path.join(root_path, 'Label/label.npy')

        a, b = args.ratio_a, args.ratio_b
        self.all_ids, self.train_ids, self.val_ids, self.test_ids = get_id_list_adftd(args, self.label_path, a, b)

        if flag == 'TRAIN':
            ids = self.train_ids
            print('train ids:', ids)
        elif flag == 'VAL':
            ids = self.val_ids
            print('val ids:', ids)
        elif flag == 'TEST':
            ids = self.test_ids
            print('test ids:', ids)
        elif flag == 'PRETRAIN':
            ids = self.all_ids
            print('all ids:', ids)
        else:
            raise ValueError('Invalid flag. Please use TRAIN, VAL, TEST, or PRETRAIN.')

        self.X, self.y = load_data_by_ids(self.data_path, self.label_path, ids, args)
        if flag != 'PRETRAIN':   # no label processing for pretraining, use all data
            if args.classify_choice == 'ad_vs_hc':  # 1 vs 0
                # delete FTD
                print('Delete FTD subjects in train ids list for AD vs HC classification')
                mask = (self.y[:, 0] < 2)
                self.X = self.X[mask]
                self.y = self.y[mask]
            elif args.classify_choice == 'ad_vs_nonad':
                # all other diseases are also 0, change FTD label from 2 ot 0
                print('Change FTD label from 2 to 0 in train ids list for AD vs non-AD classification')
                self.y[:, 0] = np.where(self.y[:, 0] > 1, 0, self.y[:, 0])
            elif args.classify_choice == 'hc_vs_abnormal':
                # change all other diseases to 1
                print('Change FTD label from 2 to 1 in train ids list for HC vs abnormal classification')
                self.y[:, 0] = np.where(self.y[:, 0] > 1, 1, self.y[:, 0])
            elif args.classify_choice == 'multi_class':
                print('Keep all subjects in train ids list for multi-class classification')
                pass
        # self.X = bandpass_filter_func(self.X, fs=args.sampling_rate, lowcut=args.low_cut, highcut=args.high_cut)
        self.X = normalize_batch_ts(self.X)

        self.X = self.X.astype(np.float32)
        self.y = self.y.astype(np.float32)

        self.max_seq_len = self.X.shape[1]

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), \
               torch.from_numpy(np.asarray(self.y[index]))

    def __len__(self):
        return len(self.y)"""


def get_id_list_adftd(args, data_list: np.ndarray, a=0.6, b=0.8):
    """
    Same logic as get_id_list_adftd, but data_list is given directly.
    data_list: shape (N_subject, 2) => [disease_label, subject_id]
    """
    # all subjects
    all_ids = list(data_list[:, 1])
    hc_list = list(data_list[np.where(data_list[:, 0] == 0)][:, 1])   # healthy IDs (label == 0)
    ad_list = list(data_list[np.where(data_list[:, 0] == 1)][:, 1])  # Alzheimer's disease IDs (label == 1)
    ftd_list = list(data_list[np.where(data_list[:, 0] == 2)][:, 1])  # frontotemporal dementia IDs (label == 2)

    if args.cross_val == 'fixed' or args.cross_val == 'mccv':
        # fixed split or Monte Carlo cross-validation
        if args.cross_val == 'fixed':
            random.seed(42)
        else:
            random.seed(args.seed)

        random.shuffle(hc_list)
        random.shuffle(ad_list)
        random.shuffle(ftd_list)

        train_ids = (hc_list[:int(a * len(hc_list))] +
                     ad_list[:int(a * len(ad_list))] +
                     ftd_list[:int(a * len(ftd_list))])
        val_ids = (hc_list[int(a * len(hc_list)):int(b * len(hc_list))] +
                   ad_list[int(a * len(ad_list)):int(b * len(ad_list))] +
                   ftd_list[int(a * len(ftd_list)):int(b * len(ftd_list))])
        test_ids = (hc_list[int(b * len(hc_list)):] +
                    ad_list[int(b * len(ad_list)):] +
                    ftd_list[int(b * len(ftd_list)):])

        return sorted(all_ids), sorted(train_ids), sorted(val_ids), sorted(test_ids)

    elif args.cross_val == 'loso':
        # leave-one-subject-out cross-validation
        if args.classify_choice == 'ad_vs_hc':
            hc_ad_list = sorted(hc_list + ad_list)
            test_ids = [hc_ad_list[(args.seed - 41) % len(hc_ad_list)]]
            train_ids = [sid for sid in hc_ad_list if sid not in test_ids]
        else:
            all_ids = sorted(all_ids)
            test_ids = [all_ids[(args.seed - 41) % len(all_ids)]]
            train_ids = [sid for sid in all_ids if sid not in test_ids]

        random.seed(args.seed)
        random.shuffle(train_ids)
        val_ids = train_ids

        return sorted(all_ids), sorted(train_ids), sorted(val_ids), sorted(test_ids)

    else:
        raise ValueError('Invalid cross_val. Please use fixed, mccv, or loso.')


class ADFTDLoader(BaseLoader):
    """
    Memmap-based loader for ADFTD.
    Only needs to implement:
      - _get_id_lists   (subject split rule)
      - _postprocess_labels (classify_choice mapping)
    """
    def _get_id_lists(self, args, data_list: np.ndarray, a: float, b: float):
        # reuse existing splitting logic
        return get_id_list_adftd(args, data_list, a, b)

    def _postprocess_labels(self, args, flag: str):
        """
        Classification choice processing (same as original ADFTDLoader).
        disease label is in self.y[:, 0]
        """
        if flag != 'PRETRAIN':
            if args.classify_choice == 'ad_vs_hc':
                # delete FTD (label 2)
                print('Delete FTD subjects (label=2) for AD vs HC classification')
                label_mask = (self.y[:, 0] < 2)
                self.indices = self.indices[label_mask]
                self.y = self.y[label_mask]
            elif args.classify_choice == 'ad_vs_nonad':
                # change FTD label from 2 to 0
                print('Change FTD label from 2 to 0 for AD vs non-AD classification')
                self.y[:, 0] = np.where(self.y[:, 0] > 1, 0, self.y[:, 0])
            elif args.classify_choice == 'hc_vs_abnormal':
                # change all other diseases to 1
                print('Change non-HC label (>0) to 1 for HC vs abnormal classification')
                self.y[:, 0] = np.where(self.y[:, 0] > 1, 1, self.y[:, 0])
            elif args.classify_choice == 'multi_class':
                print('Keep all subjects for multi-class classification')

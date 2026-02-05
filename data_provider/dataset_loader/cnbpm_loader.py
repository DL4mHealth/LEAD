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


def get_id_list_cnbpm(args, data_list: np.ndarray, a=0.6, b=0.8):
    '''
    Loads subject IDs for all, training, validation, and test sets for CNBPM data
    Use healthy, Alzheimer's disease, and mild cognitive impairment subjects
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
    mci_list = list(data_list[np.where(data_list[:, 0] == 2)][:, 1])  # mild cognitive impairment IDs
    if args.cross_val == 'fixed' or args.cross_val == 'mccv':  # fixed split or Monte Carlo cross-validation
        if args.cross_val == 'fixed':
            random.seed(42)  # fixed seed for fixed split
        else:
            random.seed(args.seed)  # random seed for Monte Carlo cross-validation

        random.shuffle(hc_list)
        random.shuffle(ad_list)
        random.shuffle(mci_list)

        train_ids = hc_list[:int(a * len(hc_list))] + ad_list[:int(a * len(ad_list))] + mci_list[
                                                                                        :int(a * len(mci_list))]
        val_ids = (hc_list[int(a * len(hc_list)):int(b * len(hc_list))] +
                   ad_list[int(a * len(ad_list)):int(b * len(ad_list))] +
                   mci_list[int(a * len(mci_list)):int(b * len(mci_list))])
        test_ids = hc_list[int(b * len(hc_list)):] + ad_list[int(b * len(ad_list)):] + mci_list[int(b * len(mci_list)):]

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


class CNBPMLoader(BaseLoader):
    def _get_id_lists(self, args, data_list: np.ndarray, a: float, b: float):
        # reuse existing splitting logic
        return get_id_list_cnbpm(args, data_list, a, b)

    def _postprocess_labels(self, args, flag: str):
        """
        Classification choice processing (same as original ADFTDLoader).
        disease label is in self.y[:, 0]
        """
        if flag != 'PRETRAIN':
            if args.classify_choice == 'ad_vs_hc':
                # delete FTD (label 2)
                print('Delete MCI subjects (label=2) for AD vs HC classification')
                label_mask = (self.y[:, 0] < 2)
                self.indices = self.indices[label_mask]
                self.y = self.y[label_mask]
            elif args.classify_choice == 'ad_vs_nonad':
                # change MCI label from 2 to 0
                print('Change MCI label from 2 to 0 for AD vs non-AD classification')
                self.y[:, 0] = np.where(self.y[:, 0] > 1, 0, self.y[:, 0])
            elif args.classify_choice == 'hc_vs_abnormal':
                # change all other diseases to 1
                print('Change non-HC label (>0) to 1 for HC vs abnormal classification')
                self.y[:, 0] = np.where(self.y[:, 0] > 1, 1, self.y[:, 0])
            elif args.classify_choice == 'hc_vs_mci':
                # delete AD (label 1), change MCI label from 2 to 1
                print('Delete AD subjects (label=1) and change MCI label from 2 to 1 for HC vs MCI classification')
                label_mask = (self.y[:, 0] != 1)
                self.indices = self.indices[label_mask]
                self.y = self.y[label_mask]
                self.y[:, 0] = np.where(self.y[:, 0] > 1, 1, self.y[:, 0])
            elif args.classify_choice == 'ad_vs_mci':
                # delete HC (label 0), change AD label from 1 to 0, change MCI label from 2 to 1
                print('Delete HC subjects (label=0), change AD label from 1 to 0, and change MCI label from 2 to 1, '
                      ' for AD vs MCI classification')
                label_mask = (self.y[:, 0] != 0)
                self.indices = self.indices[label_mask]
                self.y = self.y[label_mask]
                self.y[:, 0] = np.where(self.y[:, 0] == 1, 0, self.y[:, 0])
                self.y[:, 0] = np.where(self.y[:, 0] == 2, 1, self.y[:, 0])
            elif args.classify_choice == 'multi_class':
                print('Keep all subjects for multi-class classification')

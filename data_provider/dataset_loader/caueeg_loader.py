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
from data_provider.dataset_loader.base_loader import BaseLoader

warnings.filterwarnings('ignore')


def get_id_list_caueeg(args, data_list: np.ndarray, a=0.6, b=0.8):
    """
    Same logic as get_id_list_adftd, but data_list is given directly.
    data_list: shape (N_subject, 2) => [disease_label, subject_id]
    """
    # all subjects
    all_ids = list(data_list[:, 1])  # all subjects
    hc_list = list(data_list[np.where(data_list[:, 0] == 0)][:, 1])  # healthy IDs
    de_list = list(data_list[np.where(data_list[:, 0] == 1)][:, 1])  # Dementia IDs
    mci_list = list(data_list[np.where(data_list[:, 0] == 2)][:, 1])  # Mild cognitive impairment IDs
    ot_list = list(data_list[np.where(data_list[:, 0] == 3)][:, 1])  # Other IDs
    if args.cross_val == 'fixed' or args.cross_val == 'mccv':  # fixed split or Monte Carlo cross-validation
        if args.cross_val == 'fixed':
            random.seed(42)  # fixed seed for fixed split
        else:
            random.seed(args.seed)  # random seed for Monte Carlo cross-validation

        random.shuffle(hc_list)
        random.shuffle(de_list)
        random.shuffle(mci_list)
        random.shuffle(ot_list)

        train_ids = (hc_list[:int(a * len(hc_list))] +
                     de_list[:int(a * len(de_list))] +
                     mci_list[:int(a * len(mci_list))] +
                     ot_list[:int(a * len(ot_list))])
        val_ids = (hc_list[int(a * len(hc_list)):int(b * len(hc_list))] +
                   de_list[int(a * len(de_list)):int(b * len(de_list))] +
                   mci_list[int(a * len(mci_list)):int(b * len(mci_list))] +
                   ot_list[int(a * len(ot_list)):int(b * len(ot_list))])
        test_ids = (hc_list[int(b * len(hc_list)):] +
                    de_list[int(b * len(de_list)):] +
                    mci_list[int(b * len(mci_list)):] +
                    ot_list[int(b * len(ot_list)):])

        return sorted(all_ids), sorted(train_ids), sorted(val_ids), sorted(test_ids)

    elif args.cross_val == 'loso':  # leave-one-subject-out cross-validation
        if args.classify_choice == 'ad_vs_hc':
            hc_ad_list = sorted(hc_list + de_list)  # all subjects with AD and HC labels
            # take subject ID with index (args.seed-41) % len(all_ids) as test set, random seed start from 41
            test_ids = [hc_ad_list[(args.seed - 41) % len(hc_ad_list)]]
            train_ids = [id for id in hc_ad_list if id not in test_ids]
        else:
            # take subject ID with index (args.seed-41) % len(all_ids) as test set, random seed start from 41
            all_ids = sorted(all_ids)
            test_ids = [all_ids[(args.seed - 41) % len(all_ids)]]
            train_ids = [id for id in all_ids if id not in test_ids]
        # randomly take 20% of the training set as validation set
        random.seed(args.seed)
        random.shuffle(train_ids)
        val_ids = train_ids

        return sorted(all_ids), sorted(train_ids), sorted(val_ids), sorted(test_ids)
    else:
        raise ValueError('Invalid cross_val. Please use fixed, mccv, or loso.')


class CAUEEGLoader(BaseLoader):
    """
    Memmap-based loader for CAUEEG.
    Only needs to implement:
      - _get_id_lists   (subject split rule)
      - _postprocess_labels (classify_choice mapping)
    """
    def _get_id_lists(self, args, data_list: np.ndarray, a: float, b: float):
        # reuse existing splitting logic
        return get_id_list_caueeg(args, data_list, a, b)

    def _postprocess_labels(self, args, flag: str):
        """
        Classification choice processing (same as original ADFTDLoader).
        disease label is in self.y[:, 0]
        """
        if flag != 'PRETRAIN':   # no label processing for pretraining, use all data
            if args.classify_choice == 'ad_vs_hc':  # 1 vs 0
                # delete all other diseases except Dementia and HC
                print('Delete MCI and other subjects in train ids list for dementia vs HC classification')
                label_mask = (self.y[:, 0] < 2)
                self.indices = self.indices[label_mask]
                self.y = self.y[label_mask]
            elif args.classify_choice == 'ad_vs_nonad':
                # all other diseases are also 0, change labels to 0
                print('Change MCI and other subjects from 2,3 to 0 '
                      'in train ids list for dementia vs non-dementia classification')
                self.y[:, 0] = np.where(self.y[:, 0] > 1, 0, self.y[:, 0])
            elif args.classify_choice == 'hc_vs_abnormal':
                # remove all other diseases except HC, Dementia, and MCI
                label_mask = (self.y[:, 0] < 3)
                self.indices = self.indices[label_mask]
                self.y = self.y[label_mask]
                print('Change MCI subjects from 2 to 1 in train ids list for HC vs abnormal classification')
                self.y[:, 0] = np.where(self.y[:, 0] > 1, 1, self.y[:, 0])
            elif args.classify_choice == 'hc_vs_mci':
                # delete AD (label 1), change MCI label from 2 to 1
                print('Delete AD subjects (label=1) and other subjects (label=3) '
                      'and change MCI label from 2 to 1 for HC vs MCI classification')
                label_mask = (self.y[:, 0] != 1) & (self.y[:, 0] != 3)
                self.indices = self.indices[label_mask]
                self.y = self.y[label_mask]
                self.y[:, 0] = np.where(self.y[:, 0] > 1, 1, self.y[:, 0])
            elif args.classify_choice == 'multi_class':
                print('Keep HC, dementia, MCI in train ids list for three-class classification')
                label_mask = (self.y[:, 0] < 3)
                self.indices = self.indices[label_mask]
                self.y = self.y[label_mask]

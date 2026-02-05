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
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from scipy.signal import resample

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
from data_provider.dataset_loader.tuep_loader import TUEPLoader


# data folder dict to loader mapping
data_folder_dict = {
    # should use the same name as the dataset folder
    # For datasets that the raw channel number is 19, there is no -19 suffix in the dataset name
    
    # 4 downstream datasets 
    'ADFTD': ADFTDLoader,
    'ADFTD-RS': ADFTDLoader,  # resting-state subset of ADFTD
    'ADFTD-PS': ADFTDLoader,  # photo-stimulation subset of ADFTD
    'CNBPM': CNBPMLoader,
    'Cognision-RS': CognisionRSLoader,
    'APAVA': APAVALoader,
    'ADFSU': ADFSULoader,
    'ADSZ': ADSZLoader,

    # 13 pretraining datasets
    'AD-Auditory': ADAuditoryLoader,
    'BACA-RS': BACARSLoader,
    'BrainLat': BrainLatLoader,
    'Depression': DepressionLoader,
    'FEPCR': FEPCRLoader,
    'MCEF-RS': MCEFRSLoader,
    'P-ADIC': PADICLoader,
    'PD-RS': PDRSLoader,
    'PEARL-Neuro': PEARLNeuroLoader,
    'SRM-RS': SRMRSLoader,
    'TDBrain': TBDRAINLoader,
    'TUEP': TUEPLoader,
    'CAUEEG': CAUEEGLoader,
}
warnings.filterwarnings('ignore')


class MultiDatasetsLoader(Dataset):
    """
    Index-based multi-dataset loader:
      - Do NOT concatenate all X into memory.
      - Keep a list of child datasets and a global index list [(ds_idx, local_idx), ...].
      - Maintain subject-id offsets to avoid collisions across datasets.
      - Build global_sids for samplers that need subject grouping.
    """
    def __init__(self, args, root_path, flag=None):
        self.no_normalize = args.no_normalize
        self.root_path = root_path

        print(f"Loading {flag} samples from multiple datasets...")
        if flag == 'PRETRAIN':
            data_folder_list = args.pretraining_datasets.split(",")
        elif flag == 'TRAIN':
            data_folder_list = args.training_datasets.split(",")
        elif flag in ('TEST', 'VAL'):
            data_folder_list = args.testing_datasets.split(",")
        else:
            raise ValueError("flag must be PRETRAIN, TRAIN, VAL, or TEST")
        print("Datasets used ", data_folder_list)

        self.datasets = []      # child datasets
        self.index = []         # [(ds_idx, local_idx)]
        self.sid_offset = []    # offset for subject ids of each child dataset

        global_ids_range = 1  # running offset to avoid duplicate subject IDs

        for i, data in enumerate(data_folder_list):
            if data not in data_folder_dict.keys():
                raise ValueError("Data not matched, please check data_folder_dict keys.")
            print("Start loading data from dataset: ", data)
            Data = data_folder_dict[data]
            ds = Data(
                root_path=os.path.join(args.root_path, data),
                args=args,
                flag=flag,
            )

            # Add dataset-id as the last column of y (1-based dataset id)
            # y is expected to be [label, subject_id, ...]; we append dataset_id.
            ds.y = np.concatenate(
                (ds.y, np.full((ds.y.shape[0], 1), i + 1, dtype=ds.y.dtype)),
                axis=1
            )
            print(f"Number of samples in dataset {data} {flag} set: {ds.y.shape[0]}\n")

            # Compute current dataset subject-id span.
            # Use max(len(all_ids), max(all_ids)) to handle non-contiguous / non-1-based ids.
            current_ids_range = max(len(ds.all_ids), max(ds.all_ids))

            # Record offset BEFORE adding this dataset (offset to be applied to its subject ids).
            self.sid_offset.append(global_ids_range)

            # Build global index entries for this child dataset
            base_ds_idx = len(self.datasets)
            for j in range(len(ds)):
                self.index.append((base_ds_idx, j))

            # Advance the global offset
            global_ids_range += current_ids_range

            # Register child dataset
            self.datasets.append(ds)

        # Shuffle the index once in a reproducible way (sampler may reshuffle per epoch)
        rng = np.random.default_rng(seed=42)
        rng.shuffle(self.index)

        # Precompute global subject ids for every sample index for samplers
        # NOTE: subject id is assumed at y[:, 1]
        self.global_sids = []
        for ds_idx, j in self.index:
            raw_sid = int(self.datasets[ds_idx].y[j, 1])
            self.global_sids.append(raw_sid + self.sid_offset[ds_idx])

        # For compatibility: expose the maximum sequence length among child datasets
        self.max_seq_len = max(getattr(d, "max_seq_len", 0) for d in self.datasets)

        # Derive channel dim (enc_in) from child datasets
        # NOTE: we assume all child datasets share the same channel count
        self.enc_in = max(getattr(d, "X").shape[2] for d in self.datasets)

        # Derive number of classes from child datasets' labels (y[:, 0])
        # This concatenates 1D label arrays only, which is memory-light.
        all_labels_1d = np.concatenate([d.y[:, 0] for d in self.datasets], axis=0)
        self.num_class = int(np.unique(all_labels_1d).shape[0])

        print()  # spacing

    def __len__(self):
        # Number of samples across all child datasets
        return len(self.index)

    def __getitem__(self, k):
        """
        Returns:
            x (torch.Tensor): sample tensor from child dataset
            y (torch.Tensor): label tensor; its subject id will be shifted to global space
        """
        ds_idx, j = self.index[k]
        x, y = self.datasets[ds_idx][j]  # x: torch.Tensor, y: torch.Tensor (at least [label, sid, ...])

        # Shift subject-id to global space to avoid collisions across datasets.
        y = y.clone()
        # y format: [label, subject_id, ..., dataset_id]; shift subject_id at index 1
        y[1] = y[1] + self.sid_offset[ds_idx]

        return x, y

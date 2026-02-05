from data_provider.data_loader import MultiDatasetsLoader

from data_provider.uea import collate_fn
from torch.utils.data import DataLoader
from utils.tools import CustomGroupSampler
import torch
import numpy as np
import random

# data type dict to loader mapping
data_type_dict = {
    # loading multiple datasets, concatenating them
    'MultiDatasets': MultiDatasetsLoader,  # datasets folder names presented in args.data_folder_list
}


# Control random seed for each worker
def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def data_provider(args, flag):
    Data = data_type_dict[args.data]

    if flag in ('VAL', 'TEST'):
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    if args.task_name in ('supervised', 'pretrain_lead', 'finetune'):
        data_set = Data(root_path=args.root_path, args=args, flag=flag)

        if args.group_shuffle:
            sampler = CustomGroupSampler(data_set, batch_size=args.batch_size, group_size=args.group_size)
            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                sampler=sampler,            # use sampler instead of shuffle
                num_workers=args.num_workers,
                worker_init_fn=worker_init_fn,
                pin_memory=True,
                drop_last=drop_last,
                collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
            )
        else:
            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                worker_init_fn=worker_init_fn,
                pin_memory=True,
                drop_last=drop_last,
                collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
            )

        return data_set, data_loader

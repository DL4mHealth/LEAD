from torch.utils.data import DataLoader

from data_provider.uea import collate_fn
from data_provider.data_loader import (
    MultiDataset,
    DatasetHomogeneousBatchSampler,
    SubjectRateGroupedDatasetBatchSampler,
)


data_type_dict = {
    "MultiDatasets": MultiDataset,
}


def data_provider(args, flag):
    Data = data_type_dict[args.data]
    data_set = Data(root_path=args.root_path, args=args, flag=flag)

    normalized_flag = str(flag).upper()
    shuffle_flag = normalized_flag in ["TRAIN", "PRETRAIN"]
    drop_last = normalized_flag == "TRAIN"

    use_group_shuffle = bool(getattr(args, "group_shuffle", False))
    is_training_flag = normalized_flag in ["TRAIN", "PRETRAIN"]

    if use_group_shuffle and is_training_flag:
        # Explicit opt-in for any training stage (pre-training, supervised
        # training, fine-tuning, or probing): organize samples by
        # (subject_id, sampling_rate). group_size is meaningful whenever
        # --group_shuffle is enabled.
        batch_sampler = SubjectRateGroupedDatasetBatchSampler(
            dataset=data_set,
            batch_size=args.batch_size,
            group_size=args.group_size,
            drop_last=drop_last,
            shuffle=True,
            seed=getattr(args, "seed", 42),
            sequential_first_epoch=False,
        )

        data_loader = DataLoader(
            data_set,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    elif normalized_flag == "PRETRAIN":
        # Default PRETRAIN behavior without --group_shuffle:
        # no subject/rate grouping. Datasets with identical T/C/montage/channel
        # names/order may share a batch; incompatible geometries stay separate.
        # group_size is intentionally ignored in this branch.
        batch_sampler = DatasetHomogeneousBatchSampler(
            dataset=data_set,
            batch_size=args.batch_size,
            drop_last=drop_last,
            shuffle=shuffle_flag,
            seed=getattr(args, "seed", 42),
            sequential_first_epoch=False,
        )

        data_loader = DataLoader(
            data_set,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    else:
        # TRAIN/VAL/TEST are guaranteed to contain exactly one downstream dataset.
        # TRAIN reaches this branch only when --group_shuffle is disabled.
        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=shuffle_flag,
            drop_last=drop_last,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    return data_set, data_loader

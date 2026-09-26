import random
import numpy as np

from data_provider.dataset_loader.base_loader import BaseLoader
from data_provider.uea import limit_samples_per_subject


def get_id_list_tueg(args, data_list: np.ndarray, a=0.6, b=0.8):
    """Return subject-level TRAIN/VAL/TEST splits for TUEG/TUEP."""
    all_ids = list(data_list[:, 1])
    if args.cross_val == "fixed":
        random.seed(42)
    elif args.cross_val == "mccv":
        random.seed(args.seed)
    elif args.cross_val == "5-fold":
        # Preserve the existing project behavior for now; this option is not a
        # true K-fold split in the generic loaders.
        pass
    else:
        raise ValueError("Invalid cross_val. Please use fixed, mccv, or 5-fold.")

    random.shuffle(all_ids)
    train_ids = all_ids[: int(a * len(all_ids))]
    val_ids = all_ids[int(a * len(all_ids)) : int(b * len(all_ids))]
    test_ids = all_ids[int(b * len(all_ids)) :]
    return sorted(all_ids), sorted(train_ids), sorted(val_ids), sorted(test_ids)


class TUEGLoader(BaseLoader):
    """TUEG/TUEP loader using the common LEAD memmap/meta format.

    y.dat is expected to contain exactly three float32 columns:
        [disease_label, subject_id, sampling_rate]
    """

    def _get_id_lists(self, args, data_list: np.ndarray, a: float, b: float):
        return get_id_list_tueg(args, data_list, a, b)

    def _postprocess_labels(self, args, flag: str):
        # TUEG contains many more segments per subject than most LEAD datasets.
        # Keep the original cap, but y no longer needs any column remapping.
        self.indices, self.y = limit_samples_per_subject(
            self.indices,
            self.y,
            max_samples=args.sample_per_subject,
            subject_col=1,
        )

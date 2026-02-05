from torch.utils.data import Sampler
import numpy as np
import torch
import matplotlib.pyplot as plt
import math
import os
import random
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
import mne

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate / 2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Metric score decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...\n')
        try:
            torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        except Exception as e:
            print(f"Error saving model: {e}")
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def multiclass_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    num_classes = cm.shape[0]
    specificities = []

    for i in range(num_classes):
        # TN: sum of all except row i and col i
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        # FP: sum of column i except cm[i, i]
        fp = np.sum(cm[:, i]) - cm[i, i]

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificities.append(specificity)

    return np.mean(specificities)


class CustomGroupSampler(Sampler):
    """
    Group samples by subject ids:
      1) Sort indices by subject id (stable).
      2) Split into groups of size `group_size`.
      3) Shuffle the groups.
      4) Flatten, then bucketize by `batch_size` and shuffle within each bucket.

    It rebuilds the order at every epoch in __iter__.

    Compatible with:
      - dataset.global_sids (preferred; index-aligned global subject ids)
      - dataset.y[:, 1]     (fallback; requires dataset.y to be materialized)
    """
    def __init__(self, dataset, batch_size=128, group_size=2):
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.group_size = int(group_size)

        # Prefer global_sids prepared by MultiDatasetsLoader (index-aligned)
        if hasattr(dataset, "global_sids"):
            self.subject_ids = np.asarray(dataset.global_sids)
        else:
            # Fallback: requires dataset.y to exist in memory (not recommended for big data)
            self.subject_ids = np.asarray(dataset.y)[:, 1]

        assert len(self.subject_ids) == len(self.dataset), \
            "Length of subject_ids must match dataset length."

    def __len__(self):
        return len(self.dataset)

    def _make_order(self):
        # Stable sort by subject id
        sorted_indices = np.argsort(self.subject_ids, kind="mergesort")
        N = len(sorted_indices)

        # Group into chunks of size group_size (keep tail)
        groups = [sorted_indices[i:i + self.group_size] for i in range(0, N, self.group_size)]

        # Shuffle groups
        rng = np.random.default_rng()
        rng.shuffle(groups)

        # Flatten
        order = np.concatenate(groups, axis=0)

        # Bucketize by batch_size and shuffle within buckets
        batches = [order[i:i + self.batch_size] for i in range(0, len(order), self.batch_size)]
        for b in batches:
            rng.shuffle(b)
        order = np.concatenate(batches, axis=0)

        return order

    def __iter__(self):
        # Rebuild a different order each epoch
        order = self._make_order()
        return iter(order.tolist())


def calculate_subject_level_metrics(predictions, true_labels, subject_ids, num_classes):
    # Step 1: Get unique subject_ids
    unique_subjects = np.unique(subject_ids)

    # Step 2: Aggregate predictions and true labels for each subject_id
    subject_predictions = []
    subject_trues = []

    for subject in unique_subjects:
        # Find all sample indices for the current subject
        indices = np.where(subject_ids == subject)[0]

        # Get predictions and true labels for the current subject
        subject_preds = predictions[indices]
        subject_true = true_labels[indices][0]  # The true_label should be the same for all samples of a subject

        # Determine the majority vote prediction for the subject
        majority_label = Counter(subject_preds).most_common(1)[0][0]

        # Record subject-level results
        subject_predictions.append(majority_label)
        subject_trues.append(subject_true)

    # Convert to numpy arrays
    subject_predictions = np.array(subject_predictions)
    subject_trues = np.array(subject_trues)

    # Step 3: Calculate metrics
    # Convert true labels to one-hot encoding for AUROC and AUPRC
    subject_true_onehot = label_binarize(subject_trues, classes=list(range(num_classes)))
    subject_probs = label_binarize(subject_predictions, classes=list(range(num_classes)))  # Simplified assumption

    metrics = {"Accuracy": accuracy_score(subject_trues, subject_predictions)}
    # Check how many unique classes are present in the true labels
    unique_labels = np.unique(subject_trues)
    if len(unique_labels) < 2:
        # If there is only one class(e,g, leave-one-subject-out validation),
        # subject-level AUROC and AUPRC are meaningless
        metrics.update({"Precision": -1, "Recall": -1, "Specificity": -1, "F1": -1, "AUROC": -1, "AUPRC": -1})
    else:
        metrics["Precision"] = precision_score(subject_trues, subject_predictions, average="macro")
        metrics["Recall"] = recall_score(subject_trues, subject_predictions, average="macro")
        metrics["Specificity"] = multiclass_specificity(subject_trues, subject_predictions)
        metrics["F1"] = f1_score(subject_trues, subject_predictions, average="macro")
        metrics["AUROC"] = roc_auc_score(subject_true_onehot, subject_probs, multi_class="ovr")
        metrics["AUPRC"] = average_precision_score(subject_true_onehot, subject_probs, average="macro")

    return metrics


def compute_avg_std(args, sample_val_metrics_dict_list, subject_val_metrics_dict_list,
                    sample_test_metrics_dict_list, subject_test_metrics_dict_list, total_params):
    print('>>>>>>>average testing<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    # Compute average and std for sample-level metrics
    sample_val_metrics_dict_avg_std = {}
    sample_test_metrics_dict_avg_std = {}
    for key in sample_val_metrics_dict_list[0].keys():
        # convert to percentage
        sample_val_avg = np.mean([val_metrics_dict[key] for val_metrics_dict in sample_val_metrics_dict_list]) * 100
        sample_val_std = np.std([val_metrics_dict[key] for val_metrics_dict in sample_val_metrics_dict_list]) * 100
        sample_test_avg = np.mean([test_metrics_dict[key] for test_metrics_dict in sample_test_metrics_dict_list]) * 100
        sample_test_std = np.std([test_metrics_dict[key] for test_metrics_dict in sample_test_metrics_dict_list]) * 100

        sample_val_metrics_dict_avg_std[key] = (sample_val_avg, sample_val_std)
        sample_test_metrics_dict_avg_std[key] = (sample_test_avg, sample_test_std)

    # Format results into a single line with two decimal places and percentages
    sample_val_results = "Validation results --- " + ", ".join(
        [f"{key}: {sample_val_metrics_dict_avg_std[key][0]:.2f}+-{sample_val_metrics_dict_avg_std[key][1]:.2f}%"
         for key in sample_val_metrics_dict_avg_std.keys()]
    )
    sample_test_results = "Test results --- " + ", ".join(
        [f"{key}: {sample_test_metrics_dict_avg_std[key][0]:.2f}+-{sample_test_metrics_dict_avg_std[key][1]:.2f}%"
         for key in sample_test_metrics_dict_avg_std.keys()]
    )

    if args.use_subject_vote:
        # Compute average and std for subject-level metrics
        subject_val_metrics_dict_avg_std = {}
        subject_test_metrics_dict_avg_std = {}
        for key in subject_val_metrics_dict_list[0].keys():
            subject_val_avg = np.mean([val_metrics_dict[key] for val_metrics_dict in subject_val_metrics_dict_list]) * 100
            subject_val_std = np.std([val_metrics_dict[key] for val_metrics_dict in subject_val_metrics_dict_list]) * 100
            subject_test_avg = np.mean([test_metrics_dict[key] for test_metrics_dict in subject_test_metrics_dict_list]) * 100
            subject_test_std = np.std([test_metrics_dict[key] for test_metrics_dict in subject_test_metrics_dict_list]) * 100

            subject_val_metrics_dict_avg_std[key] = (subject_val_avg, subject_val_std)
            subject_test_metrics_dict_avg_std[key] = (subject_test_avg, subject_test_std)

        subject_val_results = "Validation results --- " + ", ".join(
            [f"{key}: {subject_val_metrics_dict_avg_std[key][0]:.2f}+-{subject_val_metrics_dict_avg_std[key][1]:.2f}%"
             for key in subject_val_metrics_dict_avg_std.keys()]
        )
        subject_test_results = "Test results --- " + ", ".join(
            [f"{key}: {subject_test_metrics_dict_avg_std[key][0]:.2f}+-{subject_test_metrics_dict_avg_std[key][1]:.2f}%"
             for key in subject_test_metrics_dict_avg_std.keys()]
        )

    folder_path = (
        "./results/"
        + args.method
        + "/"
        + args.task_name
        + "/"
        + args.model
        + "/"
        + args.model_id
        + "/"
    )
    file_name = "results.txt"
    file_path = os.path.join(folder_path, file_name)
    # Write results to file
    with open(file_path, 'a') as f:
        if args.is_training == 1:
            if 'pretrain' in args.task_name:
                f.write(f"Pretraining Datasets: {args.pretraining_datasets}\n")
            f.write(f"Training Datasets: {args.training_datasets}\n")
        f.write(f"Test Datasets: {args.testing_datasets}\n")
        f.write(f"Model_id: {args.model_id}; Model: {args.model}; Total Params: {total_params}\n")
        f.write('Average and std of validation and testing results over {} runs\n'.format(args.itr))
        f.write('Sample-level results: \n')
        f.write(sample_val_results + "\n")
        f.write(sample_test_results + "\n")
        if args.use_subject_vote:
            f.write('Subject-level results after majority voting: \n')
            f.write(subject_val_results + "\n")
            f.write(subject_test_results + "\n")
        f.write("\n\n\n")

    print('Sample-level results: \n')
    print(sample_val_results)
    print(sample_test_results)
    if args.use_subject_vote:
        print('Subject-level results after majority voting: \n')
        print(subject_val_results)
        print(subject_test_results)


def get_metrics_string(val_metrics_dict, test_metrics_dict):
    metrics_string = (
        f"Validation results --- "
        f"Accuracy: {val_metrics_dict['Accuracy']:.5f}, "
        f"Precision: {val_metrics_dict['Precision']:.5f}, "
        f"Recall: {val_metrics_dict['Recall']:.5f}, "
        f"Specificity: {val_metrics_dict['Specificity']:.5f}, "
        f"F1: {val_metrics_dict['F1']:.5f}, "
        f"AUROC: {val_metrics_dict['AUROC']:.5f}, "
        f"AUPRC: {val_metrics_dict['AUPRC']:.5f}\n"
        f"Test results --- "
        f"Accuracy: {test_metrics_dict['Accuracy']:.5f}, "
        f"Precision: {test_metrics_dict['Precision']:.5f}, "
        f"Recall: {test_metrics_dict['Recall']:.5f} "
        f"Specificity: {test_metrics_dict['Specificity']:.5f}, "
        f"F1: {test_metrics_dict['F1']:.5f}, "
        f"AUROC: {test_metrics_dict['AUROC']:.5f}, "
        f"AUPRC: {test_metrics_dict['AUPRC']:.5f}\n")

    return metrics_string


def get_eeg_coords_from_montage(channel_names, montage_name="standard_1005"):
    """
    Given a list of channel names, return their 3D coordinates from the montage.

    channel_names: list[str]  e.g. ["Fp1", "Fp2", "C3", "C4"]
    montage_name: str, default = "standard_1005"
    return: coords (C, 3) in float32
    """

    # Load montage
    montage = mne.channels.make_standard_montage(montage_name)

    # Get mapping: name -> xyz coordinates
    pos_dict = montage.get_positions()["ch_pos"]

    coords = np.zeros((len(channel_names), 3), dtype=np.float32)

    for i, ch_name in enumerate(channel_names):
        if ch_name not in pos_dict:
            raise ValueError(f"Channel '{ch_name}' not found in montage '{montage_name}'.")
        coords[i] = pos_dict[ch_name]  # (3,)

    return coords

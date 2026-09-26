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


CONFUSION_MATRIX_KEY = "_confusion_matrix"


def row_normalized_confusion_matrix(y_true, y_pred, num_classes):
    """Return a fixed-size row-normalized confusion matrix.

    Rows are true classes and columns are predicted classes. Missing true-class
    rows are left as zeros so every MCCV run has the same matrix shape.
    """
    cm = confusion_matrix(
        y_true, y_pred, labels=np.arange(int(num_classes))
    ).astype(np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)


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
        self.val_loss_min = np.inf
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


def calculate_subject_level_metrics(predictions, true_labels, subject_ids, num_classes):
    unique_subjects = np.unique(subject_ids)

    subject_predictions = []
    subject_scores = []
    subject_trues = []

    for subject in unique_subjects:
        indices = np.where(subject_ids == subject)[0]
        subject_preds = predictions[indices].astype(int)
        subject_true = true_labels[indices][0]

        # Fraction of samples predicted as each class
        class_counts = np.bincount(subject_preds, minlength=num_classes)
        class_scores = class_counts / len(subject_preds)

        # Hard subject prediction for Accuracy/F1/etc.
        majority_label = np.argmax(class_scores)

        subject_predictions.append(majority_label)
        subject_scores.append(class_scores)
        subject_trues.append(subject_true)

    subject_predictions = np.asarray(subject_predictions)
    subject_scores = np.asarray(subject_scores)
    subject_trues = np.asarray(subject_trues)

    metrics = {"Accuracy": accuracy_score(subject_trues, subject_predictions)}

    unique_labels = np.unique(subject_trues)
    if len(unique_labels) < 2:
        metrics.update({"Precision": -1, "Recall": -1, "Specificity": -1, "F1": -1, "AUROC": -1, "AUPRC": -1})
    else:
        metrics["Precision"] = precision_score(subject_trues, subject_predictions, average="macro")
        metrics["Recall"] = recall_score(subject_trues, subject_predictions, average="macro")
        metrics["Specificity"] = multiclass_specificity(subject_trues, subject_predictions)
        metrics["F1"] = f1_score(subject_trues, subject_predictions, average="macro")

        if num_classes == 2:
            metrics["AUROC"] = roc_auc_score(subject_trues, subject_scores[:, 1])
            metrics["AUPRC"] = average_precision_score(subject_trues, subject_scores[:, 1])
        else:
            subject_true_onehot = label_binarize(subject_trues, classes=list(range(num_classes)))
            metrics["AUROC"] = roc_auc_score(subject_true_onehot, subject_scores, multi_class="ovr", average="macro")
            metrics["AUPRC"] = average_precision_score(subject_true_onehot, subject_scores, average="macro")

    metrics[CONFUSION_MATRIX_KEY] = row_normalized_confusion_matrix(subject_trues, subject_predictions, num_classes)
    return metrics


def _mean_std_confusion_matrices(metrics_dict_list):
    matrices = np.stack(
        [np.asarray(metrics[CONFUSION_MATRIX_KEY], dtype=np.float64) for metrics in metrics_dict_list],
        axis=0,
    )
    return matrices.mean(axis=0) * 100.0, matrices.std(axis=0) * 100.0


def _format_confusion_matrix(title, mean_matrix, std_matrix):
    lines = [title]
    for row_mean, row_std in zip(mean_matrix, std_matrix):
        lines.append(
            "  " + "  ".join(
                f"{mean:.2f}±{std:.2f}" for mean, std in zip(row_mean, row_std)
            )
        )
    return "\n".join(lines)


def compute_avg_std(args, sample_val_metrics_dict_list, subject_val_metrics_dict_list,
                    sample_test_metrics_dict_list, subject_test_metrics_dict_list, total_params):
    print('>>>>>>>average testing<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

    # Scalar sample-level metrics. Confusion matrices are aggregated separately below.
    metric_keys = [
        key for key in sample_val_metrics_dict_list[0].keys()
        if key != CONFUSION_MATRIX_KEY
    ]
    sample_val_metrics_dict_avg_std = {}
    sample_test_metrics_dict_avg_std = {}
    for key in metric_keys:
        sample_val_avg = np.mean([val_metrics_dict[key] for val_metrics_dict in sample_val_metrics_dict_list]) * 100
        sample_val_std = np.std([val_metrics_dict[key] for val_metrics_dict in sample_val_metrics_dict_list]) * 100
        sample_test_avg = np.mean([test_metrics_dict[key] for test_metrics_dict in sample_test_metrics_dict_list]) * 100
        sample_test_std = np.std([test_metrics_dict[key] for test_metrics_dict in sample_test_metrics_dict_list]) * 100

        sample_val_metrics_dict_avg_std[key] = (sample_val_avg, sample_val_std)
        sample_test_metrics_dict_avg_std[key] = (sample_test_avg, sample_test_std)

    sample_val_results = "Validation results --- " + ", ".join(
        [f"{key}: {sample_val_metrics_dict_avg_std[key][0]:.2f}+-{sample_val_metrics_dict_avg_std[key][1]:.2f}%"
         for key in sample_val_metrics_dict_avg_std.keys()]
    )
    sample_test_results = "Test results --- " + ", ".join(
        [f"{key}: {sample_test_metrics_dict_avg_std[key][0]:.2f}+-{sample_test_metrics_dict_avg_std[key][1]:.2f}%"
         for key in sample_test_metrics_dict_avg_std.keys()]
    )

    subject_val_results = None
    subject_test_results = None
    if args.use_subject_vote:
        subject_metric_keys = [
            key for key in subject_val_metrics_dict_list[0].keys()
            if key != CONFUSION_MATRIX_KEY
        ]
        subject_val_metrics_dict_avg_std = {}
        subject_test_metrics_dict_avg_std = {}
        for key in subject_metric_keys:
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

    # Each run is row-normalized first; then compute the element-wise mean/std
    # across runs. Values are converted to percentages only for presentation.
    sample_val_cm_mean, sample_val_cm_std = _mean_std_confusion_matrices(
        sample_val_metrics_dict_list
    )
    sample_test_cm_mean, sample_test_cm_std = _mean_std_confusion_matrices(
        sample_test_metrics_dict_list
    )

    confusion_blocks = [
        _format_confusion_matrix(
            "Sample-level Confusion Matrix (Val) [mean±std, %]",
            sample_val_cm_mean,
            sample_val_cm_std,
        ),
        "",
        _format_confusion_matrix(
            "Sample-level Confusion Matrix (Test) [mean±std, %]",
            sample_test_cm_mean,
            sample_test_cm_std,
        ),
    ]

    if args.use_subject_vote:
        subject_val_cm_mean, subject_val_cm_std = _mean_std_confusion_matrices(
            subject_val_metrics_dict_list
        )
        subject_test_cm_mean, subject_test_cm_std = _mean_std_confusion_matrices(
            subject_test_metrics_dict_list
        )
        confusion_blocks.extend([
            "",
            _format_confusion_matrix(
                "Subject-level Confusion Matrix (Val) [mean±std, %]",
                subject_val_cm_mean,
                subject_val_cm_std,
            ),
            "",
            _format_confusion_matrix(
                "Subject-level Confusion Matrix (Test) [mean±std, %]",
                subject_test_cm_mean,
                subject_test_cm_std,
            ),
        ])

    confusion_results = "\n".join(confusion_blocks)

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
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'a') as f:
        if args.is_training == 1 and 'pretrain' in args.task_name:
            f.write(f"Pretraining Datasets: {args.pretraining_datasets}\n")
        f.write(f"Downstream Dataset: {args.training_dataset}\n")
        f.write(f"Model_id: {args.model_id}; Model: {args.model}; Total Params: {total_params}\n")
        f.write('Average and std of validation and testing results over {} runs\n'.format(args.itr))
        f.write('Sample-level results: \n')
        f.write(sample_val_results + "\n")
        f.write(sample_test_results + "\n")
        if args.use_subject_vote:
            f.write('Subject-level results after majority voting: \n')
            f.write(subject_val_results + "\n")
            f.write(subject_test_results + "\n")
        f.write("\nConfusion Matrices (row-normalized; element-wise mean±std across runs; in %):\n")
        f.write(confusion_results + "\n")
        f.write("\n\n\n")

    print('Sample-level results: \n')
    print(sample_val_results)
    print(sample_test_results)
    if args.use_subject_vote:
        print('Subject-level results after majority voting: \n')
        print(subject_val_results)
        print(subject_test_results)

    print('\nConfusion Matrices (row-normalized; element-wise mean±std across runs; in %):')
    print(confusion_results)


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
    """Return MNE montage coordinates for channel names with clear validation."""
    if not montage_name:
        raise ValueError("MONTAGE is missing from meta.json.")
    if not channel_names:
        raise ValueError("CHANNELS is missing or empty in meta.json.")

    try:
        montage = mne.channels.make_standard_montage(montage_name)
    except Exception as exc:
        raise ValueError(
            f"Unknown or unsupported MNE montage '{montage_name}'."
        ) from exc

    pos_dict = montage.get_positions()["ch_pos"]
    name_map = {name.lower(): name for name in pos_dict.keys()}

    missing = [name for name in channel_names if name.lower() not in name_map]
    if missing:
        raise ValueError(
            f"Channels {missing} were not found in montage '{montage_name}'. "
            f"Check meta.json CHANNELS/MONTAGE spelling."
        )

    coords = np.zeros((len(channel_names), 3), dtype=np.float32)
    for i, ch_name in enumerate(channel_names):
        coords[i] = pos_dict[name_map[ch_name.lower()]]
    return coords

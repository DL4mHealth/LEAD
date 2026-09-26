import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import optim

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.losses import subject_ce_loss
from utils.tools import (
    CONFUSION_MATRIX_KEY, EarlyStopping, calculate_subject_level_metrics,
    get_metrics_string, multiclass_specificity, row_normalized_confusion_matrix,
)


class Exp_Supervised(Exp_Basic):
    """Supervised downstream training on exactly one dataset."""

    def __init__(self, args):
        super().__init__(args)
        self.swa = getattr(args, "swa", False)
        self.swa_model = optim.swa_utils.AveragedModel(self.model)

    @staticmethod
    def _metadata_sampling_map(*datasets):
        rates = set()
        for dataset in datasets:
            for values in getattr(dataset, "sampling_rates_by_id", {}).values():
                rates.update(int(v) for v in values)
        if not rates:
            raise ValueError("Could not derive sampling rates from dataset metadata/labels.")
        return {rate: idx for idx, rate in enumerate(sorted(rates))}

    def _build_model(self):
        train_data, _ = self._get_data(flag="TRAIN")
        self.args.seq_len = int(train_data.seq_len)
        self.args.max_seq_len = self.args.seq_len
        self.args.pred_len = 0
        self.args.enc_in = int(train_data.enc_in)
        self.args.max_num_channels = self.args.enc_in
        self.args.num_class = int(train_data.num_class)
        if getattr(self.args, "use_sampling_embedding", False):
            self.args.sampling_rate_to_id = self._metadata_sampling_map(train_data)
        else:
            self.args.sampling_rate_to_id = {}

        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        random.seed(self.args.seed)
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)

    @staticmethod
    def _select_criterion():
        return nn.CrossEntropyLoss()

    def _set_train_mode(self):
        self.model.train()

    @staticmethod
    def _checkpoint_dir(args, setting):
        return os.path.join(
            "./checkpoints", args.method, args.task_name, args.model, args.model_id, setting
        )

    @staticmethod
    def _result_dir(args):
        return os.path.join("./results", args.method, args.task_name, args.model, args.model_id)

    def _forward_model(self, model, batch_x, label_id):
        """Unified model call shared by LEAD and every baseline."""
        label_id_device = label_id.to(batch_x.device)
        return model(batch_x, label_id=label_id_device)

    @staticmethod
    def _safe_metrics(trues, predictions, probs, num_class):
        metrics = {
            "Accuracy": accuracy_score(trues, predictions),
            CONFUSION_MATRIX_KEY: row_normalized_confusion_matrix(
                trues, predictions, num_class
            ),
        }
        if len(np.unique(trues)) < 2:
            metrics.update({
                "Precision": -1,
                "Recall": -1,
                "Specificity": -1,
                "F1": -1,
                "AUROC": -1,
                "AUPRC": -1,
            })
            return metrics

        metrics["Precision"] = precision_score(trues, predictions, average="macro", zero_division=0)
        metrics["Recall"] = recall_score(trues, predictions, average="macro", zero_division=0)
        metrics["Specificity"] = multiclass_specificity(trues, predictions)
        metrics["F1"] = f1_score(trues, predictions, average="macro", zero_division=0)

        onehot = torch.nn.functional.one_hot(
            torch.as_tensor(trues, dtype=torch.long), num_classes=num_class
        ).numpy()
        try:
            if num_class == 2:
                metrics["AUROC"] = roc_auc_score(trues, probs[:, 1])
                metrics["AUPRC"] = average_precision_score(onehot, probs, average="macro")
            else:
                metrics["AUROC"] = roc_auc_score(onehot, probs, multi_class="ovr")
                metrics["AUPRC"] = average_precision_score(onehot, probs, average="macro")
        except ValueError:
            metrics["AUROC"] = -1
            metrics["AUPRC"] = -1
        return metrics

    @staticmethod
    def _format_metrics(prefix, metrics):
        if metrics is None:
            return f"{prefix}: None"
        return prefix + " --- " + ", ".join(
            f"{k}: {v:.5f}" for k, v in metrics.items() if k != CONFUSION_MATRIX_KEY
        )

    def vali(self, vali_data, vali_loader, criterion):
        eval_model = self.swa_model if self.swa else self.model
        was_training = eval_model.training
        eval_model.eval()

        losses, preds, trues, subject_ids, dataset_ids = [], [], [], [], []
        with torch.no_grad():
            for batch_x, label_id in vali_loader:
                batch_x = batch_x.float().to(self.device)
                label = label_id[:, 0].long().to(self.device)
                outputs = self._forward_model(eval_model, batch_x, label_id)
                losses.append(criterion(outputs, label).item())
                preds.append(outputs.detach().cpu())
                trues.append(label.detach().cpu())
                subject_ids.append(label_id[:, 1].detach().cpu())
                dataset_ids.append(label_id[:, 3].detach().cpu())

        eval_model.train(was_training)
        logits = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0).numpy().astype(np.int64)
        subject_ids = torch.cat(subject_ids, dim=0).numpy().astype(np.int64)
        dataset_ids = torch.cat(dataset_ids, dim=0).numpy().astype(np.int64)
        probs = torch.softmax(logits, dim=1).numpy()
        predictions = probs.argmax(axis=1)

        sample_metrics = self._safe_metrics(trues, predictions, probs, self.args.num_class)
        subject_metrics = None
        if self.args.use_subject_vote:
            # Dataset ID is included defensively even though downstream is one dataset.
            vote_ids = dataset_ids.astype(np.int64) * 1_000_000_000 + subject_ids
            subject_metrics = calculate_subject_level_metrics(
                predictions, trues, vote_ids, self.args.num_class
            )
        return float(np.mean(losses)), sample_metrics, subject_metrics

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="TRAIN")
        vali_data, vali_loader = self._get_data(flag="VAL")
        test_data, test_loader = self._get_data(flag="TEST")
        print("Training data summary:\n" + train_data.summary())
        print("\nValidation data summary:\n" + vali_data.summary())
        print("\nTest data summary:\n" + test_data.summary())

        path = self._checkpoint_dir(self.args, setting)
        os.makedirs(path, exist_ok=True)
        optimizer = self._select_optimizer()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.args.train_epochs
        )
        criterion = self._select_criterion()
        early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True, delta=1e-5
        )
        train_steps = len(train_loader)
        self._print_common_run_config()
        print(f"Subject-level majority voting: {bool(getattr(self.args, 'use_subject_vote', False))}")
        print(f"Subject-regularized training loss: {bool(getattr(self.args, 'use_subject_loss', False))}")
        if self.swa:
            print("SWA enabled for supervised/downstream training.")
        else:
            print("SWA enabled for supervised/downstream training: False")

        for epoch in range(self.args.train_epochs):
            self._set_train_mode()
            start = time.time()
            train_losses = []
            for i, (batch_x, label_id) in enumerate(train_loader):
                optimizer.zero_grad()
                batch_x = batch_x.float().to(self.device)
                label = label_id[:, 0].long().to(self.device)
                outputs = self._forward_model(self.model, batch_x, label_id)
                if self.args.use_subject_loss:
                    loss = subject_ce_loss(
                        outputs, label, label_id[:, 1].long().to(self.device)
                    )
                else:
                    loss = criterion(outputs, label)
                train_losses.append(loss.item())
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                optimizer.step()

                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start
                    speed = elapsed / (i + 1)
                    print(
                        f"\tEpoch {epoch + 1}, iter {i + 1}/{train_steps}, "
                        f"loss: {loss.item():.7f}, speed: {speed:.4f}s/iter"
                    )

            if self.swa:
                self.swa_model.update_parameters(self.model)

            val_loss, val_metrics, val_subject = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_metrics, test_subject = self.vali(test_data, test_loader, criterion)
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch: {epoch + 1} cost time: {time.time() - start:.2f}s")
            print(
                f"Epoch: {epoch + 1}, Steps: {train_steps}, | "
                f"Train Loss: {np.mean(train_losses):.5f} | Learning Rate: {current_lr:.5e}\n"
            )
            sample_metrics_string = get_metrics_string(val_metrics, test_metrics)
            print(f"Sample-level results: \n{sample_metrics_string}")
            if self.args.use_subject_vote:
                subject_metrics_string = get_metrics_string(val_subject, test_subject)
                print(
                    f"Subject-level results after majority voting: \n"
                    f"{subject_metrics_string}"
                )

            early_stopping(-val_metrics["F1"], self.swa_model if self.swa else self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            scheduler.step()
            print(f"------------------End of Epoch {epoch + 1}---------------------\n")

        best_path = os.path.join(path, "checkpoint.pth")
        if self.swa:
            self.swa_model.load_state_dict(torch.load(best_path, map_location=self.device))
            return self.swa_model
        self.model.load_state_dict(torch.load(best_path, map_location=self.device))
        return self.model

    def test(self, setting, test=0):
        vali_data, vali_loader = self._get_data(flag="VAL")
        test_data, test_loader = self._get_data(flag="TEST")
        if test:
            model_path = os.path.join(self._checkpoint_dir(self.args, setting), "checkpoint.pth")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"No model found at {model_path}")
            target_model = self.swa_model if self.swa else self.model
            target_model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model from {model_path}")

        criterion = self._select_criterion()
        val_loss, val_metrics, val_subject = self.vali(vali_data, vali_loader, criterion)
        test_loss, test_metrics, test_subject = self.vali(test_data, test_loader, criterion)
        sample_metrics_string = get_metrics_string(val_metrics, test_metrics)
        print(f"Sample-level results: \n{sample_metrics_string}")
        subject_metrics_string = None
        if self.args.use_subject_vote:
            subject_metrics_string = get_metrics_string(val_subject, test_subject)
            print(
                f"Subject-level results after majority voting: \n"
                f"{subject_metrics_string}"
            )

        result_dir = self._result_dir(self.args)
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, "results.txt"), "a") as f:
            f.write(f"Model Setting: {setting}\n")
            f.write(f"Sample-level results: \n{sample_metrics_string}")
            if self.args.use_subject_vote:
                f.write(
                    f"Subject-level results after majority voting: \n"
                    f"{subject_metrics_string}"
                )
            f.write("\n")

        total_params = sum(p.numel() for p in self.model.parameters())
        return val_metrics, val_subject, test_metrics, test_subject, total_params

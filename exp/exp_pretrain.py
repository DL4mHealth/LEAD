import os
import random
import time
from collections import OrderedDict

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
from utils import eval_protocols
from utils.losses import simclr_id_loss
from utils.tools import (
    CONFUSION_MATRIX_KEY, calculate_subject_level_metrics, get_metrics_string,
    multiclass_specificity, row_normalized_confusion_matrix,
)


class Exp_Pretrain(Exp_Basic):
    """Subject-level contrastive pretraining for LEAD.

    PRETRAIN may contain heterogeneous datasets. Datasets with identical input
    geometry/channel metadata may share a batch, while index group shuffling
    remains grouped by (dataset_id, subject_id, sampling_rate). LEAD creates
    two augmented views, applies structured channel/patch subsampling after
    embedding, and treats all trials from the same subject as subject-level
    positives regardless of sampling rate. No SWA or early stopping is used for
    pretraining.
    """

    def __init__(self, args):
        super().__init__(args)
        self.lambda2 = float(getattr(args, "lambda2", 0.75))
        if not (0.0 <= self.lambda2 <= 1.0):
            raise ValueError("lambda2 must be in [0, 1].")

    @staticmethod
    def _merge_dicts(*dicts):
        out = {}
        for d in dicts:
            out.update(dict(d or {}))
        return out

    @staticmethod
    def _sampling_map(*datasets):
        rates = set()
        for dataset in datasets:
            for values in getattr(dataset, "sampling_rates_by_id", {}).values():
                rates.update(int(v) for v in values)
        if not rates:
            raise ValueError("Could not derive sampling rates from dataset metadata/labels.")
        return {rate: idx for idx, rate in enumerate(sorted(rates))}

    def _build_model(self):
        if self.args.model != "LEAD":
            raise ValueError(
                "The new contrastive pretraining pipeline is defined for LEAD. "
                "Baseline models remain available for supervised evaluation."
            )

        pretrain_data, _ = self._get_data(flag="PRETRAIN")
        pretrain_channels = dict(getattr(self.args, "channel_names_by_id", {}))
        pretrain_montage = dict(getattr(self.args, "montage_by_id", {}))
        pretrain_names = dict(getattr(self.args, "dataset_id_to_name", {}))

        # Load the unique downstream TRAIN split before model construction. This
        # preserves the ERP-FM convention that fixed downstream dimensions always
        # come from the downstream training dataset, never PRETRAIN max_T/max_C.
        train_data, _ = self._get_data(flag="TRAIN")
        downstream_channels = dict(getattr(self.args, "channel_names_by_id", {}))
        downstream_montage = dict(getattr(self.args, "montage_by_id", {}))
        downstream_names = dict(getattr(self.args, "dataset_id_to_name", {}))

        self.args.channel_names_by_id = self._merge_dicts(pretrain_channels, downstream_channels)
        self.args.montage_by_id = self._merge_dicts(pretrain_montage, downstream_montage)
        self.args.dataset_id_to_name = self._merge_dicts(pretrain_names, downstream_names)
        if getattr(self.args, "use_sampling_embedding", False):
            self.args.sampling_rate_to_id = self._sampling_map(pretrain_data, train_data)
        else:
            self.args.sampling_rate_to_id = {}

        self.args.seq_len = int(train_data.seq_len)
        self.args.max_seq_len = self.args.seq_len
        self.args.pred_len = 0
        self.args.enc_in = int(train_data.enc_in)
        self.args.max_num_channels = self.args.enc_in
        self.args.num_class = int(train_data.num_class)

        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        init_path = str(getattr(self.args, "pretrain_init_path", "")).strip()
        if init_path:
            self._load_pretrain_initialization(model, init_path)
        return model

    def _get_data(self, flag):
        random.seed(self.args.seed)
        return data_provider(self.args, flag)

    @staticmethod
    def _strip_prefixes(state_dict):
        cleaned = OrderedDict()
        for key, value in state_dict.items():
            if key == "n_averaged":
                continue
            while key.startswith("module."):
                key = key[len("module."):]
            cleaned[key] = value
        return cleaned

    def _load_pretrain_initialization(self, model, init_path):
        if os.path.isdir(init_path):
            init_path = os.path.join(init_path, "checkpoint.pth")
        if not os.path.exists(init_path):
            raise FileNotFoundError(f"No initialization checkpoint found at {init_path}")
        checkpoint = torch.load(init_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        checkpoint = self._strip_prefixes(checkpoint)
        target = model.module if isinstance(model, nn.DataParallel) else model
        target_state = target.state_dict()
        matched = {
            k: v for k, v in checkpoint.items()
            if k in target_state and target_state[k].shape == v.shape
            and not k.startswith("projection_head.")
        }
        missing, unexpected = target.load_state_dict(matched, strict=False)
        print(f"Initialized pretraining from {init_path}: loaded {len(matched)} tensors.")
        print(f"Missing keys: {missing}")
        print(f"Unexpected keys: {unexpected}")

    def _select_optimizer(self):
        return optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)

    @staticmethod
    def _checkpoint_dir(args, setting):
        return os.path.join(
            "./checkpoints", args.method, args.task_name, args.model, args.model_id, setting
        )

    @staticmethod
    def _result_dir(args):
        return os.path.join("./results", args.method, args.task_name, args.model, args.model_id)

    @staticmethod
    def _unwrap_model(model):
        return model.module if isinstance(model, nn.DataParallel) else model

    def _contrastive_loss(self, outputs, label_id):
        z1, z2 = outputs["z1"], outputs["z2"]
        subject_id = label_id[:, 1].long().to(z1.device)
        dataset_id = label_id[:, 3].long().to(z1.device)
        # Subject IDs are dataset-local. Once compatible datasets can share a
        # batch, dataset_id must be part of the identity key to avoid treating
        # equal local IDs from different datasets as the same person. Sampling
        # rate is intentionally excluded, so 50/100/200-Hz trials from the same
        # subject remain positive pairs.
        subject_keys = torch.stack([dataset_id, subject_id], dim=1)
        _, positive_pair_id = torch.unique(subject_keys, dim=0, return_inverse=True)
        z1 = z1.reshape(z1.shape[0], -1)
        z2 = z2.reshape(z2.shape[0], -1)

        return simclr_id_loss(
            z1,
            z2,
            positive_pair_id,
            lambda1=1.0 - self.lambda2,
            lambda2=self.lambda2,
        )

    def encode(self, loader):
        reprs, labels, subject_ids, dataset_ids = [], [], [], []
        base_model = self._unwrap_model(self.model)
        was_training = self.model.training
        self.model.eval()

        with torch.no_grad():
            for batch_x, label_id in loader:
                batch_x = batch_x.float().to(self.device)
                label_device = label_id.to(self.device)
                rep = base_model.encode(batch_x, label_id=label_device)
                reprs.append(rep.detach().cpu().float().numpy())
                labels.append(label_id[:, 0].detach().cpu().numpy())
                subject_ids.append(label_id[:, 1].detach().cpu().numpy())
                dataset_ids.append(label_id[:, 3].detach().cpu().numpy())

        self.model.train(was_training)
        return (
            np.concatenate(reprs, axis=0),
            np.concatenate(labels, axis=0).astype(np.int64),
            np.concatenate(subject_ids, axis=0).astype(np.int64),
            np.concatenate(dataset_ids, axis=0).astype(np.int64),
        )

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
                "Precision": -1, "Recall": -1, "Specificity": -1,
                "F1": -1, "AUROC": -1, "AUPRC": -1,
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
            else:
                metrics["AUROC"] = roc_auc_score(onehot, probs, multi_class="ovr")
            metrics["AUPRC"] = average_precision_score(onehot, probs, average="macro")
        except ValueError:
            metrics["AUROC"] = -1
            metrics["AUPRC"] = -1
        return metrics

    def linear_probe(self, train_loader, eval_loader):
        train_repr, train_labels, _, _ = self.encode(train_loader)
        eval_repr, eval_labels, eval_subject_ids, eval_dataset_ids = self.encode(eval_loader)
        clf = eval_protocols.fit_lr(train_repr, train_labels)
        probs = clf.predict_proba(eval_repr)
        predictions = probs.argmax(axis=1)
        sample_metrics = self._safe_metrics(
            eval_labels, predictions, probs, self.args.num_class
        )
        subject_metrics = None
        if self.args.use_subject_vote:
            vote_ids = eval_dataset_ids * 1_000_000_000 + eval_subject_ids
            subject_metrics = calculate_subject_level_metrics(
                predictions, eval_labels, vote_ids, self.args.num_class
            )
        return sample_metrics, subject_metrics

    @staticmethod
    def _format_metrics(prefix, metrics):
        if metrics is None:
            return f"{prefix}: None"
        return prefix + " --- " + ", ".join(
            f"{k}: {v:.5f}" for k, v in metrics.items() if k != CONFUSION_MATRIX_KEY
        )

    def train(self, setting):
        pretrain_data, pretrain_loader = self._get_data(flag="PRETRAIN")
        use_linear_probe = bool(getattr(self.args, "pretrain_linear_probe", False))

        train_loader = vali_loader = test_loader = None
        print("Pretraining data summary:\n" + pretrain_data.summary())

        if use_linear_probe:
            train_data, train_loader = self._get_data(flag="TRAIN")
            vali_data, vali_loader = self._get_data(flag="VAL")
            test_data, test_loader = self._get_data(flag="TEST")
            print("\nDownstream training data summary:\n" + train_data.summary())
            print("\nDownstream validation data summary:\n" + vali_data.summary())
            print("\nDownstream test data summary:\n" + test_data.summary())

        path = self._checkpoint_dir(self.args, setting)
        os.makedirs(path, exist_ok=True)
        optimizer = self._select_optimizer()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.args.train_epochs
        )
        train_steps = len(pretrain_loader)
        self._print_common_run_config()
        print(f"Subject contrastive weight (lambda2): {self.lambda2}")
        print(f"Structured contrastive token ratio: {self.args.contrastive_token_ratio}")
        print("Linear-probe representation h: last-token-per-channel -> GELU -> Dropout -> channel mean pooling.")
        print("Contrastive representation z: h -> shared projection head (D -> 2D -> D).")
        if getattr(self.args, "group_shuffle", False):
            print("Grouped pretraining sampler: enabled")
            print(f"Subject + sampling-rate group size: {self.args.group_size}")
        else:
            print("Grouped pretraining sampler: disabled (group_size inactive; input-compatible dataset mixing enabled)")
        if use_linear_probe:
            print("Pretraining linear probe: enabled")
        else:
            print("Pretraining linear probe: disabled (use --pretrain_linear_probe to enable)")
        print("Pretraining uses no SWA and no early stopping.")

        for epoch in range(self.args.train_epochs):
            if hasattr(pretrain_loader, "batch_sampler") and hasattr(pretrain_loader.batch_sampler, "set_epoch"):
                pretrain_loader.batch_sampler.set_epoch(epoch)

            self.model.train()
            start = time.time()
            losses = []
            for i, (batch_x, label_id) in enumerate(pretrain_loader):
                optimizer.zero_grad()
                batch_x = batch_x.float().to(self.device)
                label_device = label_id.to(self.device)
                outputs = self.model(batch_x, label_id=label_device)
                loss = self._contrastive_loss(outputs, label_id)
                losses.append(loss.item())
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                optimizer.step()

                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start
                    speed = elapsed / (i + 1)
                    print(
                        f"\tEpoch {epoch + 1}, iter {i + 1}/{train_steps}, "
                        f"contrastive loss: {loss.item():.7f}, speed: {speed:.4f}s/iter"
                    )

            scheduler.step()
            print(
                f"Epoch: {epoch + 1}, cost time: {time.time() - start:.2f}s, "
                f"Steps: {train_steps}, Loss: {np.mean(losses):.6f}, "
                f"LR: {scheduler.get_last_lr()[0]:.5e}"
            )

            if use_linear_probe:
                print("Linear probing on downstream dataset...")
                sample_val, subject_val = self.linear_probe(train_loader, vali_loader)
                sample_test, subject_test = self.linear_probe(train_loader, test_loader)
                sample_metrics_string = get_metrics_string(sample_val, sample_test)
                print(f"Sample-level results: \n{sample_metrics_string}")
                if self.args.use_subject_vote:
                    subject_metrics_string = get_metrics_string(subject_val, subject_test)
                    print(
                        f"Subject-level results after majority voting: \n"
                        f"{subject_metrics_string}"
                    )

            ckpt = os.path.join(path, "checkpoint.pth")
            torch.save(self.model.state_dict(), ckpt)
            print(f"Saved pretraining checkpoint to {ckpt}")
            print(f"------------------End of Epoch {epoch + 1}---------------------\n")

        return self.model

    def test(self, setting, test=0):
        if not bool(getattr(self.args, "pretrain_linear_probe", False)):
            raise RuntimeError(
                "Pretraining linear probe is disabled. Add --pretrain_linear_probe to enable it."
            )

        train_data, train_loader = self._get_data(flag="TRAIN")
        vali_data, vali_loader = self._get_data(flag="VAL")
        test_data, test_loader = self._get_data(flag="TEST")
        if test:
            ckpt = os.path.join(self._checkpoint_dir(self.args, setting), "checkpoint.pth")
            if not os.path.exists(ckpt):
                raise FileNotFoundError(f"No model found at {ckpt}")
            self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
            print(f"Loaded pretraining model from {ckpt}")

        sample_val, subject_val = self.linear_probe(train_loader, vali_loader)
        sample_test, subject_test = self.linear_probe(train_loader, test_loader)
        sample_metrics_string = get_metrics_string(sample_val, sample_test)
        print(f"Sample-level results: \n{sample_metrics_string}")
        subject_metrics_string = None
        if self.args.use_subject_vote:
            subject_metrics_string = get_metrics_string(subject_val, subject_test)
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
        return sample_val, subject_val, sample_test, subject_test, total_params

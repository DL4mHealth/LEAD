from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os
import time
import warnings
import numpy as np
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from utils.tools import multiclass_specificity
from utils import eval_protocols
from utils.losses import simclr_id_loss, simclr_loss, id_loss

from utils.tools import calculate_subject_level_metrics, get_metrics_string

warnings.filterwarnings("ignore")


class Exp_Pretrain_LEAD(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

        self.swa_model = optim.swa_utils.AveragedModel(self.model)
        self.swa = args.swa
        self.lambda2 = args.lambda2

    def _build_model(self):
        # model input depends on data
        train_data, train_loader = self._get_data(flag='TRAIN')
        # test_data, test_loader = self._get_data(flag="TEST")
        self.args.seq_len = train_data.max_seq_len  # redefine seq_len
        self.args.pred_len = 0
        self.args.enc_in = int(train_data.enc_in)
        self.args.num_class = int(train_data.num_class)
        # model init
        model = (
            self.model_dict[self.args.model].Model(self.args).float()
        )  # pass args to model
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        random.seed(self.args.seed)
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = simclr_id_loss
        return criterion

    def encode(self, loader):
        labels_list = []
        ids_list = []
        reprs_list = []
        dataset_ids_list = []
        org_training = self.model.training
        if self.swa:
            self.swa_model.eval()
        else:
            self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label_id, padding_mask) in enumerate(loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label_id[:, 0]
                sub_id = label_id[:, 1]
                fs = label_id[:, 2]
                dataset_id = label_id[:, 3]
                label = label.to(self.device)
                sub_id = sub_id.to(self.device)
                dataset_id = dataset_id.to(self.device)

                if self.swa:
                    reprs_h, reprs_z = self.swa_model(batch_x, padding_mask, None, None, fs, None)
                else:
                    reprs_h, reprs_z = self.model(batch_x, padding_mask, None, None, fs, None)
                # ---- important: immediately move to cpu() to avoid GPU memory leak ----
                reprs_h = reprs_h.reshape(reprs_h.shape[0], -1).detach().cpu().float().numpy()
                reprs_list.append(reprs_h)
                labels_list.append(label.detach().cpu().numpy())
                ids_list.append(sub_id.detach().cpu().numpy())
                dataset_ids_list.append(dataset_id.detach().cpu().numpy())

            reprs_array = np.concatenate(reprs_list, axis=0)
            labels = np.concatenate(labels_list, axis=0)
            sub_ids = np.concatenate(ids_list, axis=0)
            dataset_ids = np.concatenate(dataset_ids_list, axis=0)

        if self.swa:
            self.swa_model.train(org_training)
        else:
            self.model.train(org_training)
        return reprs_array, labels, sub_ids, dataset_ids

    def vali(self, train_loader, vali_loader):
        train_reprs, train_labels, train_ids, train_dataset_ids = self.encode(train_loader)
        vali_reprs, vali_labels, vali_ids, vali_dataset_ids = self.encode(vali_loader)

        fit_clf = eval_protocols.fit_lr
        clf = fit_clf(train_reprs, train_labels)

        probs = clf.predict_proba(vali_reprs)
        trues_onehot = (F.one_hot(torch.tensor(vali_labels).long(), num_classes=int(train_labels.max() + 1))).numpy()
        predictions = probs.argmax(axis=1)
        trues = vali_labels

        dataset_sample_results_list = []
        dataset_subject_results_list = []
        # calculate weighted metrics among datasets
        for dataset_id in np.unique(vali_dataset_ids):
            mask = vali_dataset_ids == dataset_id  # mask for samples from the same dataset

            sample_metrics_dict = {
                "Accuracy": accuracy_score(trues[mask], predictions[mask]),
                "Precision": precision_score(trues[mask], predictions[mask], average="macro"),
                "Recall": recall_score(trues[mask], predictions[mask], average="macro"),
                "Specificity": multiclass_specificity(trues[mask], predictions[mask]),
                "F1": f1_score(trues[mask], predictions[mask], average="macro"),
                "AUROC": roc_auc_score(trues_onehot[mask], probs[mask], multi_class="ovr"),
                "AUPRC": average_precision_score(trues_onehot[mask], probs[mask], average="macro"),
            }  # sample-level performance metrics
            dataset_sample_results_list.append(sample_metrics_dict)

            if self.args.use_subject_vote:
                subject_metrics_dict = calculate_subject_level_metrics(
                    predictions[mask], trues[mask], vali_ids[mask], self.args.num_class
                )  # subject-level performance metrics, do voting for each subject
            else:
                subject_metrics_dict = None
            dataset_subject_results_list.append(subject_metrics_dict)

        def get_average_metrics(metrics_list):
            average_metrics = {key: 0 for key in metrics_list[0]}
            for metrics in metrics_list:
                for key, value in metrics.items():
                    average_metrics[key] += value
            for key in average_metrics:
                average_metrics[key] /= len(metrics_list)
            return average_metrics

        average_sample_metrics = get_average_metrics(dataset_sample_results_list)
        if self.args.use_subject_vote:
            average_subject_metrics = get_average_metrics(dataset_subject_results_list)
        else:
            average_subject_metrics = None

        if self.swa:
            self.swa_model.train()
        else:
            self.model.train()
        return average_sample_metrics, average_subject_metrics

    def train(self, setting):
        pretrain_data, pretrain_loader = self._get_data(flag="PRETRAIN")
        train_data, train_loader = self._get_data(flag="TRAIN")
        vali_data, vali_loader = self._get_data(flag="VAL")
        test_data, test_loader = self._get_data(flag="TEST")
        print(
            f"Pretraining samples: {len(pretrain_data)}, "
            f"seq_len (max): {getattr(pretrain_data, 'max_seq_len', 'NA')}, "
            f"enc_in: {getattr(pretrain_data, 'enc_in', 'NA')},")
        print(
            f"Training samples: {len(train_data)}, "
            f"seq_len (max): {getattr(train_data, 'max_seq_len', 'NA')}, "
            f"enc_in: {getattr(train_data, 'enc_in', 'NA')}, "
            f"num_class: {getattr(train_data, 'num_class', 'NA')}")
        print(f"Validation samples: {len(vali_data)}")
        print(f"Test samples: {len(test_data)}")

        path = (
            "./checkpoints/"
            + self.args.method
            + "/"
            + self.args.task_name
            + "/"
            + self.args.model
            + "/"
            + self.args.model_id
            + "/"
            + setting
            + "/"
        )
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        """early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True, delta=1e-5
        )"""  # no early stopping for pretraining

        model_optim = self._select_optimizer()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.train_epochs)
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label_id, padding_mask) in enumerate(pretrain_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                sub_id = label_id[:, 1]   # first column is the label, second column is subject id
                fs = label_id[:, 2]
                dataset_id = label_id[:, 3]
                sub_id = sub_id.to(self.device)

                _, reprs_z_1 = self.model(batch_x, padding_mask, None, None, fs, None)
                _, reprs_z_2 = self.model(batch_x, padding_mask, None, None, fs, None)
                # cross contrast, use different losses for ablation study
                loss = criterion(reprs_z_1, reprs_z_2, sub_id, lambda1=(1 - self.lambda2), lambda2=self.lambda2)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            self.swa_model.update_parameters(self.model)

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print("Linear Probe for contrastive pretraining on downstream training, validation and test sets...")
                sample_val_metrics_dict, subject_val_metrics_dict = self.vali(train_loader, vali_loader)
                sample_test_metrics_dict, subject_test_metrics_dict = self.vali(train_loader, test_loader)

                current_lr = scheduler.get_last_lr()[0]
                sample_metrics_string = get_metrics_string(sample_val_metrics_dict, sample_test_metrics_dict)
                print(f"Epoch: {epoch + 1}, "
                      f"Steps: {train_steps}, | Train Loss: {train_loss:.5f} | Learning Rate: {current_lr:.5e}\n")
                print(f"Sample-level results: \n{sample_metrics_string}")
                if self.args.use_subject_vote:  # subject-level results
                    subject_metrics_string = get_metrics_string(subject_val_metrics_dict, subject_test_metrics_dict)
                    print(f"Subject-level results after majority voting: \n{subject_metrics_string}")
            """early_stopping(
                train_loss,
                self.swa_model if self.swa else self.model,
                path,
            )
            if early_stopping.early_stop:
                print("Early stopping")
                break"""  # no early stopping for pretraining
            print("Saving model...\n")
            try:
                if self.swa:
                    torch.save(self.swa_model.state_dict(), path + '/' + "checkpoint.pth")
                else:
                    torch.save(self.model.state_dict(), path + '/' + 'checkpoint.pth')
            except Exception as e:
                print(f"Error saving model: {e}")
            scheduler.step()
            print(f"------------------End of Epoch {epoch + 1}---------------------\n")

        model_path = path + "checkpoint.pth"
        print("Saving model...")
        if self.swa:
            self.swa_model.load_state_dict(torch.load(model_path))
        else:
            self.model.load_state_dict(torch.load(model_path))

        return self.model

    def test(self, setting, test=0):
        train_data, train_loader = self._get_data(flag="TRAIN")
        vali_data, vali_loader = self._get_data(flag="VAL")
        test_data, test_loader = self._get_data(flag="TEST")
        if test:
            print("loading model")
            path = (
                "./checkpoints/"
                + self.args.method
                + "/"
                + self.args.task_name
                + "/"
                + self.args.model
                + "/"
                + self.args.model_id
                + "/"
                + setting
                + "/"
            )
            model_path = path + "checkpoint.pth"
            if not os.path.exists(model_path):
                raise Exception("No model found at %s" % model_path)
            if self.swa:
                self.swa_model.load_state_dict(torch.load(model_path))
            else:
                self.model.load_state_dict(torch.load(model_path))
            print(f"Loading model successful at {model_path}, start to test")

        total_params = sum(p.numel() for p in self.model.parameters())
        criterion = self._select_criterion()
        print("Linear Probe for contrastive pretraining model on downstream training, validation and test sets...")
        sample_val_metrics_dict, subject_val_metrics_dict = self.vali(train_loader, vali_loader)
        sample_test_metrics_dict, subject_test_metrics_dict = self.vali(train_loader, test_loader)

        # result save
        folder_path = (
            "./results/"
            + self.args.method
            + "/"
            + self.args.task_name
            + "/"
            + self.args.model
            + "/"
            + self.args.model_id
            + "/"
        )
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        sample_metrics_string = get_metrics_string(sample_val_metrics_dict, sample_test_metrics_dict)
        print(f"Sample-level results: \n{sample_metrics_string}")
        if self.args.use_subject_vote:  # subject-level results
            subject_metrics_string = get_metrics_string(subject_val_metrics_dict, subject_test_metrics_dict)
            print(f"Subject-level results after majority voting: \n{subject_metrics_string}")
        file_name = "results.txt"
        file_path = os.path.join(folder_path, file_name)
        f = open(file_path, "a")
        f.write("Model Setting: " + setting + "  \n")
        f.write(f"Sample-level results: \n{sample_metrics_string}")
        if self.args.use_subject_vote:
            f.write(f"Subject-level results after majority voting: \n{subject_metrics_string}")
        f.write("\n")
        f.close()

        return (sample_val_metrics_dict, subject_val_metrics_dict,
                sample_test_metrics_dict, subject_test_metrics_dict,
                total_params)

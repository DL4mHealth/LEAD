from copy import deepcopy
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping
import torch
import torch.nn as nn
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
from typing import Dict, Tuple

from utils.tools import calculate_subject_level_metrics, get_metrics_string
from utils.losses import subject_ce_loss

warnings.filterwarnings("ignore")


class Exp_Supervised(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

        self.swa_model = optim.swa_utils.AveragedModel(self.model)
        self.swa = args.swa

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
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        ids = []
        dataset_ids = []
        if self.swa:
            self.swa_model.eval()
        else:
            self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label_id, padding_mask) in enumerate(vali_loader):
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
                    outputs = self.swa_model(batch_x, padding_mask, None, None, fs, None)
                else:
                    outputs = self.model(batch_x, padding_mask, None, None, fs, None)

                pred = outputs.detach().cpu()
                loss = criterion(pred, label.long().cpu())
                total_loss.append(loss)

                preds.append(outputs.detach())
                trues.append(label)
                ids.append(sub_id)
                dataset_ids.append(dataset_id)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        ids = torch.cat(ids, 0)
        dataset_ids = torch.cat(dataset_ids, 0)
        probs = torch.nn.functional.softmax(
            preds
        )  # (total_samples, num_classes) est. prob. for each class and sample
        trues_onehot = (
            torch.nn.functional.one_hot(
                trues.reshape(
                    -1,
                ).to(torch.long),
                num_classes=self.args.num_class,
            )
            .float()
            .cpu()
            .numpy()
        )
        # print(trues_onehot.shape)
        predictions = (
            torch.argmax(probs, dim=1).cpu().numpy()
        )  # (total_samples,) int class index for each sample
        probs = probs.cpu().numpy()
        trues = trues.flatten().cpu().numpy()
        ids = ids.flatten().cpu().numpy()
        dataset_ids = dataset_ids.flatten().cpu().numpy()

        dataset_sample_results_list = []
        dataset_subject_results_list = []
        # calculate weighted metrics among datasets
        for dataset_id in np.unique(dataset_ids):
            mask = dataset_ids == dataset_id  # mask for samples from the same dataset

            # sample-level performance metrics
            sample_metrics_dict = {"Accuracy": accuracy_score(trues[mask], predictions[mask])}
            # Check how many unique classes are present in the true labels
            unique_labels = np.unique(trues[mask])
            if len(unique_labels) < 2:
                # If there is only one class(e,g, leave-one-subject-out validation),
                # sample-level precision, recall, F1, AUROC and AUPRC are meaningless
                sample_metrics_dict.update({"Precision": -1, "Recall": -1, "Specificity": -1, "F1": -1, "AUROC": -1, "AUPRC": -1})
            else:
                sample_metrics_dict["Precision"] = precision_score(trues[mask], predictions[mask], average="macro")
                sample_metrics_dict["Recall"] = recall_score(trues[mask], predictions[mask], average="macro")
                sample_metrics_dict["Specificity"] = multiclass_specificity(trues[mask], predictions[mask])
                sample_metrics_dict["F1"] = f1_score(trues[mask], predictions[mask], average="macro")
                sample_metrics_dict["AUROC"] = roc_auc_score(trues_onehot[mask], probs[mask], multi_class="ovr")
                sample_metrics_dict["AUPRC"] = average_precision_score(trues_onehot[mask], probs[mask], average="macro")
            dataset_sample_results_list.append(sample_metrics_dict)

            if self.args.use_subject_vote:
                subject_metrics_dict = calculate_subject_level_metrics(
                    predictions[mask], trues[mask], ids[mask], self.args.num_class
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
        return total_loss, average_sample_metrics, average_subject_metrics

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="TRAIN")
        vali_data, vali_loader = self._get_data(flag="VAL")
        test_data, test_loader = self._get_data(flag="TEST")
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
        early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True, delta=1e-5
        )

        model_optim = self._select_optimizer()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.train_epochs)
        criterion = self._select_criterion()
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {total_params}")


        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label_id, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label_id[:, 0]
                ids = label_id[:, 1]
                fs = label_id[:, 2]
                dataset_id = label_id[:, 3]
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None, fs, None)
                if self.args.use_subject_loss:
                    loss = subject_ce_loss(outputs, label.long(), ids)
                else:
                    loss = criterion(outputs, label.long())
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
            vali_loss, sample_val_metrics_dict, subject_val_metrics_dict = self.vali(vali_data, vali_loader, criterion)
            test_loss, sample_test_metrics_dict, subject_test_metrics_dict = self.vali(test_data, test_loader, criterion)

            current_lr = scheduler.get_last_lr()[0]
            sample_metrics_string = get_metrics_string(sample_val_metrics_dict, sample_test_metrics_dict)
            print(f"Epoch: {epoch + 1}, "
                  f"Steps: {train_steps}, | Train Loss: {train_loss:.5f} | Learning Rate: {current_lr:.5e}\n")
            print(f"Sample-level results: \n{sample_metrics_string}")
            if self.args.use_subject_vote:  # subject-level results
                subject_metrics_string = get_metrics_string(subject_val_metrics_dict, subject_test_metrics_dict)
                print(f"Subject-level results after majority voting: \n{subject_metrics_string}")
            early_stopping(
                -sample_val_metrics_dict["F1"],
                self.swa_model if self.swa else self.model,
                path,
            )
            if early_stopping.early_stop:
                print("Early stopping")
                break
            scheduler.step()
            print(f"------------------End of Epoch {epoch + 1}---------------------\n")

        best_model_path = path + "checkpoint.pth"
        """# for swa model, to best leverage the functionality of stochastic weight average,
        # we take the last model as the final model,
        # which is the model after args.patience epochs (early stop counting) of best model on validation set"""
        if self.swa:
            """try:
                print("Saving the last model to leverage the functionality of stochastic weight average")
                torch.save(self.swa_model.state_dict(), best_model_path)
            except Exception as e:
                print(f"Error saving model: {e}")"""
            self.swa_model.load_state_dict(torch.load(best_model_path))
        # for normal model, we simply take the best model on validation set
        else:
            self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
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
        vali_loss, sample_val_metrics_dict, subject_val_metrics_dict = self.vali(vali_data, vali_loader, criterion)
        test_loss, sample_test_metrics_dict, subject_test_metrics_dict = self.vali(test_data, test_loader, criterion)

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

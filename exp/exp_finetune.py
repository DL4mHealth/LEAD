from collections import OrderedDict
import os

import torch
import torch.nn as nn

from exp.exp_supervised import Exp_Supervised


class Exp_Finetune(Exp_Supervised):
    """Downstream fine-tuning initialized from a LEAD pretraining checkpoint."""

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

    def _load_pretrained_checkpoint(self):
        ckpt_path = self.args.checkpoints_path
        if os.path.isdir(ckpt_path):
            ckpt_path = os.path.join(ckpt_path, "checkpoint.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"No pretraining checkpoint found at {ckpt_path}")

        print(f"Loading pretraining checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        checkpoint = self._strip_prefixes(checkpoint)

        model_to_load = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        model_state = model_to_load.state_dict()

        filtered = OrderedDict()
        skipped = []
        for key, value in checkpoint.items():
            # Projection/classification heads are task-specific and are rebuilt downstream.
            if key.startswith("projection_head.") or key.startswith("classifier."):
                skipped.append(key)
                continue
            if key in model_state and model_state[key].shape == value.shape:
                filtered[key] = value
            else:
                skipped.append(key)

        missing, unexpected = model_to_load.load_state_dict(filtered, strict=False)
        print(f"Loaded {len(filtered)} matched tensors from the pretraining checkpoint.")
        if skipped:
            print(f"Skipped {len(skipped)} task-specific or shape-mismatched tensors.")
        print(f"Missing keys: {missing}")
        print(f"Unexpected keys: {unexpected}")

    def train(self, setting):
        self._load_pretrained_checkpoint()
        return super().train(setting)

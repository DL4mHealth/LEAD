import torch
import torch.nn as nn

from exp.exp_finetune import Exp_Finetune


class Exp_Probe(Exp_Finetune):
    """GPU linear probe: freeze the pretrained backbone and train only classifier."""

    def _freeze_backbone(self):
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("classifier.")

        trainable = [name for name, p in model.named_parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError(
                "Probe mode found no classifier parameters. "
                "The model must create self.classifier for task_name='probe'."
            )
        print("Probe mode: pretrained backbone frozen; classifier trainable.")
        print(f"Trainable tensors: {trainable}")

    def _load_pretrained_checkpoint(self):
        super()._load_pretrained_checkpoint()
        self._freeze_backbone()

    def _select_optimizer(self):
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("Probe mode found no trainable parameters.")
        return torch.optim.AdamW(trainable, lr=self.args.learning_rate)

    def _set_train_mode(self):
        # Keep dropout/norm behavior deterministic in the frozen backbone.
        self.model.eval()
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        model.classifier.train()

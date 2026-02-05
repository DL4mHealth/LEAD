import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Paper link: https://www.sciencedirect.com/science/article/pii/S0893608023007037
    - Input:  x_enc (B, T, C)
    - Task:   supervised classification
    - Removed: age/sex/MMSE embedding, MMSE is one kind of label for dementia detection and should not be used as input
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len     # T
        self.enc_in = configs.enc_in       # C
        self.num_class = configs.num_class

        # ----- Conv backbone (from original MNet_1000) -----
        # Input will be reshaped to (B, 1, C, T)

        self.conv1 = nn.Conv2d(1, 40, kernel_size=(self.enc_in, 4))
        self.act1 = nn.Mish()

        self.conv2 = nn.Conv2d(40, 40, kernel_size=(1, 4))
        self.bn2 = nn.BatchNorm2d(40)
        self.pool2 = nn.MaxPool2d((1, 5))
        self.act2 = nn.Mish()

        # SwapLayer functionality: x = x.transpose(1, 2)

        self.conv3 = nn.Conv2d(1, 50, kernel_size=(8, 12))
        self.bn3 = nn.BatchNorm2d(50)
        self.pool3 = nn.MaxPool2d((3, 3))
        self.act3 = nn.Mish()

        self.conv4 = nn.Conv2d(50, 50, kernel_size=(1, 5))
        self.bn4 = nn.BatchNorm2d(50)
        self.pool4 = nn.MaxPool2d((1, 2))
        self.act4 = nn.Mish()

        # ----- Compute final feature dimension -----
        with torch.no_grad():
            dummy = torch.zeros(1, self.seq_len, self.enc_in)
            feat = self._forward_conv(dummy)
            n_fc_in = feat.shape[1]

        # ----- Classification head -----
        n_fc1, n_fc2, d_ratio1, d_ratio2 = 1024, 256, 0.85, 0.85  # from original MNet_1000
        self.fc = nn.Sequential(
            nn.Linear(n_fc_in, n_fc1),
            nn.Mish(),
            nn.Dropout(d_ratio1),

            nn.Linear(n_fc1, n_fc2),
            nn.Mish(),
            nn.Dropout(d_ratio2),

            nn.Linear(n_fc2, self.num_class),
        )


    def _reshape_input(self, x, n_chs):
        """
        (B, T, C) or (B, C, T) -> (B, 1, C, T)
        This ensures compatibility with original MNet conv structure.
        """
        if x.ndim == 3:
            x = x.unsqueeze(-1)  # (B, T, C, 1)
        if x.shape[2] == n_chs:  # If 3rd dim is channels
            x = x.transpose(1, 2)
        # -> (B, 1, C, T)
        x = x.transpose(1, 3).transpose(2, 3)
        return x

    def _forward_conv(self, x):
        """
        Forward pass through conv backbone only.
        Input:  x (B, T, C)
        Output: flattened feature (B, F)
        """
        B, T, C = x.shape

        # reshape to (B, 1, C, T)
        x = self._reshape_input(x, self.enc_in)

        # conv1
        x = self.act1(self.conv1(x))

        # conv2 + bn + pool
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.act2(x)

        # swap channel/time axis: (B, 40, 1, T2) -> (B, 1, 40, T2)
        x = x.transpose(1, 2)

        # conv3 + bn + pool
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.pool3(x)
        x = self.act3(x)

        if x.shape[-1] <= 10:
            # pad time dimension if too small for conv4
            x = F.pad(x, (0, 10 - x.shape[-1]))

        # conv4 + bn + pool
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.pool4(x)
        x = self.act4(x)

        # flatten
        x = x.reshape(B, -1)
        return x

    def supervised(self, x_enc, x_mark_enc):
        """
        x_enc: (B, T, C)
        x_mark_enc: unused
        """
        feat = self._forward_conv(x_enc)
        logits = self.fc(feat)
        return logits

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, fs=None, mask=None):
        if self.task_name == 'supervised':
            return self.supervised(x_enc, x_mark_enc)
        else:
            raise ValueError("Unsupported task for this simplified MNet model.")

import torch
import torch.nn as nn


class Jitter(nn.Module):
    # apply noise on each element
    def __init__(self, scale=0.1):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        if self.training:
            x += torch.randn_like(x) * self.scale
        return x


class Flip(nn.Module):
    # left-right flip
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        if self.training and torch.rand(1) < self.prob:
            return torch.flip(x, [-1])
        return x


class Shuffle(nn.Module):
    # shuffle channels order
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        if self.training and torch.rand(1) < self.prob:
            B, C, T = x.shape
            perm = torch.randperm(C)
            return x[:, perm, :]
        return x


class TemporalMask(nn.Module):
    # Randomly mask a portion of timestamps across all channels
    def __init__(self, ratio=0.1):
        super().__init__()
        self.ratio = ratio

    def forward(self, x):
        if self.training:
            B, C, T = x.shape
            num_mask = int(T * self.ratio)
            mask_indices = torch.randperm(T)[:num_mask]
            x[:, :, mask_indices] = 0
        return x


class ChannelMask(nn.Module):
    # Randomly mask a portion of channels across all timestamps
    def __init__(self, ratio=0.1):
        super().__init__()
        self.ratio = ratio

    def forward(self, x):
        if self.training:
            B, C, T = x.shape
            num_mask = int(C * self.ratio)
            mask_indices = torch.randperm(C)[:num_mask]
            x[:, mask_indices, :] = 0
        return x


class PatchMask(nn.Module):
    """
    Randomly mask single-channel patches.
    x: (B, C, T)
    mask unit: one (channel, patch_index)
    """

    def __init__(self, patch_len=25, ratio=0.1):
        super().__init__()
        self.patch_len = patch_len
        self.ratio = ratio

    def forward(self, x):
        if self.training:
            B, C, T = x.shape
            num_patches = T // self.patch_len  # number of patches per channel
            total_units = C * num_patches  # total (channel, patch) patches
            num_mask = int(total_units * self.ratio)  # number of patches to mask

            # select random (channel, patch_index)
            idx = torch.randperm(total_units)[:num_mask].to(x.device)
            ch_indices = idx // num_patches
            p_indices = idx % num_patches

            # mask the selected patches
            for ch, p in zip(ch_indices, p_indices):
                start = p * self.patch_len
                end = start + self.patch_len
                x[:, ch, start:end] = 0

        return x


class FrequencyMask(nn.Module):
    def __init__(self, ratio=0.1):
        super().__init__()
        self.ratio = ratio

    def forward(self, x):
        if self.training:
            B, C, T = x.shape
            # Perform rfft
            x_fft = torch.fft.rfft(x, dim=-1)
            # Generate random indices for masking
            mask = torch.rand(x_fft.shape, device=x.device) > self.ratio
            # Apply mask
            x_fft = x_fft * mask
            # Perform inverse rfft
            x = torch.fft.irfft(x_fft, n=T, dim=-1)
        return x


def get_augmentation(augmentation, patch_len=25):
    if augmentation.startswith("jitter"):
        if len(augmentation) == 6:
            return Jitter()
        return Jitter(float(augmentation[6:]))
    elif augmentation.startswith("drop"):
        if len(augmentation) == 4:
            return nn.Dropout(0.1)
        return nn.Dropout(float(augmentation[4:]))
    elif augmentation.startswith("flip"):
        if len(augmentation) == 4:
            return Flip()
        return Flip(float(augmentation[4:]))
    elif augmentation.startswith("shuffle"):
        if len(augmentation) == 7:
            return Shuffle()
        return Shuffle(float(augmentation[7:]))
    elif augmentation.startswith("frequency"):
        if len(augmentation) == 9:
            return FrequencyMask()
        return FrequencyMask(float(augmentation[9:]))
    elif augmentation.startswith("mask"):
        if len(augmentation) == 4:
            return TemporalMask()
        return TemporalMask(float(augmentation[4:]))
    elif augmentation.startswith("channel"):
        if len(augmentation) == 7:
            return ChannelMask()
        return ChannelMask(float(augmentation[7:]))
    elif augmentation.startswith("patch"):
        if len(augmentation) == 5:
            return PatchMask(patch_len)
        return PatchMask(patch_len, float(augmentation[5:]))
    elif augmentation == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown augmentation {augmentation}")

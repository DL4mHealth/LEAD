import torch
import torch.nn as nn
import torch.nn.functional as F


class Inception_Block_V1(nn.Module):  # handle different spatial sizes
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)  # register kernels by ModuleList
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class Inception_Block_V2(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels // 2):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[1, 2 * i + 3], padding=[0, i + 1]))
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[2 * i + 3, 1], padding=[i + 1, 0]))
        kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels + 1):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class CausalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        # Compute the padding size required for causality
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=0,
            dilation=dilation,
            groups=groups
        )

    def forward(self, x):
        # only left-side padding is required
        x = F.pad(x, (self.padding, 0))
        # Perform the convolution
        out = self.conv(x)
        return out


class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = CausalConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = CausalConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None

    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual


class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        self.net = nn.Sequential(*[
            DilatedConvBlock(
                channels[i - 1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2 ** i,
                final=(i == len(channels) - 1)
            )
            for i in range(len(channels))
        ])

    def forward(self, x):
        return self.net(x)


class TemporalSpatialConv(nn.Module):
    """
    Input:  x of shape (B, T, C)
    Output: y of shape (B, 128) after flattening (because of AdaptiveAvgPool2d((1, 4)) and 32 channels)
    Pipeline:
      1) Temporal conv along time (kernel (k_t, 1)) on layout (B, 1, T, C)
      2) Depthwise temporal refinement
      3) Permute to (B, 32, C, T) to do spatial (channel) conv across C
      4) Depthwise separable conv + AdaptiveAvgPool2d((1, 4)) to make fixed spatial size
    """
    def __init__(self, channels: int, dropout: float = 0.5, k_t: int = 7):
        super().__init__()
        self.channels = channels

        # --- Temporal branch on (B, 1, T, C) ---
        # temporal conv: kernel on time axis only (height=T axis), keep channel axis intact
        self.temporal_conv1 = nn.Conv2d(1, 16, kernel_size=(k_t, 1), padding=(k_t // 2, 0), bias=False)   # (B,16,T,C)
        self.bn1 = nn.BatchNorm2d(16, affine=False)

        # depthwise temporal conv (refine along time), keep per-feature maps separated
        self.temporal_conv2 = nn.Conv2d(16, 16, kernel_size=(k_t, 1), padding=(k_t // 2, 0),
                                        bias=False, groups=16)  # (B,16,T,C)
        self.bn2 = nn.BatchNorm2d(16, affine=False)
        self.pool_t = nn.AdaptiveAvgPool2d((1, None))  # pool time -> 1, keep channel dim C unchanged
        self.drop1 = nn.Dropout(dropout)

        # --- Spatial (channel) branch ---
        # After temporal pooling: (B,16,1,C). To convolve across channels, permute to (B,16,C,1)
        # depthwise separable conv across channels (spatial):
        self.spatial_dw = nn.Conv2d(16, 16, kernel_size=(3, 1), padding=(1, 0), bias=False, groups=16)  # (B,16,C,1)
        self.spatial_pw = nn.Conv2d(16, 32, kernel_size=(1, 1), bias=False)                             # (B,32,C,1)
        self.bn3 = nn.BatchNorm2d(32, affine=False)

        # local mixing along channel axis (C) using 1xk conv on that axis (now height=C)
        self.mix_conv_dw = nn.Conv2d(32, 32, kernel_size=(5, 1), padding=(2, 0), bias=False, groups=32)  # (B,32,C,1)
        self.mix_conv_pw = nn.Conv2d(32, 32, kernel_size=(1, 1), bias=False)                             # (B,32,C,1)
        self.bn4 = nn.BatchNorm2d(32, affine=False)

        # Final pooling to fixed (1,4) so flatten -> 32*1*4 = 128
        self.pool_final = nn.AdaptiveAvgPool2d((1, 4))
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.shape

        # --- Temporal conv on (B,1,T,C) ---
        x = x.unsqueeze(1)                               # (B,1,T,C)
        x = F.elu(self.bn1(self.temporal_conv1(x)))      # (B,16,T,C)
        x = F.elu(self.bn2(self.temporal_conv2(x)))      # (B,16,T,C)

        # pool time to 1 -> shape becomes (B,16,1,C)
        x = self.pool_t(x)                               # (B,16,1,C)
        x = self.drop1(x)

        # --- Spatial conv across channels ---
        # Move channels axis to "height" so conv kernel can scan across channel dimension
        x = x.permute(0, 1, 3, 2).contiguous()           # (B,16,C,1)

        # depthwise separable conv on channel axis
        x = F.elu(self.bn3(self.spatial_pw(self.spatial_dw(x))))  # (B,32,C,1)
        x = F.elu(self.bn4(self.mix_conv_pw(self.mix_conv_dw(x))))# (B,32,C,1)

        # Final adaptive pooling to (1,4) over (height=C, width=1) -> output (B,32,1,4)
        x = self.pool_final(x)                           # (B,32,1,4)
        x = self.drop2(x)

        # Flatten to (B, 128) regardless of T or C
        x = x.view(B, -1)                                # (B, 32*1*4) = (B, 128)
        return x


class InceptionBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_sizes,
        bottleneck_channels,
        activation,
        dropout,
    ):
        super().__init__()
        self.activation = activation
        self.use_bottleneck = bottleneck_channels is not None and bottleneck_channels > 0

        # 1×1 bottleneck
        self.bottleneck = (
            nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
            if self.use_bottleneck else nn.Identity()
        )
        branch_in = bottleneck_channels if self.use_bottleneck else in_channels

        # 每个分支输出通道数均等
        branch_out = out_channels // len(kernel_sizes)
        self.branches = nn.ModuleList([
            nn.Conv1d(
                branch_in, branch_out,
                kernel_size=k, stride=1,
                padding=k // 2,  # SAME padding
                bias=False)
            for k in kernel_sizes
        ])

        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):  # x: [B, C, T]
        x = self.bottleneck(x)
        outs = [branch(x) for branch in self.branches]          # [ [B, out_channels // num_kernels, T] for each branch ]
        x = torch.cat(outs, dim=1)                              # [B, out_channels, T]
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.drop(x)
        return x


class SpatialBlock(nn.Module):
    """
    Depthwise-Separable Spatial Block for 1D time series data.
    """
    def __init__(self, in_channels: int, depth_multiplier: int = 2, activation=nn.ReLU(inplace=True)):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels * depth_multiplier,
            kernel_size=(1, 1), groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(
            in_channels * depth_multiplier, in_channels,
            kernel_size=(1, 1), bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.activation = activation

    def forward(self, x):  # x: [B, C, T]  →  [B, C, 1, T]
        x = x.unsqueeze(2)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.activation(x)
        return x.squeeze(2)            # back to [B, C, T]



import torch
import torch.nn.functional as F


def swiglu(x):  # x: (B, 2*d_ff, T)
    x1, x2 = x.chunk(2, dim=1)  # split on channel dim
    return x1 * F.silu(x2)      # SwiGLU = x1 * SiLU(x2)

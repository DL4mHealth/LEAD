import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Activation_Family import swiglu
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from einops import rearrange
import math


"""class TopKRouter(nn.Module):
    '''
    Fully vectorized Top-K router with capacity truncation (Switch-style) and load-balancing loss.

    Inputs:
        h: [B, L, D]

    Outputs:
        topk_idx:  [B, L, K]      -- indices of top-K experts
        topk_vals:[B, L, K]      -- probs of top-K experts
        aux_lb:    scalar         -- load-balancing loss (Switch-style)
        dispatch:  [B, L, E, K]   -- hard assignment (0/1) after capacity truncation
        combine:   [B, L, E, K]   -- same mask weighted by probs
    '''
    def __init__(self, d_model, n_experts, top_k=2, capacity_factor=1.25, drop_tokens=True):
        super().__init__()
        assert n_experts >= 1 and top_k >= 1
        self.w_gate = nn.Linear(d_model, n_experts, bias=False)
        self.n_experts = n_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.drop_tokens = drop_tokens

    def forward(self, h: torch.Tensor):
        # h: [B, L, D]
        B, L, D = h.shape
        E = self.n_experts
        K = min(self.top_k, E)  # safety clip

        # ---- gating ----
        logits = self.w_gate(h)                 # [B, L, E]
        gates = torch.softmax(logits, dim=-1)   # [B, L, E]
        topk_vals, topk_idx = gates.topk(k=K, dim=-1)  # [B, L, K], [B, L, K]

        # ---- capacity per expert ----
        BLK = B * L * K
        cap = math.ceil(self.capacity_factor * BLK / E)

        # init outputs
        dispatch = torch.zeros(B, L, E, K, device=h.device, dtype=h.dtype)
        combine  = torch.zeros(B, L, E, K, device=h.device, dtype=h.dtype)

        # ---- build [BLK, E] sparse-like score table ----
        idx_flat = topk_idx.reshape(BLK)      # [BLK]
        val_flat = topk_vals.reshape(BLK)     # [BLK]
        scores = torch.full((BLK, E), float("-inf"),
                            device=h.device, dtype=val_flat.dtype)  # [BLK, E]
        row_ids = torch.arange(BLK, device=h.device)
        scores[row_ids, idx_flat] = val_flat  # only the chosen routes have finite scores

        if self.drop_tokens and cap < BLK:
            # For each expert (column), select top 'cap' rows among BLK candidates
            k_keep = min(cap, BLK)
            keep_vals, keep_pos = scores.topk(k=k_keep, dim=0, largest=True)  # [k_keep, E]
            valid = keep_vals > float("-inf")                                  # [k_keep, E]

            kept_mask_flat = torch.zeros(BLK, E, dtype=torch.bool, device=h.device)
            e_idx = torch.arange(E, device=h.device).unsqueeze(0).expand_as(keep_pos)  # [k_keep, E]
            kept_mask_flat[keep_pos[valid], e_idx[valid]] = True

            kept_mask_blk_e = kept_mask_flat.view(B, L, K, E)  # [B, L, K, E]
            weights_flat = torch.zeros(BLK, E, dtype=h.dtype, device=h.device)
            weights_flat[keep_pos[valid], e_idx[valid]] = keep_vals[valid].to(h.dtype)
            weights_blk_e = weights_flat.view(B, L, K, E)      # [B, L, K, E]

            # permute to [B, L, E, K]
            dispatch = kept_mask_blk_e.permute(0, 1, 3, 2).to(h.dtype)
            combine  = weights_blk_e.permute(0, 1, 3, 2)
        else:
            # No truncation: keep all selected routes
            one_hot_flat = F.one_hot(idx_flat, num_classes=E).to(torch.bool)   # [BLK, E]
            dispatch_blk_e = one_hot_flat.view(B, L, K, E)                      # [B, L, K, E]
            weights_flat = torch.zeros(BLK, E, dtype=h.dtype, device=h.device)
            weights_flat[one_hot_flat] = val_flat.to(h.dtype)
            weights_blk_e = weights_flat.view(B, L, K, E)

            dispatch = dispatch_blk_e.permute(0, 1, 3, 2).to(h.dtype)           # [B, L, E, K]
            combine  = weights_blk_e.permute(0, 1, 3, 2)                        # [B, L, E, K]

        # ---- load-balancing loss (Switch-style) ----
        # meangates: mean prob per expert before truncation
        meangates = gates.mean(dim=(0, 1))                   # [E]
        # assigned: number of kept routes per expert after truncation
        assigned = dispatch.sum(dim=(0, 1, 3))               # [E]
        denom = (B * L * K) + 1e-9
        tokens_per_expert = assigned / denom                 # [E]
        aux_lb = (meangates * tokens_per_expert).sum() * E   # scalar

        return topk_idx, topk_vals, aux_lb, dispatch, combine


class MoEFeedForward(nn.Module):
    '''
    Switch-style MoE FFN with configurable Top-K routing and capacity truncation.
    '''
    def __init__(self, d_model, d_ff, n_experts=8, top_k=2, dropout=0.1,
                 activation=nn.GELU(), capacity_factor=1.25, drop_tokens=True):
        super().__init__()
        self.router = TopKRouter(d_model, n_experts, top_k=top_k,
                                 capacity_factor=capacity_factor, drop_tokens=drop_tokens)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                activation,
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
            ) for _ in range(n_experts)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # x: [B, L, D]
        in_dtype = x.dtype
        topk_idx, topk_vals, aux_lb, dispatch, combine = self.router(x)
        B, L, D = x.shape
        E = len(self.experts)

        # mask of tokens that actually go to expert e (after truncation)
        # shape: [B, L, E, 1]
        mask = (dispatch.sum(dim=-1) > 0).unsqueeze(-1)

        # Flatten and pre-allocate expert I/O buffers
        expert_inputs = (x.unsqueeze(2).expand(-1, -1, E, -1) * mask).view(B * L * E, D)
        expert_outputs = torch.zeros_like(expert_inputs)

        # Batch per expert to avoid per-token scatter/gather overhead
        for e, expert in enumerate(self.experts):
            idx = mask[:, :, e, 0].reshape(-1).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() > 0:
                out = expert(expert_inputs[idx])
                if out.dtype != expert_outputs.dtype:
                    out = out.to(expert_outputs.dtype)
                expert_outputs[idx] = out

        expert_outputs = expert_outputs.view(B, L, E, D)

        # Sum weights over K routes for each expert, then weighted sum over experts
        # combine: [B, L, E, K] -> [B, L, E, 1]
        combine_weight = combine.sum(dim=-1).unsqueeze(-1)
        y = (expert_outputs * combine_weight).sum(dim=2)  # [B, L, D]

        return self.dropout(y).to(in_dtype), aux_lb"""


"""class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff, dropout, activation="relu", use_moe=False, n_experts=8):
        super().__init__()
        self.attention = attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        act = F.relu if activation == "relu" else F.gelu
        if use_moe:
            self.ffn = MoEFeedForward(d_model, d_ff, n_experts=n_experts, dropout=dropout, activation=nn.GELU())
            self.use_moe = True
        else:
            self.conv1 = nn.Conv1d(d_model, d_ff, 1)
            self.conv2 = nn.Conv1d(d_ff, d_model, 1)
            self.activation = act
            self.use_moe = False

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, channel_cls, patch_cls = self.attention(x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = x + self.dropout(new_x)
        y = self.norm1(x)

        aux_lb = None
        if self.use_moe:
            y, aux_lb = self.ffn(y)                 # [B,L,D], aux load-balance loss
        else:
            y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
            y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), channel_cls, patch_cls, aux_lb  # aux_lb is None if not using MoE


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # accumulate only moe_aux across layers
        aux_total = None
        aux_count = 0
        for attn_layer in self.attn_layers:
            x, channel_cls, patch_cls, aux_lb = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)

            if aux_lb is not None:
                # initialize accumulator lazily to match device/dtype
                if aux_total is None:
                    aux_total = aux_lb
                else:
                    aux_total = aux_total + aux_lb
                aux_count += 1

        if self.norm is not None:
            x = self.norm(x)

        # if no MoE layer produced aux_lb, return a zero scalar tensor (safe for DP/DDP and autograd)
        if aux_total is None:
            aux_total = torch.zeros((), device=x.device, dtype=x.dtype)
        # If you prefer average across MoE layers instead of sum, uncomment below:
        if aux_count > 0:
            aux_total = aux_total / aux_count

        return x, channel_cls, patch_cls, aux_total"""


class EncoderLayerV2(nn.Module):
    def __init__(self, attention, d_model, d_ff, dropout, activation="relu"):
        super().__init__()
        self.attention = attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.conv1 = nn.Conv1d(d_model, 2 * d_ff, 1)  # 2*d_ff for SwiGLU split
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn_t, attn_c = self.attention(x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = x + self.dropout(new_x)
        y = self.norm1(x)
        y = self.conv1(y.transpose(-1, 1))
        y = swiglu(y)
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn_t, attn_c


class EncoderV2(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(EncoderV2, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # accumulate only moe_aux across layers
        attns_t = []
        attns_s = []
        for attn_layer in self.attn_layers:
            x, attn_t, attn_c = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns_t.append(attn_t)
            attns_s.append(attn_c)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns_t, attns_s


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff, dropout, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv4 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

    def forward(self, x_t, x_s, attn_mask=None, tau=None, delta=None):
        new_x_t, new_x_s, attn_t, attn_s = self.attention(x_t, x_s, attn_mask=attn_mask, tau=tau, delta=delta)
        x_t = x_t + self.dropout(new_x_t)
        x_s = x_s + self.dropout(new_x_s)

        y_t = x_t = self.norm1(x_t)
        y_s = x_s = self.norm3(x_s)

        y_t = self.dropout(self.activation(self.conv1(y_t.transpose(-1, 1))))
        y_t = self.dropout(self.conv2(y_t).transpose(-1, 1))

        y_s = self.dropout(self.activation(self.conv3(y_s.transpose(-1, 1))))
        y_s = self.dropout(self.conv4(y_s).transpose(-1, 1))

        return self.norm2(x_t + y_t), self.norm4(x_s + y_s), attn_t, attn_s


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x_t, x_s, attn_mask=None, tau=None, delta=None):
        attns_t = []
        attns_s = []
        for attn_layer in self.attn_layers:
            x_t, x_s, attn_t, attn_s = attn_layer(x_t, x_s, attn_mask=attn_mask, tau=tau, delta=delta)
            attns_t.append(attn_t)
            attns_s.append(attn_s)

        # (batch_size, patch_num+1, d_model), (batch_size, scaled_channel_num+1, d_model)
        return x_t, x_s, attns_t, attns_s


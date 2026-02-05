import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention
from einops import rearrange, repeat


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))  # Scaled Dot-Product Attention
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)  # multi-head
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class LEADLayerV2(nn.Module):
    '''
    Gated Temporal & Spatial attentions in parallel
    '''
    def __init__(
        self,
        d_model: int,
        enc_in: int,              # C
        n_heads: int,
        patch_num: int,   # P
        dropout: float = 0.1,
        output_attention: bool = False,
    ):
        super().__init__()
        self.patch_num = patch_num  # P
        self.d_model = d_model
        self.enc_in = enc_in                        # C
        self.output_attention = output_attention

        # Temporal attention (per-channel)
        self.temporal_attention = AttentionLayer(
            FullAttention(False, factor=1, attention_dropout=dropout, output_attention=output_attention),
            d_model, n_heads
        )
        # Spatial attention (per-patch across channels)
        self.spatial_attention = AttentionLayer(
            FullAttention(False, factor=1, attention_dropout=dropout, output_attention=output_attention),
            d_model, n_heads
        )

        # Optional norms (pre-norm style)
        self.t_norm = nn.LayerNorm(d_model)
        self.c_norm = nn.LayerNorm(d_model)

        # Fusion projection: map concatenated [x_t, x_c] to D-dim
        self.fuse_proj = nn.Linear(2 * d_model, d_model)  # Map [x_t, x_c] concat to D-dim
        nn.init.xavier_uniform_(self.fuse_proj.weight)  # Xavier init for stability
        nn.init.zeros_(self.fuse_proj.bias)  # Bias init to 0

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        '''
        x: [B, C*P, D]
        returns:
            x_next:        [B, C*P, D]
            'attn_t':   ...,
            'attn_c':   ...,
            }
        '''
        B, C_P, D = x.shape
        C, P = self.enc_in, int(C_P // self.enc_in)
        assert C_P == C * P and D == self.d_model, "Shape mismatch: expect [B, C*P, D]"

        # -------------------- Temporal branch --------------------
        # [B, C*P, D] -> [B*C, P, D]
        x_t = rearrange(x, 'b (c n) d -> (b c) n d', b=B, c=C, n=P)
        x_t = self.t_norm(x_t)
        # temporal attention
        x_t_out, attn_t = self.temporal_attention(x_t, x_t, x_t,
                                                  attn_mask=None, tau=tau, delta=delta)  # [B*C, P, D]
        x_t_out = rearrange(x_t_out, '(b c) n d -> b (c n) d', b=B, c=C)  # [B, C*P, D]

        # -------------------- Spatial branch --------------------
        # [B, C*P, D] -> [B*P, C, D]
        x_c = rearrange(x, 'b (c n) d -> (b n) c d', b=B, c=C, n=P)
        x_c = self.c_norm(x_c)
        # spatial attention
        x_c_out, attn_c = self.spatial_attention(x_c, x_c, x_c,
                                                 attn_mask=None, tau=tau, delta=delta)  # [B*P, C, D]
        x_c_out = rearrange(x_c_out, '(b n) c d -> b (c n) d', b=B, n=P)  # [B, C*P, D]

        # -------------------- Merge token streams --------------------
        gate = torch.sigmoid(self.fuse_proj(torch.cat([x_t_out, x_c_out], dim=-1)))  # [B, C*P, D]
        x_next = gate * x_t_out + (1.0 - gate) * x_c_out  # [B, C*P, D]

        return x_next, attn_t, attn_c


class LEADLayer(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        dropout=0.1,
        output_attention=False,
    ):
        super().__init__()

        # Temporal attention among cross channel patch embeddings
        self.temporal_attention = AttentionLayer(
            FullAttention(False, factor=1, attention_dropout=dropout, output_attention=output_attention),
            d_model, n_heads
        )

        # Spatial attention among up-scaled channel embeddings
        self.spatial_attention = AttentionLayer(
            FullAttention(False, factor=1, attention_dropout=dropout, output_attention=output_attention),
            d_model, n_heads
        )

    def forward(self, x_t, x_s, attn_mask=None, tau=None, delta=None):
        # Temporal attention
        x_out_t, attn_t = self.temporal_attention(x_t, x_t, x_t, attn_mask=attn_mask, tau=tau, delta=delta)
        # Spatial attention
        x_out_s, attn_s = self.spatial_attention(x_s, x_s, x_s, attn_mask=attn_mask, tau=tau, delta=delta)

        return x_out_t, x_out_s, attn_t, attn_s


class ADformerLayer(nn.Module):
    def __init__(
        self,
        num_blocks_t,
        num_blocks_c,
        d_model,
        n_heads,
        dropout=0.1,
        output_attention=False,
        no_inter=False,
    ):
        super().__init__()

        self.intra_attentions_t = nn.ModuleList(
            [
                AttentionLayer(
                    FullAttention(
                        False,
                        factor=1,
                        attention_dropout=dropout,
                        output_attention=output_attention,
                    ),
                    d_model,
                    n_heads,
                )
                for _ in range(num_blocks_t)
            ]
        )
        self.intra_attentions_c = nn.ModuleList(
            [
                AttentionLayer(
                    FullAttention(
                        False,
                        factor=1,
                        attention_dropout=dropout,
                        output_attention=output_attention,
                    ),
                    d_model,
                    n_heads,
                )
                for _ in range(num_blocks_c)
            ]
        )
        if no_inter or num_blocks_t <= 1:
            # print("No inter attention for time")
            self.inter_attention_t = None
        else:
            self.inter_attention_t = AttentionLayer(
                FullAttention(
                    False,
                    factor=1,
                    attention_dropout=dropout,
                    output_attention=output_attention,
                ),
                d_model,
                n_heads,
            )
        if no_inter or num_blocks_c <= 1:
            # print("No inter attention for channel")
            self.inter_attention_c = None
        else:
            self.inter_attention_c = AttentionLayer(
                FullAttention(
                    False,
                    factor=1,
                    attention_dropout=dropout,
                    output_attention=output_attention,
                ),
                d_model,
                n_heads,
            )

    def forward(self, x_t, x_c, attn_mask=None, tau=None, delta=None):
        attn_mask_t = ([None] * len(x_t))
        attn_mask_c = ([None] * len(x_c))

        # Intra attention
        x_intra_t = []
        attn_out_t = []
        x_intra_c = []
        attn_out_c = []
        # Temporal dimension
        for x_in_t, layer_t, mask in zip(x_t, self.intra_attentions_t, attn_mask_t):
            _x_out_t, _attn_t = layer_t(x_in_t, x_in_t, x_in_t, attn_mask=mask, tau=tau, delta=delta)
            x_intra_t.append(_x_out_t)  # (B, Ni, D)
            attn_out_t.append(_attn_t)
        # Channel dimension
        for x_in_c, layer_c, mask in zip(x_c, self.intra_attentions_c, attn_mask_c):
            _x_out_c, _attn_c = layer_c(x_in_c, x_in_c, x_in_c, attn_mask=mask, tau=tau, delta=delta)
            x_intra_c.append(_x_out_c)  # (B, C, D)
            attn_out_c.append(_attn_c)

        # Inter attention
        if self.inter_attention_t is not None:
            # Temporal dimension
            routers_t = torch.cat([x[:, -1:] for x in x_intra_t], dim=1)  # (B, n, D)
            x_inter_t, attn_inter_t = self.inter_attention_t(
                routers_t, routers_t, routers_t, attn_mask=None, tau=tau, delta=delta
            )
            x_out_t = [
                torch.cat([x[:, :-1], x_inter_t[:, i : i + 1]], dim=1)  # (B, Ni, D)
                for i, x in enumerate(x_intra_t)
            ]
            attn_out_t += [attn_inter_t]
        else:
            x_out_t = x_intra_t

        if self.inter_attention_c is not None:
            # Channel dimension
            routers_c = torch.cat([x[:, -1:] for x in x_intra_c], dim=1)  # (B, n, D)
            x_inter_c, attn_inter_c = self.inter_attention_c(
                routers_c, routers_c, routers_c, attn_mask=None, tau=tau, delta=delta
            )
            x_out_c = [
                torch.cat([x[:, :-1], x_inter_c[:, i : i + 1]], dim=1)  # (B, C, D)
                for i, x in enumerate(x_intra_c)
            ]
            attn_out_c += [attn_inter_c]
        else:
            x_out_c = x_intra_c

        return x_out_t, x_out_c, attn_out_t, attn_out_c


class FormerLayer(nn.Module):
    def __init__(self, num_blocks, d_model, n_heads, dropout=0.1, output_attention=False):
        super().__init__()

        self.intra_attentions = nn.ModuleList(
            [
                AttentionLayer(
                    FullAttention(
                        False,
                        factor=1,
                        attention_dropout=dropout,
                        output_attention=output_attention,
                    ),
                    d_model,
                    n_heads,
                )
                for _ in range(num_blocks)
            ]
        )


    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attn_mask = attn_mask or ([None] * len(x))

        x_out = []
        attn_out = []
        for x_in, layer, mask in zip(x, self.intra_attentions, attn_mask):
            _x_out, _attn = layer(x_in, x_in, x_in, attn_mask=mask, tau=tau, delta=delta)
            x_out.append(_x_out)  # (B, Li, D)
            attn_out.append(_attn)

        return x_out, attn_out


class DifferenceAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(DifferenceAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(
            torch.softmax(scale * scores, dim=-1)
        )  # Scaled Dot-Product Attention
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class DifferenceAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(DifferenceAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)  # multi-head
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries, keys, values, attn_mask, tau=tau, delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class DifferenceFormerlayer(nn.Module):
    def __init__(self, enc_in, num_blocks, d_model, n_heads, dropout=0.1, output_attention=False):
        super(DifferenceFormerlayer, self).__init__()
        self.intra_attentions = nn.ModuleList(
            [
                DifferenceAttentionLayer(
                    DifferenceAttention(
                        False,
                        factor=1,
                        attention_dropout=dropout,
                        output_attention=output_attention,
                    ),
                    d_model,
                    n_heads,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attn_mask = attn_mask or ([None] * len(x))

        x_out = []
        attn_out = []
        for x_in, layer, mask in zip(x, self.intra_attentions, attn_mask):
            _x_out, _attn = layer(x_in, x_in, x_in, attn_mask=mask, tau=tau, delta=delta)
            x_out.append(_x_out)  # (B, Li, D)
            attn_out.append(_attn)

        return x_out, attn_out
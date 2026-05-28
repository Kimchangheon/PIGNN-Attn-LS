from __future__ import annotations

import math

import torch
import torch.nn as nn

from torch_scatter import scatter_add, scatter_max


@torch.no_grad()
def segmented_softmax(logits_b_e_h: torch.Tensor, dst_e: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Vectorized softmax over incoming edges per (batch, head, node).

    Args:
        logits_b_e_h: (B, E, H)
        dst_e: (E,) destination node index in [0..N)
        num_nodes: N

    Returns:
        (B, E, H) normalized weights
    """

    B, E, H = logits_b_e_h.shape
    device = logits_b_e_h.device
    N = int(num_nodes)

    b_ids = torch.arange(B, device=device).view(B, 1, 1)
    h_ids = torch.arange(H, device=device).view(1, 1, H)
    dst = dst_e.view(1, E, 1)
    seg = (b_ids * (N * H)) + (h_ids * N) + dst
    seg = seg.reshape(-1)

    src = logits_b_e_h.reshape(-1)

    max_per_seg, _ = scatter_max(src, seg, dim=0, dim_size=B * N * H)
    max_g = max_per_seg.index_select(0, seg)
    x = torch.exp(src - max_g)
    denom = scatter_add(x, seg, dim=0, dim_size=B * N * H)
    denom_g = denom.index_select(0, seg)

    return (x / (denom_g + 1e-12)).reshape(B, E, H)


class EdgeSelfAttnBlock(nn.Module):
    """Sparse graph self-attention with edge bias.

    Uses segmented-softmax over incoming edges per (batch, head, node).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        edge_feat_dim: int,
        ffn_hidden: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = int(d_model)
        self.h = int(n_heads)
        self.dh = self.d_model // self.h

        self.q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out = nn.Linear(self.d_model, self.d_model, bias=False)

        self.edge_bias = nn.Sequential(
            nn.Linear(edge_feat_dim, max(8, edge_feat_dim * 2)),
            nn.LeakyReLU(0.1),
            nn.Linear(max(8, edge_feat_dim * 2), self.h),
        )

        self.ln1 = nn.LayerNorm(self.d_model)
        self.ln2 = nn.LayerNorm(self.d_model)

        hid = int(ffn_hidden or (4 * self.d_model))
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, hid),
            nn.GELU(),
            nn.Linear(hid, self.d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index_dir: torch.Tensor, edge_feat_dir: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x: (B, N, D)
            edge_index_dir: (E, 2) directed (src=j, dst=i)
            edge_feat_dir: (E, F)
        """

        B, N, D = x.shape
        device = x.device
        src = edge_index_dir[:, 0]
        dst = edge_index_dir[:, 1]

        y = self.ln1(x)
        Q = self.q(y).view(B, N, self.h, self.dh)
        K = self.k(y).view(B, N, self.h, self.dh)
        V = self.v(y).view(B, N, self.h, self.dh)

        Qi = Q[:, dst, :, :]
        Kj = K[:, src, :, :]
        Vj = V[:, src, :, :]

        logits = (Qi * Kj).sum(dim=-1) / math.sqrt(self.dh)
        bias = self.edge_bias(edge_feat_dir).unsqueeze(0)
        logits = logits + bias

        alpha = segmented_softmax(logits, dst, N)
        attn_msg = alpha.unsqueeze(-1) * Vj

        out = torch.zeros(B, N, self.h, self.dh, device=device, dtype=x.dtype)
        out.index_add_(1, dst, attn_msg)
        out = self.drop(self.out(out.reshape(B, N, D)))
        x = x + out

        z = self.ln2(x)
        z = self.drop(self.ffn(z))
        return x + z

from __future__ import annotations

import torch


def mismatch_inf_norm_per_candidate(
    Y: torch.Tensor,
    v: torch.Tensor,
    th: torch.Tensor,
    P_set: torch.Tensor,
    Q_set: torch.Tensor,
    slack_mask: torch.Tensor,
    pv_mask: torch.Tensor,
) -> torch.Tensor:
    """Return ∞-norm mismatch for each leading candidate dimension.

    Supports v/th with leading dims, e.g. (T,B,N) for T candidates.

    Returns:
        Tensor with shape v.shape[:-1]. Each entry is max over nodes of
        max(|ΔP|, |ΔQ|), with slack/PV masking applied.
    """

    Vc = v * torch.exp(1j * th)

    Yb = Y
    while Yb.dim() < Vc.dim() + 1:
        Yb = Yb.unsqueeze(0)

    Ic = torch.matmul(Yb, Vc.unsqueeze(-1)).squeeze(-1)
    Sc = Vc * Ic.conj()

    DP = (P_set - Sc.real)
    DQ = (Q_set - Sc.imag)

    DP = DP.masked_fill(slack_mask, 0.0)
    DQ = DQ.masked_fill(slack_mask | pv_mask, 0.0)

    DP_max = DP.abs().amax(dim=-1)
    DQ_max = DQ.abs().amax(dim=-1)
    return torch.maximum(DP_max, DQ_max)

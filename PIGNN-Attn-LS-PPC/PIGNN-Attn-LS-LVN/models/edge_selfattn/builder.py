"""Builder for GNSMsg_EdgeSelfAttn.

Importing this module decorates `_build` with `@register_model`, which
self-registers the model in `models.registry.MODEL_REGISTRY`. Because
`models/edge_selfattn/__init__.py` imports this file, simply importing the
`edge_selfattn` package is enough to make the model buildable by name.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.registry import register_model

from .model import GNSMsg_EdgeSelfAttn


def _construct(*, tied_heads: bool, bus_feat_extra_dim: int = 0, **kwargs) -> GNSMsg_EdgeSelfAttn:
    device = kwargs.pop("device")
    return GNSMsg_EdgeSelfAttn(
        d=kwargs["d"],
        d_hi=kwargs["d_hi"],
        K=kwargs["K"],
        pinn=kwargs["pinn"],
        gamma=kwargs["gamma"],
        v_limit=kwargs["v_limit"],
        use_armijo=kwargs["use_armijo"],
        dtheta_max=kwargs["dtheta_max"],
        dvm_frac=kwargs["dvm_frac"],
        num_attn_layers=kwargs["num_attn_layers"],
        tied_heads=tied_heads,
        bus_feat_extra_dim=bus_feat_extra_dim,
    ).to(device)


@register_model("GNSMsg_EdgeSelfAttn")
def _build_untied(
    *,
    d: int,
    d_hi: int,
    K: int,
    pinn: bool,
    gamma: float,
    v_limit: bool,
    use_armijo: bool,
    dtheta_max: float,
    dvm_frac: float,
    num_attn_layers: int,
    device: torch.device,
    **_unused,
) -> nn.Module:
    return _construct(
        tied_heads=False,
        d=d, d_hi=d_hi, K=K, pinn=pinn, gamma=gamma,
        v_limit=v_limit, use_armijo=use_armijo,
        dtheta_max=dtheta_max, dvm_frac=dvm_frac,
        num_attn_layers=num_attn_layers, device=device,
    )


@register_model("GNSMsg_EdgeSelfAttn_VnFeat")
def _build_vn_feat(
    *,
    d: int,
    d_hi: int,
    K: int,
    pinn: bool,
    gamma: float,
    v_limit: bool,
    use_armijo: bool,
    dtheta_max: float,
    dvm_frac: float,
    num_attn_layers: int,
    device: torch.device,
    **_unused,
) -> nn.Module:
    """LVN-targeted variant of PIGNN-Attn-LS.

    Adds a per-bus vn_log feature to the model input so the model can
    distinguish voltage classes (3kV / 20kV / 110kV / 380kV) in
    multi-voltage networks. Without this, the per-bus per-unit
    normalisation removes voltage-class information and every bus looks
    identical to the model -- which causes the rmse-plateau-at-V=1.0
    pathology seen on raw LVN training.

    bus_feat_extra_dim=1 -> one extra scalar (vn_log) appended to bus_feat.
    The dataset class populates vn_log automatically when the parquet has
    a vn_log column; HVN parquets without it pass zeros (no info loss
    since HVN is single-voltage).
    """
    return _construct(
        tied_heads=False,
        bus_feat_extra_dim=1,
        d=d, d_hi=d_hi, K=K, pinn=pinn, gamma=gamma,
        v_limit=v_limit, use_armijo=use_armijo,
        dtheta_max=dtheta_max, dvm_frac=dvm_frac,
        num_attn_layers=num_attn_layers, device=device,
    )


@register_model("GNSMsg_EdgeSelfAttn_Tied")
def _build_tied(
    *,
    d: int,
    d_hi: int,
    K: int,
    pinn: bool,
    gamma: float,
    v_limit: bool,
    use_armijo: bool,
    dtheta_max: float,
    dvm_frac: float,
    num_attn_layers: int,
    device: torch.device,
    **_unused,
) -> nn.Module:
    """Weight-tied K-iteration baseline.

    Same architecture as GNSMsg_EdgeSelfAttn but shares the (theta, v, m) head
    set across all K iterations. Gives a fair-parameter comparison against
    PE_DEQ_PF, which is also weight-shared.
    """
    return _construct(
        tied_heads=True,
        d=d, d_hi=d_hi, K=K, pinn=pinn, gamma=gamma,
        v_limit=v_limit, use_armijo=use_armijo,
        dtheta_max=dtheta_max, dvm_frac=dvm_frac,
        num_attn_layers=num_attn_layers, device=device,
    )

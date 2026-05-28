from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


@dataclass(frozen=True)
class OptimBundle:
    optim: torch.optim.Optimizer
    scheduler: Optional[CosineAnnealingWarmRestarts]


def build_optimizer_and_scheduler(
    *,
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
    lr_scheduler: str,
    cosine_restart_epoch: int,
    steps_per_epoch: int,
) -> OptimBundle:
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    scheduler = None
    if lr_scheduler == "CosineAnnealingLR":
        T_0 = cosine_restart_epoch * steps_per_epoch
        scheduler = CosineAnnealingWarmRestarts(optim, T_0=T_0, T_mult=1, eta_min=1e-6)

    return OptimBundle(optim=optim, scheduler=scheduler)

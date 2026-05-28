from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn

# Importing the `models` package triggers builder registration for every
# bundled model (see `models/__init__.py`). Adding a new model does NOT
# require any edit to this file — only a new subpackage under `models/`.
import models  # noqa: F401  -- side-effect: populates MODEL_REGISTRY

from models.registry import build_model


def create_model(*, model_name: str, device: torch.device, **model_kwargs) -> nn.Module:
    """Build a model by registered name. Adding a new model = adding a builder."""
    return build_model(model_name, device=device, **model_kwargs)


def init_weights(model: nn.Module, *, weight_init: str, bias_init: float, exclude_modules: List[nn.Module] | None = None) -> None:
    exclude_modules = exclude_modules or []

    for module in model.modules():
        if module in exclude_modules:
            continue

        if isinstance(module, nn.Linear):
            if weight_init == "sd0.02":
                torch.nn.init.normal_(module.weight, mean=0, std=0.02)
            elif weight_init == "He":
                torch.nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")

            if module.bias is not None:
                module.bias.data.fill_(bias_init)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)

        else:
            for name, param in module.named_parameters(recurse=False):
                if "weight" in name and param.dim() > 1:
                    if weight_init == "sd0.02":
                        torch.nn.init.normal_(param, mean=0, std=0.02)
                    elif weight_init == "He":
                        torch.nn.init.kaiming_uniform_(param, nonlinearity="relu")
                elif "bias" in name:
                    param.data.fill_(bias_init)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

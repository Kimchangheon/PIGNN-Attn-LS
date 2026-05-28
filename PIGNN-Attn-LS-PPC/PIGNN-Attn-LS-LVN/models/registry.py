"""Model registry.

Lives alongside the model packages it indexes. Each model directory
`models/<name>/` provides a `builder.py` that uses `@register_model("<Name>")`
to self-register. Adding a new model is a matter of:

  1. Create `models/<name>/` with `model.py` and `builder.py`.
  2. Import the subpackage from `models/__init__.py` so its builder runs
     at package-import time.

No edits to `train/modeling.py` are required.

Usage::

    from models.registry import register_model, build_model

    @register_model("MyModel")
    def _build_my_model(*, d, d_hi, device, **_unused):
        return MyModel(d=d, d_hi=d_hi).to(device)

    # Later:
    model = build_model("MyModel", d=4, d_hi=16, device=device)

Builders accept arbitrary `**kwargs`; each declares the kwargs it actually
uses and ignores the rest via `**_unused`. This means different models can
have different parameter sets without forcing a uniform signature.
"""

from __future__ import annotations

from typing import Callable, Dict

import torch.nn as nn


ModelBuilder = Callable[..., nn.Module]

MODEL_REGISTRY: Dict[str, ModelBuilder] = {}


def register_model(name: str) -> Callable[[ModelBuilder], ModelBuilder]:
    """Decorator: register `builder` under `name`. Raises if the name is taken."""

    def decorator(builder: ModelBuilder) -> ModelBuilder:
        if name in MODEL_REGISTRY:
            raise ValueError(
                f"Model already registered: {name!r}. "
                f"Existing builder: {MODEL_REGISTRY[name].__qualname__}"
            )
        MODEL_REGISTRY[name] = builder
        return builder

    return decorator


def build_model(name: str, **kwargs) -> nn.Module:
    """Look up `name` in the registry and call its builder with `kwargs`."""
    if name not in MODEL_REGISTRY:
        registered = ", ".join(sorted(MODEL_REGISTRY.keys())) or "<none>"
        raise ValueError(f"Unknown model: {name!r}. Registered: {registered}")
    return MODEL_REGISTRY[name](**kwargs)


def registered_models() -> list[str]:
    """Sorted list of currently registered model names."""
    return sorted(MODEL_REGISTRY.keys())

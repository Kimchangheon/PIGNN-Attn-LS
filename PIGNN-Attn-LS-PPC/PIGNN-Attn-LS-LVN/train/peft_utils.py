from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LoRAParams:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.0

    @property
    def scaling(self) -> float:
        r = max(1, int(self.r))
        return float(self.alpha) / float(r)


class LoRALinear(nn.Module):
    """A drop-in replacement for nn.Linear with LoRA adapters.

    Computes:
        y = xW^T + b + scale * (dropout(x) A^T B^T)

    where A is (r, in_features), B is (out_features, r).

    Notes:
    - The wrapped base linear layer is kept as a submodule (so you can still
      load pretrained weights before/after wrapping if needed).
    - LoRA weights are initialized so the initial delta is ~0 (B zeros).
    """

    def __init__(
        self,
        base: nn.Linear,
        *,
        r: int,
        alpha: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got: {type(base).__name__}")

        r = int(r)
        if r <= 0:
            raise ValueError("LoRA rank r must be > 0")

        self.base = base
        self.in_features = int(base.in_features)
        self.out_features = int(base.out_features)

        self.r = int(r)
        self.alpha = int(alpha)
        self.scale = float(alpha) / float(r)
        self.drop = nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity()

        # A: (r, in), B: (out, r)
        # Important: wrap happens after model.to(device) in this repo, so we must
        # create adapter weights on the same device/dtype as the base weights.
        w = base.weight
        self.lora_A = nn.Parameter(torch.empty(self.r, self.in_features, device=w.device, dtype=w.dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.r, device=w.device, dtype=w.dtype))

        # Init A ~ N(0, 0.02), B = 0 so initial delta is zero.
        nn.init.normal_(self.lora_A, mean=0.0, std=0.02)

        # Ensure the wrapper module is on the same device as the base.
        # (This is cheap and prevents accidental CPU params.)
        self.to(device=w.device, dtype=w.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        dx = self.drop(x)
        # (.., in) @ (in, r) -> (.., r)
        z = F.linear(dx, self.lora_A)  # weight (r, in)
        # (.., r) @ (r, out) -> (.., out)
        z = F.linear(z, self.lora_B)  # weight (out, r)
        return y + (z * self.scale)


def _get_parent_module(root: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    if len(parts) == 1:
        return root, parts[0]

    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
        if not isinstance(parent, nn.Module):
            raise AttributeError(f"Parent path '{p}' in '{qualified_name}' is not a module")
    return parent, parts[-1]


def apply_lora_to_linear_modules(
    model: nn.Module,
    *,
    target_module_names: Sequence[str],
    r: int,
    alpha: int,
    dropout: float,
) -> list[str]:
    """Replace selected nn.Linear submodules with LoRALinear.

    Selection rule:
    - A linear module is wrapped if its attribute name (the final component of
      its qualified name) is in target_module_names.

    Returns:
      List of qualified names that were wrapped.
    """

    targets = set(str(x) for x in target_module_names)
    to_wrap: list[str] = []

    for name, module in model.named_modules():
        if not name:
            continue
        if isinstance(module, nn.Linear):
            attr = name.split(".")[-1]
            if attr in targets:
                to_wrap.append(name)

    wrapped: list[str] = []
    for qname in to_wrap:
        parent, attr = _get_parent_module(model, qname)
        base = getattr(parent, attr)
        if not isinstance(base, nn.Linear):
            continue
        setattr(parent, attr, LoRALinear(base, r=r, alpha=alpha, dropout=dropout))
        wrapped.append(qname)

    return wrapped


def freeze_all_except_lora(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False

    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True


def freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def _get_submodule(root: nn.Module, qualified_name: str) -> nn.Module:
    """Resolve a (possibly dotted) submodule name starting from root."""
    name = str(qualified_name).strip()
    if not name:
        raise ValueError("Empty module name")

    cur: nn.Module = root
    for part in name.split("."):
        nxt = getattr(cur, part, None)
        if not isinstance(nxt, nn.Module):
            raise AttributeError(f"Module '{type(cur).__name__}' has no submodule '{part}' (from '{qualified_name}')")
        cur = nxt
    return cur


def unfreeze_modules(model: nn.Module, module_names: Sequence[str]) -> list[str]:
    """Unfreeze (set requires_grad=True) for parameters in selected submodules.

    Returns a list of module names that were successfully unfrozen.
    """

    unfrozen: list[str] = []
    for raw_name in module_names:
        name = str(raw_name).strip()
        if not name:
            continue
        try:
            mod = _get_submodule(model, name)
        except Exception:
            continue
        for p in mod.parameters():
            p.requires_grad = True
        unfrozen.append(name)
    return unfrozen


@torch.no_grad()
def merge_lora_weights(model: nn.Module) -> list[str]:
    """Merge LoRA weights into base Linear layers and replace LoRALinear modules.

    Returns the list of qualified module names that were merged.
    """

    merged: list[str] = []

    for name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue

        parent, attr = _get_parent_module(model, name)
        base = module.base

        # Compute merged weight: W + scale * (B @ A)
        delta = torch.matmul(module.lora_B, module.lora_A) * float(module.scale)
        weight = base.weight.data + delta.to(dtype=base.weight.dtype, device=base.weight.device)

        merged_linear = nn.Linear(base.in_features, base.out_features, bias=base.bias is not None)
        merged_linear.to(device=base.weight.device, dtype=base.weight.dtype)
        merged_linear.weight.data.copy_(weight)
        if base.bias is not None and merged_linear.bias is not None:
            merged_linear.bias.data.copy_(base.bias.data)

        setattr(parent, attr, merged_linear)
        merged.append(name)

    return merged


def iter_trainable_params(model: nn.Module) -> Iterable[nn.Parameter]:
    return (p for p in model.parameters() if p.requires_grad)


def count_trainable_params(model: nn.Module) -> int:
    return sum(int(p.numel()) for p in model.parameters() if p.requires_grad)

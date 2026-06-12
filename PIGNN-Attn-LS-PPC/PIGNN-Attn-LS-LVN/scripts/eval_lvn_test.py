"""Evaluate a trained LVN checkpoint on the TEST split.

The existing scripts/eval_both_metrics.py is HVN-only (hardcoded parquet path,
no VnFeat support, only computes val metrics). This script targets the
LVN_converted_n36000_v2.parquet test split with the VnFeat variant.

Usage:
    python scripts/eval_lvn_test.py <run_dir>

Example:
    python scripts/eval_lvn_test.py results/runs/260523-181350_17a8
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import torch

import models  # registers all model names
from models.registry import build_model
from train.data import build_dataloaders


def _infer_kwargs(run_dir: Path) -> tuple[dict, str]:
    log = (run_dir / "train.log").read_text()
    m = lambda pat, default: (
        type(default)(re.search(pat, log).group(1)) if re.search(pat, log) else default
    )
    kwargs = dict(
        d=m(r"\bd:\s*(\d+)", 4),
        d_hi=m(r"\bd_hi:\s*(\d+)", 16),
        num_attn_layers=m(r"\battn_layers:\s*(\d+)", 1),
        K=m(r"\bK:\s*(\d+)", 10),
        dtheta_max=m(r"DthetaMax:\s*([\d.]+)", 0.30),
        dvm_frac=m(r"DvmFrac:\s*([\d.]+)", 0.10),
        gamma=0.9,
        v_limit=True,
        use_armijo=True,
        pinn=True,  # forces phys loss return
        bus_feat_extra_dim=1,  # VnFeat
    )
    # Extract model name from log
    m_name = re.search(r"MODEL:(\S+?),", log)
    model_name = m_name.group(1) if m_name else "GNSMsg_EdgeSelfAttn_VnFeat"
    return kwargs, model_name


@torch.no_grad()
def evaluate(run_dir: Path, parquet: str) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs, model_name = _infer_kwargs(run_dir)
    print(f"  model: {model_name}", flush=True)
    print(f"  kwargs: {kwargs}", flush=True)

    model = build_model(model_name, device=device, **kwargs)
    state = torch.load(run_dir / "ckpt" / "best.ckpt", map_location=device, weights_only=False)
    model.load_state_dict(state, strict=True)
    model.eval()

    splits = build_dataloaders(
        parquet_paths=[parquet],
        per_unit=True,
        device=device,
        batch_size=32,  # same as training
        block_diag=True,
        seed=42,  # same as training
        split_mode="ratio",
        train_ratio=0.8,
        valid_ratio=0.1,
    )
    print(f"  test split: {splits.n_test} samples", flush=True)

    results = {}
    for split_name, loader in [("val", splits.val_loader), ("test", splits.test_loader)]:
        sum_se = sum_mag = sum_ang = sum_phys = 0.0
        n = 0
        for batch in loader:
            n_nodes = batch["sizes"].to(device)
            bus_type = batch["bus_type"].to(device)
            Line = batch["Lines_connected"].to(device)
            Ys = batch["Y_Lines"].to(device)
            Yc = batch["Y_C_Lines"].to(device)
            S = batch["S_start"].to(device)
            V0 = batch["V_start"].to(device)
            Vt = batch["V_newton"].to(device)
            vn_log = batch.get("vn_log", None)
            extra = {}
            if isinstance(vn_log, torch.Tensor):
                extra["vn_log"] = vn_log.to(device)

            out = model(bus_type, Line, None, Ys, Yc, S, V0, n_nodes, **extra)
            V_pred, phys = (out if isinstance(out, tuple) else (out, None))

            B = bus_type.size(0)
            diff = V_pred - Vt
            sum_se += float((diff * diff).mean().item()) * B
            sum_mag += float((diff[..., 0] ** 2).mean().item()) * B
            sum_ang += float((diff[..., 1] ** 2).mean().item()) * B
            if phys is not None:
                sum_phys += float(phys.item()) * B
            n += B
        results[split_name] = dict(
            n=n,
            rmse=(sum_se / n) ** 0.5,
            rmse_mag=(sum_mag / n) ** 0.5,
            rmse_ang_deg=(sum_ang / n) ** 0.5 * 180.0 / math.pi,
            phys_loss=sum_phys / n,
        )
    return results


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/eval_lvn_test.py <run_dir> [parquet]", file=sys.stderr)
        return 1
    run_dir = Path(sys.argv[1]).resolve()
    parquet = sys.argv[2] if len(sys.argv) > 2 else "./datasets/LVN_converted_n36000_v2.parquet"
    print(f"Run dir: {run_dir}")
    print(f"Parquet: {parquet}")
    results = evaluate(run_dir, parquet)
    print()
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

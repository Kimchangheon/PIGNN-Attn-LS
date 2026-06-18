#!/usr/bin/env python3
"""
Train/evaluate the released GridSFM surrogate on the PPC LVN parquet pipeline.

This keeps the existing PPC dataloader, split logic, supervised Newton target,
and complex128 AC-PF residual evaluation. Only the neural surrogate is swapped:
LVN branch-row parquet batches are adapted into GridSFM's HeteroData schema.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torch_geometric.data import HeteroData

from Dataset_optimized_complex_columns import ChanghunDataset
from collate_blockdiag_optimized_complex_columns import collate_blockdiag
from train_valid_test_gridfm import (
    Tee,
    angle_diff,
    cap_subset,
    compute_power_flow_residual_metrics,
    ensure_dense_y_for_metrics,
    format_residual_distribution_compact,
    ppc_physics_loss,
    residual_distribution,
)


BUS_TYPE_PQ = 1
BUS_TYPE_PV = 2
BUS_TYPE_REF = 3


def parse_args():
    parser = argparse.ArgumentParser(
        description="GridSFM released backbone on PPC branch-row parquet data",
    )
    parser.add_argument("--PARQUET", type=str, required=True)
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--log_to_file", action="store_true")
    parser.add_argument("--log_dir", type=str, default="./results/logs/gridsfm")
    parser.add_argument("--ckpt_dir", type=str, default="./results/ckpt/gridsfm")

    parser.add_argument(
        "--pretrained_checkpoint",
        type=str,
        default="",
        help="Released GridSFM checkpoint, e.g. gridsfm_open_v1.1.pt.",
    )
    parser.add_argument(
        "--init_mode",
        choices=("pretrained", "scratch"),
        default="pretrained",
        help="Use released GridSFM weights or random GridSFM backbone init.",
    )

    parser.add_argument("--PER_UNIT", action="store_true")
    parser.add_argument("--target_S_base", type=float, default=None)
    parser.add_argument(
        "--dataset_complex_dtype",
        choices=("complex64", "complex128"),
        default="complex128",
    )
    parser.add_argument("--share_grid", action="store_true")
    parser.add_argument("--share_ybus", action="store_true")
    parser.add_argument("--lazy_parquet", action="store_true")
    parser.add_argument("--row_group_cache_size", type=int, default=2)

    parser.add_argument("--BATCH", type=int, default=4)
    parser.add_argument("--EPOCHS", type=int, default=40)
    parser.add_argument("--LR", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed_value", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.3333)
    parser.add_argument("--valid_ratio", type=float, default=0.3333)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_valid_samples", type=int, default=0)
    parser.add_argument("--max_test_samples", type=int, default=0)
    parser.add_argument("--VAL_EVERY", type=int, default=1)

    parser.add_argument("--mse_weight", type=float, default=1.0)
    parser.add_argument("--physics_weight", type=float, default=1e-2)
    parser.add_argument(
        "--physics_loss_form",
        choices=("mse", "huber", "logcosh"),
        default="logcosh",
    )
    parser.add_argument("--physics_huber_delta", type=float, default=1.0)
    parser.add_argument("--residual_tol_pu", type=float, default=1e-6)
    parser.add_argument("--convergence_tol_pu", type=float, default=1e-6)
    parser.add_argument("--vmin", type=float, default=0.5)
    parser.add_argument("--vmax", type=float, default=1.5)

    parser.add_argument(
        "--treat_voltage_mismatch_as_transformer",
        action="store_true",
        help="Classify branch rows with different endpoint vn_kv as GridSFM transformers.",
    )
    parser.add_argument(
        "--rate_a",
        type=float,
        default=0.0,
        help="Fallback GridSFM branch rate_a feature when the parquet has no rate.",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_dataset(dataset, train_ratio, valid_ratio, seed):
    n = len(dataset)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    n_test = n - n_train - n_valid
    if min(n_train, n_valid, n_test) <= 0:
        raise ValueError(
            f"Bad split sizes for n={n}: train={n_train}, valid={n_valid}, test={n_test}",
        )
    gen = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_valid, n_test], generator=gen)


def _cpu(x):
    return x.detach().cpu() if isinstance(x, torch.Tensor) else x


def _map_bus_types(ppc_bus_type: torch.Tensor) -> torch.Tensor:
    out = torch.full_like(ppc_bus_type.long(), BUS_TYPE_PQ)
    out[ppc_bus_type.long() == 2] = BUS_TYPE_PV
    out[ppc_bus_type.long() == 1] = BUS_TYPE_REF
    return out


def _safe_inverse_admittance(y: torch.Tensor) -> torch.Tensor:
    tiny = torch.tensor(1e-12, dtype=y.real.dtype)
    mask = y.abs() > tiny
    z = torch.zeros_like(y)
    z[mask] = 1.0 / y[mask]
    return z


def _make_branch_family(
    f: torch.Tensor,
    t: torch.Tensor,
    y_series: torch.Tensor,
    ysh_f: torch.Tensor,
    ysh_t: torch.Tensor,
    tau: torch.Tensor,
    shift_deg: torch.Tensor,
    mask: torch.Tensor,
    *,
    is_transformer: bool,
    rate_a: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    idx = torch.where(mask)[0]
    if idx.numel() == 0:
        cols = 11 if is_transformer else 9
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, cols), dtype=torch.float32)

    ff = f[idx].long()
    tt = t[idx].long()
    yy = y_series[idx].to(torch.complex128)
    zz = _safe_inverse_admittance(yy)
    r = zz.real.float()
    x = zz.imag.float()
    bfr = (ysh_f[idx].to(torch.complex128).imag.float()) / 2.0
    bto = (ysh_t[idx].to(torch.complex128).imag.float()) / 2.0
    shift_rad = torch.deg2rad(shift_deg[idx].float())
    tap = tau[idx].float()
    tap = torch.where(tap.abs() < 1e-9, torch.ones_like(tap), tap)
    edge_index = torch.stack([ff, tt], dim=0).contiguous()

    if is_transformer:
        attr = torch.zeros((idx.numel(), 11), dtype=torch.float32)
        attr[:, 0] = -math.pi
        attr[:, 1] = math.pi
        attr[:, 2] = r
        attr[:, 3] = x
        attr[:, 4] = float(rate_a)
        attr[:, 7] = tap
        attr[:, 8] = shift_rad
        attr[:, 9] = bfr
        attr[:, 10] = bto
    else:
        attr = torch.zeros((idx.numel(), 9), dtype=torch.float32)
        attr[:, 0] = -math.pi
        attr[:, 1] = math.pi
        attr[:, 2] = bfr
        attr[:, 3] = bto
        attr[:, 4] = r
        attr[:, 5] = x
        attr[:, 6] = float(rate_a)
    return edge_index, attr


def make_gridsfm_graphs(batch_cpu: Dict[str, torch.Tensor], args) -> Tuple[List[HeteroData], torch.Tensor]:
    from gridsfm import prepare_for_inference

    sizes = batch_cpu["sizes"].long()
    branch_sizes = batch_cpu["branch_sizes"].long()
    offsets = batch_cpu["offsets"].long()
    branch_offsets = torch.cat((branch_sizes.new_zeros(1), torch.cumsum(branch_sizes, 0)[:-1]))

    graphs: List[HeteroData] = []
    target = batch_cpu["V_newton"].squeeze(0).float()

    for gi, (n, nl, off, boff) in enumerate(zip(sizes, branch_sizes, offsets, branch_offsets)):
        n = int(n.item())
        nl = int(nl.item())
        off = int(off.item())
        boff = int(boff.item())
        bs = slice(off, off + n)
        es = slice(boff, boff + nl)

        bus_type_ppc = batch_cpu["bus_type"].squeeze(0)[bs].long()
        bus_type = _map_bus_types(bus_type_ppc)
        vn_kv = batch_cpu["vn_kv"].squeeze(0)[bs].float()
        Vstart = batch_cpu["V_start"].squeeze(0)[bs].float()
        S = batch_cpu["S_start"].squeeze(0)[bs].to(torch.complex128)
        Ysh = batch_cpu["Y_shunt_bus"].squeeze(0)[bs].to(torch.complex128)

        d = HeteroData()
        bus_x = torch.zeros((n, 4), dtype=torch.float32)
        bus_x[:, 0] = vn_kv
        bus_x[:, 1] = bus_type.float()
        bus_x[:, 2] = float(args.vmin)
        bus_x[:, 3] = float(args.vmax)
        d["bus"].x = bus_x
        d["bus"].v_setpoint = Vstart[:, 0:1].clone()

        gen_bus = torch.where((bus_type_ppc == 1) | (bus_type_ppc == 2))[0]
        if gen_bus.numel() == 0:
            gen_bus = torch.tensor([0], dtype=torch.long)
        xg = torch.zeros((gen_bus.numel(), 11), dtype=torch.float32)
        Pg0 = S.real[gen_bus].float()
        Qg0 = S.imag[gen_bus].float()
        p_margin = torch.maximum(Pg0.abs() * 0.5, torch.ones_like(Pg0) * 10.0)
        q_margin = torch.maximum(Qg0.abs() * 0.5, torch.ones_like(Qg0) * 10.0)
        xg[:, 0] = 1.0
        xg[:, 2] = Pg0 - p_margin
        xg[:, 3] = Pg0 + p_margin
        xg[:, 5] = Qg0 - q_margin
        xg[:, 6] = Qg0 + q_margin
        xg[:, 7] = Vstart[gen_bus, 0]
        d["generator"].x = xg
        gen_idx = torch.arange(gen_bus.numel(), dtype=torch.long)
        d["generator", "generator_link", "bus"].edge_index = torch.stack([gen_idx, gen_bus], dim=0)
        d["bus", "generator_link", "generator"].edge_index = torch.stack([gen_bus, gen_idx], dim=0)

        load_bus = torch.where(((-S.real).abs() > 0) | ((-S.imag).abs() > 0))[0]
        if load_bus.numel() > 0:
            xl = torch.zeros((load_bus.numel(), 2), dtype=torch.float32)
            xl[:, 0] = torch.clamp(-S.real[load_bus].float(), min=0.0)
            xl[:, 1] = torch.clamp(-S.imag[load_bus].float(), min=0.0)
            d["load"].x = xl
            li = torch.arange(load_bus.numel(), dtype=torch.long)
            d["load", "load_link", "bus"].edge_index = torch.stack([li, load_bus], dim=0)
            d["bus", "load_link", "load"].edge_index = torch.stack([load_bus, li], dim=0)

        sh_mask = Ysh.abs() > 0
        sh_bus = torch.where(sh_mask)[0]
        if sh_bus.numel() > 0:
            xs = torch.zeros((sh_bus.numel(), 2), dtype=torch.float32)
            xs[:, 0] = Ysh.imag[sh_bus].float()
            xs[:, 1] = Ysh.real[sh_bus].float()
            d["shunt"].x = xs
            si = torch.arange(sh_bus.numel(), dtype=torch.long)
            d["shunt", "shunt_link", "bus"].edge_index = torch.stack([si, sh_bus], dim=0)
            d["bus", "shunt_link", "shunt"].edge_index = torch.stack([sh_bus, si], dim=0)

        f = batch_cpu["Branch_f_bus"].squeeze(0)[es].long() - off
        t = batch_cpu["Branch_t_bus"].squeeze(0)[es].long() - off
        status = batch_cpu["Branch_status"].squeeze(0)[es].bool()
        tau = batch_cpu["Branch_tau"].squeeze(0)[es].float()
        shift = batch_cpu["Branch_shift_deg"].squeeze(0)[es].float()
        yft = batch_cpu["Branch_y_series_ft"].squeeze(0)[es].to(torch.complex128)
        yf = batch_cpu["Branch_y_series_from"].squeeze(0)[es].to(torch.complex128)
        y_series = torch.where(yft.abs() > 1e-12, yft, yf)
        ysh_f = batch_cpu["Branch_y_shunt_from"].squeeze(0)[es].to(torch.complex128)
        ysh_t = batch_cpu["Branch_y_shunt_to"].squeeze(0)[es].to(torch.complex128)

        if "Is_trafo" in batch_cpu:
            is_tr = batch_cpu["Is_trafo"].squeeze(0)[es].bool()
        else:
            is_tr = torch.zeros(nl, dtype=torch.bool)
        is_tr = is_tr | ((tau - 1.0).abs() > 1e-8) | (shift.abs() > 1e-8)
        if args.treat_voltage_mismatch_as_transformer and nl > 0:
            valid_ft = (f >= 0) & (f < n) & (t >= 0) & (t < n)
            vn_mismatch = torch.zeros(nl, dtype=torch.bool)
            vn_mismatch[valid_ft] = (vn_kv[f[valid_ft]] - vn_kv[t[valid_ft]]).abs() > 1e-6
            is_tr = is_tr | vn_mismatch

        ac_mask = status & (~is_tr)
        tr_mask = status & is_tr
        ac_ei, ac_attr = _make_branch_family(
            f, t, y_series, ysh_f, ysh_t, tau, shift, ac_mask,
            is_transformer=False, rate_a=args.rate_a,
        )
        tr_ei, tr_attr = _make_branch_family(
            f, t, y_series, ysh_f, ysh_t, tau, shift, tr_mask,
            is_transformer=True, rate_a=args.rate_a,
        )
        if ac_ei.size(1) > 0:
            d["bus", "ac_line", "bus"].edge_index = ac_ei
            d["bus", "ac_line", "bus"].edge_attr = ac_attr
        if tr_ei.size(1) > 0:
            d["bus", "transformer", "bus"].edge_index = tr_ei
            d["bus", "transformer", "bus"].edge_attr = tr_attr

        d.feasible = torch.tensor(1, dtype=torch.long)
        d = prepare_for_inference(d)
        graphs.append(d)

    return graphs, target


def gridsfm_forward(model, batch_cpu: Dict[str, torch.Tensor], args, device) -> torch.Tensor:
    from gridsfm import batch_data_list

    graphs, _ = make_gridsfm_graphs(batch_cpu, args)
    gbatch = batch_data_list(graphs, copy=False).to(device)
    out = model(gbatch)
    pred = out["bus"].pred
    theta = torch.atan2(torch.sin(pred[:, 0]), torch.cos(pred[:, 0]))
    mag = pred[:, 1]
    return torch.stack([mag, theta], dim=-1).unsqueeze(0)


def make_model(args, device):
    if args.init_mode == "pretrained":
        if not args.pretrained_checkpoint:
            raise ValueError("--init_mode pretrained requires --pretrained_checkpoint")
        from gridsfm import load_model

        model = load_model(args.pretrained_checkpoint, device=device)
        model.train()
        return model

    from gridsfm import GridTransformerBackbone

    model = GridTransformerBackbone().to(device)
    return model


def run():
    args = parse_args()
    set_seed(args.seed_value)

    if not args.run_name:
        mode = "pre" if args.init_mode == "pretrained" else "scratch"
        args.run_name = f"gridsfm_{mode}_{Path(args.PARQUET).stem}"
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    if args.log_to_file:
        log_path = os.path.join(args.log_dir, f"{args.run_name}_training_log.txt")
        log_f = open(log_path, "a", buffering=1)
        sys.stdout = Tee(sys.__stdout__, log_f)
        sys.stderr = Tee(sys.__stderr__, log_f)
        print(f"[log] tee -> {log_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device} torch={torch.__version__}")
    print(f"[run] {args.run_name}")
    print(f"[data] PARQUET={args.PARQUET}")
    print(
        f"[data] PER_UNIT={args.PER_UNIT} target_S_base={args.target_S_base} "
        f"dtype={args.dataset_complex_dtype} share_grid={args.share_grid}"
    )
    print(
        f"[model] GridSFM init_mode={args.init_mode} "
        f"pretrained_checkpoint={args.pretrained_checkpoint or '<none>'}"
    )
    print(
        f"[loss] mse_weight={args.mse_weight} physics_weight={args.physics_weight} "
        f"physics_form={args.physics_loss_form}"
    )

    dataset = ChanghunDataset(
        args.PARQUET,
        per_unit=args.PER_UNIT,
        target_S_base=args.target_S_base,
        share_grid=args.share_grid,
        share_ybus=args.share_ybus or args.share_grid,
        lazy_row_groups=args.lazy_parquet,
        row_group_cache_size=args.row_group_cache_size,
        complex_dtype=args.dataset_complex_dtype,
    )
    train_ds, val_ds, test_ds = split_dataset(
        dataset,
        args.train_ratio,
        args.valid_ratio,
        args.seed_value,
    )
    train_ds = cap_subset(train_ds, args.max_train_samples)
    val_ds = cap_subset(val_ds, args.max_valid_samples)
    test_ds = cap_subset(test_ds, args.max_test_samples)
    print(f"[split] train={len(train_ds)} valid={len(val_ds)} test={len(test_ds)}")

    loader_kwargs = {
        "batch_size": args.BATCH,
        "collate_fn": collate_blockdiag,
        "num_workers": 0,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    train_eval_loader = DataLoader(train_ds, shuffle=False, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    model = make_model(args, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable_params={n_params:,}")

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.LR,
        weight_decay=args.weight_decay,
    )
    best_val = float("inf")
    best_path = os.path.join(args.ckpt_dir, f"{args.run_name}_best.pt")

    def run_epoch(loader, train):
        model.train(train)
        sum_loss = 0.0
        sum_mse = 0.0
        sum_mse_mag = 0.0
        sum_mse_ang = 0.0
        sum_phys = 0.0
        sum_max_dp = 0.0
        sum_max_dq = 0.0
        sum_max_dp_mva = 0.0
        sum_max_dq_mva = 0.0
        n_graphs = 0
        n_conv = 0
        dp_values = []
        dq_values = []

        with torch.set_grad_enabled(train):
            for batch_cpu in loader:
                B_eff = int(batch_cpu["sizes"].numel())
                n_graphs += B_eff
                n_nodes = batch_cpu["sizes"].to(device)
                batch_dev = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch_cpu.items()
                }
                bus_type = batch_dev["bus_type"]
                Sstart_metric = batch_dev["S_start"]
                S_base_metric = batch_dev.get("S_base", None)
                if S_base_metric is None and args.target_S_base is not None:
                    S_base_metric = torch.full(
                        (B_eff,),
                        float(args.target_S_base),
                        dtype=torch.float64,
                        device=device,
                    )
                Y = batch_dev.get("Ybus", None)
                Y_metric = ensure_dense_y_for_metrics(
                    Y,
                    bus_type,
                    batch_dev["Branch_f_bus"],
                    batch_dev["Branch_t_bus"],
                    batch_dev["Branch_status"],
                    batch_dev["Branch_tau"],
                    batch_dev["Branch_shift_deg"],
                    batch_dev["Branch_y_series_from"],
                    batch_dev["Branch_y_series_to"],
                    batch_dev["Branch_y_series_ft"],
                    batch_dev["Branch_y_shunt_from"],
                    batch_dev["Branch_y_shunt_to"],
                    batch_dev["Y_shunt_bus"],
                )

                target = batch_dev["V_newton"].float()
                Vpred = gridsfm_forward(model, batch_cpu, args, device)

                dmag = Vpred[..., 0] - target[..., 0]
                dang = angle_diff(Vpred[..., 1], target[..., 1])
                mse_mag = torch.mean(dmag * dmag)
                mse_ang = torch.mean(dang * dang)
                mse = mse_mag + mse_ang
                phys = ppc_physics_loss(
                    Y_metric.to(torch.complex64),
                    Vpred,
                    Sstart_metric.to(torch.complex64),
                    bus_type,
                    form=args.physics_loss_form,
                    huber_delta=args.physics_huber_delta,
                )
                loss = args.mse_weight * mse + args.physics_weight * phys

                if train:
                    optim.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optim.step()

                residual_metrics = compute_power_flow_residual_metrics(
                    Y_metric,
                    Vpred.detach(),
                    Sstart_metric,
                    bus_type,
                    n_nodes_per_graph=n_nodes,
                    S_base=S_base_metric,
                )
                sum_loss += float(loss.item()) * B_eff
                sum_mse += float(mse.item()) * B_eff
                sum_mse_mag += float(mse_mag.item()) * B_eff
                sum_mse_ang += float(mse_ang.item()) * B_eff
                sum_phys += float(phys.item()) * B_eff
                sum_max_dp += residual_metrics["max_dp_pu"].sum().item()
                sum_max_dq += residual_metrics["max_dq_pu"].sum().item()
                if "max_dp_mva" in residual_metrics:
                    sum_max_dp_mva += residual_metrics["max_dp_mva"].sum().item()
                    sum_max_dq_mva += residual_metrics["max_dq_mva"].sum().item()
                dp_values.append(residual_metrics["dp_abs_valid"].cpu())
                dq_values.append(residual_metrics["dq_abs_valid"].cpu())
                conv = (
                    (residual_metrics["max_dp_pu"] < args.convergence_tol_pu)
                    & (residual_metrics["max_dq_pu"] < args.convergence_tol_pu)
                )
                n_conv += int(conv.sum().item())

        denom = max(n_graphs, 1)
        dist = residual_distribution(dp_values, dq_values, args.residual_tol_pu)
        dist["convergence_rate"] = n_conv / denom
        dist["n_converged"] = n_conv
        dist["n_cases"] = n_graphs
        return {
            "loss": sum_loss / denom,
            "mse": sum_mse / denom,
            "mse_mag": sum_mse_mag / denom,
            "mse_ang": sum_mse_ang / denom,
            "phys": sum_phys / denom,
            "max_dp_pu": sum_max_dp / denom,
            "max_dq_pu": sum_max_dq / denom,
            "max_dp_mva": sum_max_dp_mva / denom,
            "max_dq_mva": sum_max_dq_mva / denom,
            "dist": dist,
        }

    def fmt(prefix, m):
        rmse = math.sqrt(max(m["mse"], 0.0))
        rmse_mag = math.sqrt(max(m["mse_mag"], 0.0))
        rmse_ang_deg = math.sqrt(max(m["mse_ang"], 0.0)) * 180.0 / math.pi
        return (
            f"{prefix} loss {m['loss']:.4e} mse {m['mse']:.4e} phys {m['phys']:.4e} "
            f"rmse {rmse:.4e} (mag {rmse_mag:.4e}, ang {rmse_ang_deg:.4e}deg) "
            f"(dPinf {m['max_dp_pu']:.3e} pu, dQinf {m['max_dq_pu']:.3e} pu; "
            f"{m['max_dp_mva']:.3e} MW, {m['max_dq_mva']:.3e} MVAr) "
            f"{format_residual_distribution_compact(m['dist'])}"
        )

    print("Initial metrics before training:")
    if args.EPOCHS <= 0:
        tr0 = run_epoch(train_eval_loader, train=False)
        va0 = run_epoch(val_loader, train=False)
        te0 = run_epoch(test_loader, train=False)
        print("Epoch   0 | " + fmt("train", tr0))
        print("Epoch   0 | " + fmt("valid", va0))
        print("Epoch   0 | " + fmt("test", te0))
        return

    val0 = run_epoch(val_loader, train=False)
    print("Epoch   0 | " + fmt("valid", val0))

    for epoch in range(1, args.EPOCHS + 1):
        t0 = time.time()
        tr = run_epoch(train_loader, train=True)
        if epoch % args.VAL_EVERY == 0 or epoch == args.EPOCHS:
            va = run_epoch(val_loader, train=False)
            print(
                f"Epoch {epoch:3d} | {fmt('train', tr)} | {fmt('valid', va)} "
                f"| time {time.time() - t0:.2f}s"
            )
            if va["loss"] < best_val:
                best_val = va["loss"]
                torch.save(model.state_dict(), best_path)
                print(f"  checkpoint saved to {best_path}")
        else:
            print(f"Epoch {epoch:3d} | {fmt('train', tr)} | time {time.time() - t0:.2f}s")

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"[test] loaded best checkpoint: {best_path}")
    te = run_epoch(test_loader, train=False)
    rmse = math.sqrt(max(te["mse"], 0.0))
    rmse_mag = math.sqrt(max(te["mse_mag"], 0.0))
    rmse_ang_deg = math.sqrt(max(te["mse_ang"], 0.0)) * 180.0 / math.pi
    print(
        f"\nFinal test-set RMSE : {rmse:.4e}"
        f" (|V|: {rmse_mag:.4e}, theta: {rmse_ang_deg:.4e}deg)"
        f" | dPinf : {te['max_dp_pu']:.4e} pu ({te['max_dp_mva']:.4e} MW)"
        f" | dQinf : {te['max_dq_pu']:.4e} pu ({te['max_dq_mva']:.4e} MVAr)"
    )
    print(format_residual_distribution_compact(te["dist"]))


if __name__ == "__main__":
    run()

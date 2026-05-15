import os
import sys
import time
import math
import argparse
import atexit
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from GNSMsg_SelfAttention_armijo import GNSMsg_EdgeSelfAttn
from GNSMsg_armijo import GNSMsg   # adapt separately if you still want the non-attention baseline

from Dataset_optimized_complex_columns import ChanghunDataset
from collate_blockdiag_optimized_complex_columns import collate_blockdiag

from helper import MultiBucketBatchSampler, make_size_bucketing_loader

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.makedirs("./results/ckpt", exist_ok=True)
os.makedirs("./results/plots", exist_ok=True)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Training script for branch-row pandapower parquet format")

parser.add_argument("--PINN", action="store_true", help="Enable PINN")
parser.add_argument("--BLOCK_DIAG", action="store_true", help="Use block diagonal batching")
parser.add_argument("--NORMALIZE", action="store_true")
parser.add_argument("--PER_UNIT", action="store_true")
parser.add_argument("--float64", action="store_true")
parser.add_argument('--mode', type=str, default="train_test", help='train_valid_test | train | valid | test')
parser.add_argument("--mag_ang_mse", action="store_true", help="Use |V| + wrapped-angle reporting")

parser.add_argument('--model', type=str, default="GNSMsg_EdgeSelfAttn", help='GNSMsg_EdgeSelfAttn')
parser.add_argument("--d", type=int, default=4)
parser.add_argument("--d_hi", type=int, default=16)
parser.add_argument("--num_attn_layers", type=int, default=1)
parser.add_argument("--n_heads", type=int, default=4)

parser.add_argument("--K", type=int, default=40)
parser.add_argument('--gamma', type=float, default=0.9)
parser.add_argument("--use_armijo", action="store_true")
parser.add_argument("--armijo_mode", type=str, default="fixed", choices=("fixed", "geometric"))
parser.add_argument("--armijo_rho", type=float, default=0.5)
parser.add_argument("--armijo_c1", type=float, default=1e-4)
parser.add_argument("--armijo_max_backtracks", type=int, default=5)
parser.add_argument("--armijo_min_alpha", type=float, default=0.0625)
parser.add_argument("--vlimit", action="store_true")
parser.add_argument('--DthetaMax', type=float, default=0.3)
parser.add_argument('--DvmFrac', type=float, default=0.1)
parser.add_argument("--physics_loss_form", type=str, default="mse", choices=("mse", "huber", "logcosh"))
parser.add_argument("--physics_residual_norm", type=str, default="none", choices=("none", "setpoint", "graph"))
parser.add_argument("--physics_norm_eps", type=float, default=1e-6)
parser.add_argument("--physics_huber_delta", type=float, default=1.0)
parser.add_argument("--physics_final_weight", type=float, default=0.0)
parser.add_argument('--train_ratio', type=float, default=0.3333)
parser.add_argument('--valid_ratio', type=float, default=0.3333)

parser.add_argument('--weight_init', type=str, default="sd0.02")
parser.add_argument('--bias_init', type=float, default=0.0)
parser.add_argument('--weight_decay', type=float, default=1e-3)

parser.add_argument('--lr_scheduler', type=str, default="default", help='default | CosineAnnealingLR')
parser.add_argument('--cosineRestartEpoch', type=int, default=20)

parser.add_argument("--BATCH", type=int, default=16)
parser.add_argument("--EPOCHS", type=int, default=20)
parser.add_argument("--LR", type=float, default=1e-4)
parser.add_argument("--VAL_EVERY", type=int, default=1)
parser.add_argument("--residual_tol_pu", type=float, default=1e-6)

parser.add_argument("--PARQUET", type=str, nargs='+', required=True, help="Path to parquet data file(s)")
parser.add_argument("--seed_value", type=int, default=42)
parser.add_argument(
    "--no_cache_dense_ybus",
    "--no_cach_dense_ybus",
    action="store_true",
    help="Do not precompute/store dense Ybus in the dataset; reconstruct it inside the model instead",
)
parser.add_argument(
    "--lazy_parquet",
    action="store_true",
    help="Load parquet lazily by row group instead of materializing the full dataset in memory",
)
parser.add_argument(
    "--row_group_cache_size",
    type=int,
    default=2,
    help="How many decoded parquet row groups to keep in RAM when --lazy_parquet is enabled",
)

# NEW
parser.add_argument("--log_to_file", action="store_true", help="Save terminal output to a log file as well")
parser.add_argument("--log_dir", type=str, default="./results/logs", help="Directory for log file")

args = parser.parse_args()


# ------------------------------------------------------------------
# Effective configuration
# ------------------------------------------------------------------
SEED = args.seed_value

# Keep your old effective behavior
PINN = True
BLOCK_DIAG = True
NORMALIZE = False
PER_UNIT = True
args.mag_ang_mse = True

MODEL = args.model
BATCH = args.BATCH
EPOCHS = args.EPOCHS
LR = args.LR
VAL_EVERY = args.VAL_EVERY
PARQUET = args.PARQUET
d = args.d
d_hi = args.d_hi
n_heads = args.n_heads
K = args.K
GAMMA = args.gamma
VLIMIT = args.vlimit

torch.manual_seed(SEED)
np.random.seed(SEED)


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
class TeeLogger:
    def __init__(self, filename, stream):
        self.stream = stream
        self.log = open(filename, "a", buffering=1)  # line-buffered

    def write(self, message):
        self.stream.write(message)
        self.log.write(message)

    def flush(self):
        self.stream.flush()
        self.log.flush()

    def close(self):
        try:
            self.log.close()
        except Exception:
            pass


parquet_filenames = [os.path.splitext(os.path.basename(p))[0] for p in args.PARQUET]
shortened_names = ['_'.join(name.split('_')[:3]) for name in parquet_filenames]
parquet_filename = '_and_'.join(shortened_names)

armijo_tag = "True" if args.use_armijo else "False"
loss_tag = (
    f"_ploss{args.physics_loss_form}"
    f"_pnorm{args.physics_residual_norm}"
    f"_pfinal{args.physics_final_weight:g}"
)


RUNNAME = (
    f"{parquet_filename}_K{args.K}_d{args.d}_dhi{args.d_hi}"
    f"_nheads{args.n_heads}_numattn{args.num_attn_layers}"
    f"_armijo{armijo_tag}{loss_tag}_ep{args.EPOCHS}_TrainRatio{args.train_ratio}"
)
BEST_CKPT_PATH = f"./results/ckpt/{RUNNAME}_{EPOCHS}_best_model.ckpt"
if args.log_to_file:
    os.makedirs(args.log_dir, exist_ok=True)
    log_filename = os.path.join(args.log_dir, f"{RUNNAME}_training_log.txt")

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    sys.stdout = TeeLogger(log_filename, original_stdout)
    sys.stderr = TeeLogger(log_filename, original_stderr)

    def _cleanup_logger():
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        try:
            sys.stdout.close()
        except Exception:
            pass
        try:
            sys.stderr.close()
        except Exception:
            pass

    atexit.register(_cleanup_logger)

    print(f"[logging] stdout/stderr will also be saved to: {log_filename}")

print(
    f"MODEL:{MODEL}, PINN:{PINN}, Block:{BLOCK_DIAG}, d:{d}, d_hi:{d_hi}, n_heads:{n_heads}, "
    f"K:{K}, Runname:{RUNNAME}, PARQUET:{PARQUET}, BATCH:{BATCH}, EP:{EPOCHS}, LR:{LR}, "
    f"no_cache_dense_ybus:{args.no_cache_dense_ybus}, lazy_parquet:{args.lazy_parquet}, "
    f"row_group_cache_size:{args.row_group_cache_size}, "
    f"physics_loss_form:{args.physics_loss_form}, physics_residual_norm:{args.physics_residual_norm}, "
    f"physics_final_weight:{args.physics_final_weight}, DthetaMax:{args.DthetaMax}, DvmFrac:{args.DvmFrac}"
)


# ------------------------------------------------------------------
# Device
# ------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ------------------------------------------------------------------
# Dataset / split
# ------------------------------------------------------------------
full_ds = ChanghunDataset(
    PARQUET,
    per_unit=PER_UNIT,
    device=None,
    no_cache_dense_ybus=args.no_cache_dense_ybus,
    lazy_row_groups=args.lazy_parquet,
    row_group_cache_size=args.row_group_cache_size,
)

n_total = len(full_ds)
n_train = int(args.train_ratio * n_total)
n_val = int(args.valid_ratio * n_total)
n_test = n_total - n_train - n_val

train_ds, val_ds, test_ds = random_split(
    full_ds,
    lengths=[n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(SEED)
)

if BATCH == 1:
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False)

else:
    if BLOCK_DIAG:
        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH,
            shuffle=True,
            collate_fn=collate_blockdiag
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=BATCH,
            shuffle=False,
            collate_fn=collate_blockdiag
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=BATCH,
            shuffle=False,
            collate_fn=collate_blockdiag
        )

    else:
        # Non-blockdiag batching requires homogeneous tensor shapes.
        # With the new parquet metadata that means at least same (N, nl).
        train_signatures = [
            full_ds.get_signature(i) if hasattr(full_ds, "get_signature") else (full_ds[i]["N"], full_ds[i]["nl"])
            for i in train_ds.indices
        ]
        val_signatures = [
            full_ds.get_signature(i) if hasattr(full_ds, "get_signature") else (full_ds[i]["N"], full_ds[i]["nl"])
            for i in val_ds.indices
        ]
        test_signatures = [
            full_ds.get_signature(i) if hasattr(full_ds, "get_signature") else (full_ds[i]["N"], full_ds[i]["nl"])
            for i in test_ds.indices
        ]

        train_sampler = MultiBucketBatchSampler(
            signatures=train_signatures,
            batch_size=BATCH,
            shuffle=True,
            drop_last=True,
        )
        val_sampler = MultiBucketBatchSampler(
            signatures=val_signatures,
            batch_size=BATCH,
            shuffle=False,
            drop_last=True,
        )
        test_sampler = MultiBucketBatchSampler(
            signatures=test_signatures,
            batch_size=BATCH,
            shuffle=False,
            drop_last=True,
        )

        train_loader = DataLoader(train_ds, batch_sampler=train_sampler)
        val_loader   = DataLoader(val_ds,   batch_sampler=val_sampler)
        test_loader  = DataLoader(test_ds,  batch_sampler=test_sampler)

print(f"Dataset sizes | train {n_train}   valid {n_val}   test {n_test}")

# ------------------------------------------------------------------
# Model / optimizer / loss
# ------------------------------------------------------------------
if args.model == "GNSMsg":
    model = GNSMsg(
        d=d,
        d_hi=d_hi,
        K=K,
        pinn=PINN,
        gamma=GAMMA,
        v_limit=VLIMIT,
        use_armijo=args.use_armijo
    ).to(device)

elif args.model == "GNSMsg_EdgeSelfAttn":
    model = GNSMsg_EdgeSelfAttn(
        d=d,
        d_hi=d_hi,
        n_heads=n_heads,
        K=K,
        pinn=PINN,
        gamma=GAMMA,
        v_limit=VLIMIT,
        use_armijo=args.use_armijo,
        num_attn_layers=args.num_attn_layers,
        armijo_mode=args.armijo_mode,
        armijo_rho=args.armijo_rho,
        armijo_c1=args.armijo_c1,
        armijo_max_backtracks=args.armijo_max_backtracks,
        armijo_min_alpha=args.armijo_min_alpha,
        dtheta_max=args.DthetaMax,
        dvm_frac=args.DvmFrac,
        physics_loss_form=args.physics_loss_form,
        physics_residual_norm=args.physics_residual_norm,
        physics_norm_eps=args.physics_norm_eps,
        physics_huber_delta=args.physics_huber_delta,
        physics_final_weight=args.physics_final_weight,
    ).to(device)

else:
    raise ValueError(f"Unknown model: {args.model}")

def init_weights(model, exclude_modules):
    for module in model.modules():
        if module in exclude_modules:
            continue
        if isinstance(module, nn.Linear):
            if args.weight_init == "sd0.02":
                torch.nn.init.normal_(module.weight, mean=0, std=0.02)
            elif args.weight_init == "He":
                torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                module.bias.data.fill_(args.bias_init)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
        else:
            for name, param in module.named_parameters():
                if 'weight' in name and param.dim() > 1:
                    if args.weight_init == "sd0.02":
                        torch.nn.init.normal_(param, mean=0, std=0.02)
                    elif args.weight_init == "He":
                        torch.nn.init.kaiming_uniform_(param, nonlinearity='relu')
                elif 'bias' in name:
                    param.data.fill_(args.bias_init)

exclude_modules = []
init_weights(model, exclude_modules)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total number of parameters: {count_parameters(model)}")

if args.lr_scheduler == "CosineAnnealingLR":
    steps_per_epoch = len(train_loader)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=args.weight_decay)
    T_0 = args.cosineRestartEpoch * steps_per_epoch
    scheduler = CosineAnnealingWarmRestarts(optim, T_0=T_0, T_mult=1, eta_min=1e-6)
else:
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=args.weight_decay)
    scheduler = None


def _real_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.complex64:
        return torch.float32
    if dtype == torch.complex128:
        return torch.float64
    return dtype


def build_dense_y_from_branchrows_single(
    N,
    Branch_f_bus,
    Branch_t_bus,
    Branch_status,
    Branch_tau,
    Branch_shift_deg,
    Branch_y_series_from,
    Branch_y_series_to,
    Branch_y_series_ft,
    Branch_y_shunt_from,
    Branch_y_shunt_to,
    Y_shunt_bus,
):
    device = Branch_f_bus.device
    dtype = Branch_y_series_ft.dtype

    Y = torch.zeros(N, N, dtype=dtype, device=device)
    Y.diagonal().add_(Y_shunt_bus.to(dtype))

    mask = (Branch_status != 0)
    if mask.sum() == 0:
        return Y

    f = Branch_f_bus[mask].long()
    t = Branch_t_bus[mask].long()

    real_dtype = _real_dtype(dtype)
    tau = Branch_tau[mask].to(real_dtype)
    theta = torch.deg2rad(Branch_shift_deg[mask].to(real_dtype))
    a = tau.to(dtype) * torch.exp(1j * theta.to(dtype))

    y_from = Branch_y_series_from[mask].to(dtype)
    y_to = Branch_y_series_to[mask].to(dtype)
    ysh_f = Branch_y_shunt_from[mask].to(dtype)
    ysh_t = Branch_y_shunt_to[mask].to(dtype)

    Yff = (y_from + ysh_f / 2.0) / (a * torch.conj(a))
    Ytt = (y_to + ysh_t / 2.0)
    Yft = -y_from / torch.conj(a)
    Ytf = -y_to / a

    Y.index_put_((f, f), Yff, accumulate=True)
    Y.index_put_((t, t), Ytt, accumulate=True)
    Y.index_put_((f, t), Yft, accumulate=True)
    Y.index_put_((t, f), Ytf, accumulate=True)

    return Y


def ensure_dense_y_for_metrics(
    Y,
    bus_type,
    Branch_f_bus,
    Branch_t_bus,
    Branch_status,
    Branch_tau,
    Branch_shift_deg,
    Branch_y_series_from,
    Branch_y_series_to,
    Branch_y_series_ft,
    Branch_y_shunt_from,
    Branch_y_shunt_to,
    Y_shunt_bus,
):
    if Y is not None:
        return Y.unsqueeze(0) if Y.dim() == 2 else Y

    if Y_shunt_bus is None:
        raise ValueError("Y is None and Y_shunt_bus is None; cannot reconstruct Y for residual metrics.")

    B, N = bus_type.shape
    if B == 1:
        return build_dense_y_from_branchrows_single(
            N,
            Branch_f_bus.squeeze(0),
            Branch_t_bus.squeeze(0),
            Branch_status.squeeze(0),
            Branch_tau.squeeze(0),
            Branch_shift_deg.squeeze(0),
            Branch_y_series_from.squeeze(0),
            Branch_y_series_to.squeeze(0),
            Branch_y_series_ft.squeeze(0),
            Branch_y_shunt_from.squeeze(0),
            Branch_y_shunt_to.squeeze(0),
            Y_shunt_bus.squeeze(0),
        ).unsqueeze(0)

    Ys = []
    for b in range(B):
        Ys.append(build_dense_y_from_branchrows_single(
            N,
            Branch_f_bus[b],
            Branch_t_bus[b],
            Branch_status[b],
            Branch_tau[b],
            Branch_shift_deg[b],
            Branch_y_series_from[b],
            Branch_y_series_to[b],
            Branch_y_series_ft[b],
            Branch_y_shunt_from[b],
            Branch_y_shunt_to[b],
            Y_shunt_bus[b],
        ))
    return torch.stack(Ys, dim=0)


def compute_power_flow_residual_metrics(Y, Vpred, Sset, bus_type, *, n_nodes_per_graph=None, S_base=None):
    v = Vpred[..., 0]
    th = Vpred[..., 1]

    if Y.dim() == 2:
        Y = Y.unsqueeze(0)

    Vc = v * torch.exp(1j * th)
    Ic = torch.matmul(Y, Vc.unsqueeze(-1)).squeeze(-1)
    Sc = Vc * Ic.conj()

    P_set, Q_set = Sset.real, Sset.imag
    slack_mask = (bus_type == 1)
    pv_mask = (bus_type == 2)

    dp_abs = (P_set - Sc.real).abs()
    dq_abs = (Q_set - Sc.imag).abs()
    p_mask = ~slack_mask
    q_mask = ~(slack_mask | pv_mask)

    if n_nodes_per_graph is not None:
        max_dp_pu = []
        max_dq_pu = []
        offset = 0
        for size in n_nodes_per_graph.tolist():
            size = int(size)
            sl = slice(offset, offset + size)

            dp_g = dp_abs[0, sl]
            dq_g = dq_abs[0, sl]
            p_mask_g = p_mask[0, sl]
            q_mask_g = q_mask[0, sl]

            if p_mask_g.any():
                max_dp_pu.append(dp_g[p_mask_g].max())
            else:
                max_dp_pu.append(dp_g.new_zeros(()))

            if q_mask_g.any():
                max_dq_pu.append(dq_g[q_mask_g].max())
            else:
                max_dq_pu.append(dq_g.new_zeros(()))

            offset += size

        max_dp_pu = torch.stack(max_dp_pu)
        max_dq_pu = torch.stack(max_dq_pu)
    else:
        max_dp_pu = dp_abs.masked_fill(~p_mask, 0.0).amax(dim=-1)
        max_dq_pu = dq_abs.masked_fill(~q_mask, 0.0).amax(dim=-1)
        max_dq_pu = torch.where(q_mask.any(dim=-1), max_dq_pu, torch.zeros_like(max_dq_pu))

    metrics = {
        "max_dp_pu": max_dp_pu,
        "max_dq_pu": max_dq_pu,
        "dp_abs_valid": dp_abs[p_mask].detach(),
        "dq_abs_valid": dq_abs[q_mask].detach(),
    }

    if S_base is not None:
        S_base = S_base.to(max_dp_pu.device, dtype=max_dp_pu.dtype).reshape(-1)
        if S_base.numel() == 1 and max_dp_pu.numel() != 1:
            S_base = S_base.expand(max_dp_pu.numel())
        metrics["max_dp_mva"] = max_dp_pu * S_base
        metrics["max_dq_mva"] = max_dq_pu * S_base

    return metrics


def format_residual_summary(max_dp_pu, max_dq_pu):
    return f"(ΔP∞ {max_dp_pu:.3e} pu, ΔQ∞ {max_dq_pu:.3e} pu)"


def _empty_residual_distribution():
    return {
        "mean_dp_pu": 0.0,
        "mean_dq_pu": 0.0,
        "median_dp_pu": 0.0,
        "median_dq_pu": 0.0,
        "p95_dp_pu": 0.0,
        "p95_dq_pu": 0.0,
        "p99_dp_pu": 0.0,
        "p99_dq_pu": 0.0,
        "rmse_dp_pu": 0.0,
        "rmse_dq_pu": 0.0,
        "frac_dp_below_tol": 0.0,
        "frac_dq_below_tol": 0.0,
        "n_dp": 0,
        "n_dq": 0,
    }


def finalize_residual_distribution(dp_values, dq_values, *, tol_pu):
    dist = _empty_residual_distribution()

    if dp_values:
        dp = torch.cat(dp_values).float()
        dist.update({
            "mean_dp_pu": dp.mean().item(),
            "median_dp_pu": dp.median().item(),
            "p95_dp_pu": torch.quantile(dp, 0.95).item(),
            "p99_dp_pu": torch.quantile(dp, 0.99).item(),
            "rmse_dp_pu": torch.sqrt((dp ** 2).mean()).item(),
            "frac_dp_below_tol": (dp <= tol_pu).float().mean().item(),
            "n_dp": int(dp.numel()),
        })

    if dq_values:
        dq = torch.cat(dq_values).float()
        dist.update({
            "mean_dq_pu": dq.mean().item(),
            "median_dq_pu": dq.median().item(),
            "p95_dq_pu": torch.quantile(dq, 0.95).item(),
            "p99_dq_pu": torch.quantile(dq, 0.99).item(),
            "rmse_dq_pu": torch.sqrt((dq ** 2).mean()).item(),
            "frac_dq_below_tol": (dq <= tol_pu).float().mean().item(),
            "n_dq": int(dq.numel()),
        })

    return dist


def format_residual_distribution_compact(dist):
    return (
        f"(mean |ΔP| {dist['mean_dp_pu']:.3e}, |ΔQ| {dist['mean_dq_pu']:.3e} pu; "
        f"p95 |ΔP| {dist['p95_dp_pu']:.3e}, |ΔQ| {dist['p95_dq_pu']:.3e} pu; "
        f"tol≤ {dist['frac_dp_below_tol']:.2%} P, {dist['frac_dq_below_tol']:.2%} Q)"
    )


def format_residual_distribution_full(dist, *, tol_pu):
    return (
        f"Residual distribution over PV+PQ/PQ buses (tol={tol_pu:.1e} pu):\n"
        f"  mean   |ΔP| {dist['mean_dp_pu']:.4e} pu | |ΔQ| {dist['mean_dq_pu']:.4e} pu\n"
        f"  median |ΔP| {dist['median_dp_pu']:.4e} pu | |ΔQ| {dist['median_dq_pu']:.4e} pu\n"
        f"  p95    |ΔP| {dist['p95_dp_pu']:.4e} pu | |ΔQ| {dist['p95_dq_pu']:.4e} pu\n"
        f"  p99    |ΔP| {dist['p99_dp_pu']:.4e} pu | |ΔQ| {dist['p99_dq_pu']:.4e} pu\n"
        f"  RMSE   ΔP   {dist['rmse_dp_pu']:.4e} pu | ΔQ   {dist['rmse_dq_pu']:.4e} pu\n"
        f"  frac below tol: P {dist['frac_dp_below_tol']:.2%} ({dist['n_dp']} entries), "
        f"Q {dist['frac_dq_below_tol']:.2%} ({dist['n_dq']} entries)"
    )


# ------------------------------------------------------------------
# Epoch runner
# ------------------------------------------------------------------
def run_epoch(loader, *, train: bool, pinn: bool):
    model.train() if train else model.eval()

    sum_loss = 0.0
    sum_mse = 0.0
    sum_mse_mag = 0.0
    sum_mse_ang = 0.0
    sum_max_dp_pu = 0.0
    sum_max_dq_pu = 0.0
    sum_max_dp_mva = 0.0
    sum_max_dq_mva = 0.0
    dp_dist_values = []
    dq_dist_values = []
    n_graphs_total = 0

    with torch.set_grad_enabled(train):
        for batch in loader:
            if BLOCK_DIAG and "sizes" in batch:
                B_eff = int(batch["sizes"].numel())
            else:
                B_eff = int(batch["bus_type"].size(0))
            n_graphs_total += B_eff

            if BLOCK_DIAG:
                n_nodes_per_graph = batch["sizes"].to(device)
            else:
                n_nodes_per_graph = None

            bus_type = batch["bus_type"].to(device)

            Branch_f_bus = batch["Branch_f_bus"].to(device)
            Branch_t_bus = batch["Branch_t_bus"].to(device)
            Branch_status = batch["Branch_status"].to(device)
            Branch_tau = batch["Branch_tau"].to(device)
            Branch_shift_deg = batch["Branch_shift_deg"].to(device)

            Branch_y_series_from = batch["Branch_y_series_from"].to(device)
            Branch_y_series_to   = batch["Branch_y_series_to"].to(device)
            Branch_y_series_ft   = batch["Branch_y_series_ft"].to(device)

            Branch_y_shunt_from = batch["Branch_y_shunt_from"].to(device)
            Branch_y_shunt_to   = batch["Branch_y_shunt_to"].to(device)

            Is_trafo = batch["Is_trafo"].to(device)
            Y_shunt_bus = batch["Y_shunt_bus"].to(device)

            Y = batch.get("Ybus", None)
            if Y is not None:
                Y = Y.to(device)
            S_base = batch.get("S_base", None)
            if S_base is not None:
                S_base = S_base.to(device)

            Sstart = batch["S_start"].to(device)
            Ustart = batch["U_start"].to(device)
            Vstart = batch["V_start"].to(device)
            Vnewton = batch["V_newton"].to(device)

            if pinn:
                Vpred, loss_phys = model(
                    bus_type,
                    Branch_f_bus, Branch_t_bus, Branch_status,
                    Branch_tau, Branch_shift_deg,
                    Branch_y_series_from, Branch_y_series_to, Branch_y_series_ft,
                    Branch_y_shunt_from, Branch_y_shunt_to,
                    Is_trafo,
                    Y,
                    Sstart,
                    Vstart,
                    n_nodes_per_graph=n_nodes_per_graph,
                    Y_shunt_bus=Y_shunt_bus,
                )

                dmag = (Vpred[..., 0] - Vnewton[..., 0])
                dang = torch.atan2(
                    torch.sin(Vpred[..., 1] - Vnewton[..., 1]),
                    torch.cos(Vpred[..., 1] - Vnewton[..., 1])
                )
                mse_mag = torch.mean(dmag ** 2)
                mse_ang = torch.mean(dang ** 2)
                mse = mse_mag + mse_ang
                loss = loss_phys

                if train and not loss.requires_grad:
                    p0 = next(model.parameters())
                    loss = loss + 0.0 * p0.norm()
                    print("[warn] physics loss detached for this batch; applied zero-grad guard.")
            else:
                Vpred = model(
                    bus_type,
                    Branch_f_bus, Branch_t_bus, Branch_status,
                    Branch_tau, Branch_shift_deg,
                    Branch_y_series_from, Branch_y_series_to, Branch_y_series_ft,
                    Branch_y_shunt_from, Branch_y_shunt_to,
                    Is_trafo,
                    Y,
                    Sstart,
                    Vstart,
                    n_nodes_per_graph=n_nodes_per_graph,
                    Y_shunt_bus=Y_shunt_bus,
                )

                dmag = (Vpred[..., 0] - Vnewton[..., 0])
                dang = torch.atan2(
                    torch.sin(Vpred[..., 1] - Vnewton[..., 1]),
                    torch.cos(Vpred[..., 1] - Vnewton[..., 1])
                )
                mse_mag = torch.mean(dmag ** 2)
                mse_ang = torch.mean(dang ** 2)
                mse = mse_mag + mse_ang
                loss = mse

            residual_metrics = compute_power_flow_residual_metrics(
                ensure_dense_y_for_metrics(
                    Y,
                    bus_type,
                    Branch_f_bus,
                    Branch_t_bus,
                    Branch_status,
                    Branch_tau,
                    Branch_shift_deg,
                    Branch_y_series_from,
                    Branch_y_series_to,
                    Branch_y_series_ft,
                    Branch_y_shunt_from,
                    Branch_y_shunt_to,
                    Y_shunt_bus,
                ),
                Vpred,
                Sstart,
                bus_type,
                n_nodes_per_graph=n_nodes_per_graph,
                S_base=S_base,
            )

            if train:
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optim.step()
                if scheduler is not None:
                    scheduler.step()

            sum_loss += loss.item() * B_eff
            sum_mse += mse.item() * B_eff
            sum_mse_mag += mse_mag.item() * B_eff
            sum_mse_ang += mse_ang.item() * B_eff
            sum_max_dp_pu += residual_metrics["max_dp_pu"].sum().item()
            sum_max_dq_pu += residual_metrics["max_dq_pu"].sum().item()
            dp_dist_values.append(residual_metrics["dp_abs_valid"].detach().cpu())
            dq_dist_values.append(residual_metrics["dq_abs_valid"].detach().cpu())
            if "max_dp_mva" in residual_metrics:
                sum_max_dp_mva += residual_metrics["max_dp_mva"].sum().item()
                sum_max_dq_mva += residual_metrics["max_dq_mva"].sum().item()

    mean_loss = sum_loss / max(n_graphs_total, 1)
    mean_mse = sum_mse / max(n_graphs_total, 1)
    mean_mse_mag = sum_mse_mag / max(n_graphs_total, 1)
    mean_mse_ang = sum_mse_ang / max(n_graphs_total, 1)
    mean_max_dp_pu = sum_max_dp_pu / max(n_graphs_total, 1)
    mean_max_dq_pu = sum_max_dq_pu / max(n_graphs_total, 1)
    mean_max_dp_mva = sum_max_dp_mva / max(n_graphs_total, 1)
    mean_max_dq_mva = sum_max_dq_mva / max(n_graphs_total, 1)
    residual_dist = finalize_residual_distribution(
        dp_dist_values,
        dq_dist_values,
        tol_pu=args.residual_tol_pu,
    )
    return (
        mean_loss,
        mean_mse,
        mean_mse_mag,
        mean_mse_ang,
        mean_max_dp_pu,
        mean_max_dq_pu,
        mean_max_dp_mva,
        mean_max_dq_mva,
        residual_dist,
    )


# ------------------------------------------------------------------
# Training / validation
# ------------------------------------------------------------------
if "train" in args.mode:
    train_loss_hist, train_rmse_hist = [], []
    train_rmse_mag_hist, train_rmse_ang_hist_deg = [], []

    val_loss_hist, val_rmse_hist = [], []
    val_rmse_mag_hist, val_rmse_ang_hist_deg = [], []

    best_val_loss = float('inf')

    print("Initial metrics before training:")
    (
        train_loss,
        train_mse,
        train_mse_mag,
        train_mse_ang,
        train_max_dp_pu,
        train_max_dq_pu,
        _train_max_dp_mva,
        _train_max_dq_mva,
        _train_residual_dist,
    ) = run_epoch(train_loader, train=False, pinn=PINN)
    train_rmse = math.sqrt(train_mse)
    train_rmse_mag = math.sqrt(train_mse_mag)
    train_rmse_ang_deg = math.sqrt(train_mse_ang) * (180.0 / math.pi)

    (
        val_loss,
        val_mse,
        val_mse_mag,
        val_mse_ang,
        val_max_dp_pu,
        val_max_dq_pu,
        _val_max_dp_mva,
        _val_max_dq_mva,
        val_residual_dist,
    ) = run_epoch(val_loader, train=False, pinn=PINN)
    val_rmse = math.sqrt(val_mse)
    val_rmse_mag = math.sqrt(val_mse_mag)
    val_rmse_ang_deg = math.sqrt(val_mse_ang) * (180.0 / math.pi)

    print(
        f"Epoch   0 | "
        f"train loss {train_loss:.4e}  rmse {train_rmse:.4e} "
        f"(mag {train_rmse_mag:.4e}, ang {train_rmse_ang_deg:.4e}°) "
        f"{format_residual_summary(train_max_dp_pu, train_max_dq_pu)} | "
        f"valid loss {val_loss:.4e}  rmse {val_rmse:.4e} "
        f"(mag {val_rmse_mag:.4e}, ang {val_rmse_ang_deg:.4e}°) "
        f"{format_residual_summary(val_max_dp_pu, val_max_dq_pu)} "
        f"{format_residual_distribution_compact(val_residual_dist)}"
    )

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        (
            train_loss,
            train_mse,
            train_mse_mag,
            train_mse_ang,
            train_max_dp_pu,
            train_max_dq_pu,
            _train_max_dp_mva,
            _train_max_dq_mva,
            _train_residual_dist,
        ) = run_epoch(train_loader, train=True, pinn=PINN)
        train_rmse = math.sqrt(train_mse)
        train_rmse_mag = math.sqrt(train_mse_mag)
        train_rmse_ang_deg = math.sqrt(train_mse_ang) * (180.0 / math.pi)

        train_loss_hist.append(train_loss)
        train_rmse_hist.append(train_rmse)
        train_rmse_mag_hist.append(train_rmse_mag)
        train_rmse_ang_hist_deg.append(train_rmse_ang_deg)

        if epoch % VAL_EVERY == 0 or epoch == EPOCHS:
            (
                val_loss,
                val_mse,
                val_mse_mag,
                val_mse_ang,
                val_max_dp_pu,
                val_max_dq_pu,
                _val_max_dp_mva,
                _val_max_dq_mva,
                val_residual_dist,
            ) = run_epoch(val_loader, train=False, pinn=PINN)
            val_rmse = math.sqrt(val_mse)
            val_rmse_mag = math.sqrt(val_mse_mag)
            val_rmse_ang_deg = math.sqrt(val_mse_ang) * (180.0 / math.pi)

            val_loss_hist.append(val_loss)
            val_rmse_hist.append(val_rmse)
            val_rmse_mag_hist.append(val_rmse_mag)
            val_rmse_ang_hist_deg.append(val_rmse_ang_deg)

            print(
                f"Epoch {epoch:3d} | "
                f"train loss {train_loss:.4e}  rmse {train_rmse:.4e} "
                f"(mag {train_rmse_mag:.4e}, ang {train_rmse_ang_deg:.4e}°) "
                f"{format_residual_summary(train_max_dp_pu, train_max_dq_pu)} | "
                f"valid loss {val_loss:.4e}  rmse {val_rmse:.4e} "
                f"(mag {val_rmse_mag:.4e}, ang {val_rmse_ang_deg:.4e}°) "
                f"{format_residual_summary(val_max_dp_pu, val_max_dq_pu)} | "
                f"{format_residual_distribution_compact(val_residual_dist)} | "
                f"time {time.time() - t0:.2f}s"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), BEST_CKPT_PATH)
                print(f"  ↳ checkpoint saved to {BEST_CKPT_PATH}")
        else:
            print(
                f"Epoch {epoch:3d} | "
                f"train loss {train_loss:.4e}  rmse {train_rmse:.4e} "
                f"(mag {train_rmse_mag:.4e}, ang {train_rmse_ang_deg:.4e}°) "
                f"{format_residual_summary(train_max_dp_pu, train_max_dq_pu)} | "
                f"time {time.time() - t0:.2f}s"
            )

    import matplotlib.pyplot as plt

    epochs = range(1, len(train_loss_hist) + 1)

    if PINN:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, train_loss_hist, label="Train Physics Loss")
        plt.plot(epochs[:len(val_loss_hist)], val_loss_hist, label="Validation Physics Loss")
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("PINN: Physics Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"./results/plots/{RUNNAME}_physics_loss.png")
        plt.clf()

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_rmse_hist, label="Train RMSE")
    plt.plot(epochs[:len(val_rmse_hist)], val_rmse_hist, label="Val RMSE")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("Supervised RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./results/plots/{RUNNAME}_rmse_total.png")
    plt.clf()

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(epochs, train_rmse_mag_hist, label="Train |V|")
    ax[0].plot(epochs[:len(val_rmse_mag_hist)], val_rmse_mag_hist, label="Val |V|")
    ax[0].set_title("Magnitude RMSE")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("Epoch")
    ax[0].legend()

    ax[1].plot(epochs, train_rmse_ang_hist_deg, label="Train θ (deg)")
    ax[1].plot(epochs[:len(val_rmse_ang_hist_deg)], val_rmse_ang_hist_deg, label="Val θ (deg)")
    ax[1].set_title("Angle RMSE (degrees)")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("Epoch")
    ax[1].legend()

    fig.suptitle("Magnitude vs Angle RMSE")
    fig.tight_layout()
    fig.savefig(f"./results/plots/{RUNNAME}_rmse_components.png")
    plt.close(fig)


# ------------------------------------------------------------------
# Final test
# ------------------------------------------------------------------
if "test" in args.mode:
    if os.path.exists(BEST_CKPT_PATH):
        model.load_state_dict(torch.load(BEST_CKPT_PATH, map_location=device))
        print(f"[test] loaded best checkpoint: {BEST_CKPT_PATH}")
    else:
        print(f"[test] best checkpoint not found at {BEST_CKPT_PATH}; using current model weights.")

    (
        test_loss,
        test_mse,
        test_mse_mag,
        test_mse_ang,
        test_max_dp_pu,
        test_max_dq_pu,
        test_max_dp_mva,
        test_max_dq_mva,
        test_residual_dist,
    ) = run_epoch(test_loader, train=False, pinn=PINN)

    test_rmse = math.sqrt(test_mse)
    test_rmse_mag = math.sqrt(test_mse_mag)
    test_rmse_ang_deg = math.sqrt(test_mse_ang) * (180.0 / math.pi)

    if PINN:
        print(
            f"\nTest physics-loss : {test_loss:.4e}"
            f" | total RMSE : {test_rmse:.4e}"
            f" | |V| RMSE : {test_rmse_mag:.4e}"
            f" | θ RMSE : {test_rmse_ang_deg:.4e}°"
            f" | ΔP∞ : {test_max_dp_pu:.4e} pu ({test_max_dp_mva:.4e} MW)"
            f" | ΔQ∞ : {test_max_dq_pu:.4e} pu ({test_max_dq_mva:.4e} MVAr)"
        )
        print(format_residual_distribution_full(test_residual_dist, tol_pu=args.residual_tol_pu))
    else:
        print(
            f"\nFinal test-set RMSE : {test_rmse:.4e}"
            f" (|V|: {test_rmse_mag:.4e}, θ: {test_rmse_ang_deg:.4e}°)"
        )

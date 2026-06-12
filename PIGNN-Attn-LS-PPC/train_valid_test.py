import os
import sys
import time
import math
import argparse
import atexit
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from GNSMsg_SelfAttention_armijo import GNSMsg_EdgeSelfAttn
from GNSMsg_SelfAttention_armijo_khop import GNSMsg_EdgeSelfAttnKHop
from GNSMsg_armijo import GNSMsg   # adapt separately if you still want the non-attention baseline

from Dataset_optimized_complex_columns import ChanghunDataset
from collate_blockdiag_optimized_complex_columns import collate_blockdiag

from helper import MultiBucketBatchSampler, make_size_bucketing_loader

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.makedirs("./results/plots", exist_ok=True)
# Note: checkpoint dir is created later, after --ckpt_dir is parsed.


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Training script for branch-row pandapower parquet format")

parser.add_argument(
    "--ckpt_dir",
    type=str,
    default="./results/ckpt",
    help=(
        "Directory to write best-model checkpoints into. Default keeps the "
        "historical ./results/ckpt path (relative to working dir). For Alex "
        "runs prefer an absolute path under /home/vault (TB-scale quota), "
        "e.g. --ckpt_dir /home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/ckpt/<run_group>"
    ),
)
parser.add_argument("--PINN", action="store_true", help="Enable PINN")
parser.add_argument("--BLOCK_DIAG", action="store_true", help="Use block diagonal batching")
parser.add_argument("--NORMALIZE", action="store_true")
parser.add_argument("--PER_UNIT", action="store_true")
parser.add_argument(
    "--target_S_base",
    type=float,
    default=None,
    help=(
        "Optional S_base (VA) for per-unit rebasing. If unset, keep the "
        "parquet's S_base as-is. Pass e.g. --target_S_base 1e8 to force "
        "100 MVA (normalizes LVN-style 1 MVA distribution data into the "
        "IEEE/PEGASE scale). Only effective when --PER_UNIT is set."
    ),
)
parser.add_argument(
    "--dataset_complex_dtype",
    type=str,
    default="complex64",
    choices=("complex64", "complex128"),
    help=(
        "Complex dtype used by ChanghunDataset when decoding raw parquet "
        "admittance/voltage/power fields. Use complex128 for LVN validation/"
        "test residual diagnostics; complex64 is the memory-efficient default "
        "for training."
    ),
)
parser.add_argument(
    "--share_ybus",
    action="store_true",
    help=(
        "Build the dense Ybus tensor from row 0 once and reuse it across "
        "every row in the dataset. Safe when all rows describe the SAME "
        "grid sampled under different load/generation perturbations -- which "
        "is the case for LVN and the IEEE perturbation parquets. Skips "
        "per-row Y reconstruction (~722^2 complex matmul) and removes the "
        "single biggest source of lazy-parquet disk thrashing."
    ),
)
parser.add_argument(
    "--share_grid",
    action="store_true",
    help=(
        "Stronger than --share_ybus: cache ALL grid-only tensors "
        "(Branch_y_*, vn_kv, bus_type, Y_shunt_bus, Y_Lines, Y_C_Lines, "
        "topology, transformer flags) from row 0 and reuse for every row. "
        "Implies --share_ybus. Skips parquet decode + tensor allocation + "
        "per-unit math for ~20 grid-only fields. Use when every row is the "
        "same grid under different load/generation perturbations."
    ),
)
parser.add_argument(
    "--vn_feat",
    action="store_true",
    help=(
        "Add per-bus vn_log (log10 of rated kV) as an extra bus feature. "
        "Enables bus_feat_extra_dim=1 so the model can distinguish voltage "
        "classes (3/20/110/380 kV) in multi-voltage grids like LVN Heo1. "
        "Without this flag all buses look identical after per-unit "
        "normalization (V=1.0 plateau pathology)."
    ),
)
parser.add_argument("--float64", action="store_true")
parser.add_argument('--mode', type=str, default="train_test", help='train_valid_test | train | valid | test')
parser.add_argument("--mag_ang_mse", action="store_true", help="Use |V| + wrapped-angle reporting")
parser.add_argument(
    "--mse_weight",
    type=float,
    default=0.0,
    help=(
        "Weight w for combined loss: L = L_phys + w * L_MSE. "
        "Default 0.0 = pure PINN (original behaviour). "
        "Set e.g. --mse_weight 10.0 to replicate the student's combined loss "
        "that stabilizes training on stiff/multi-voltage grids like LVN Heo1 "
        "where pure PINN converges to a non-V_newton solution. "
        "Only effective when --PINN is set."
    ),
)

parser.add_argument(
    '--model',
    type=str,
    default="GNSMsg_EdgeSelfAttn",
    help='GNSMsg_EdgeSelfAttn | GNSMsg_EdgeSelfAttnKHop',
)
parser.add_argument("--d", type=int, default=4)
parser.add_argument("--d_hi", type=int, default=16)
parser.add_argument("--num_attn_layers", type=int, default=1)
parser.add_argument("--n_heads", type=int, default=4)
parser.add_argument("--khop_K", type=int, default=3)
parser.add_argument("--khop_sigma", type=float, default=1.5)
parser.add_argument("--khop_norm", type=str, default="row", choices=("row", "sym", "col"))
parser.add_argument("--khop_source", type=str, default="yabs", choices=("yabs", "adj"))

parser.add_argument("--K", type=int, default=40)
parser.add_argument('--gamma', type=float, default=0.9)
parser.add_argument("--use_armijo", action="store_true")
parser.add_argument(
    "--armijo_mode",
    type=str,
    default="fixed",
    choices=("fixed", "geometric", "geometric_safe", "reject"),
)
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
parser.add_argument("--max_train_samples", type=int, default=0, help="Cap train split size after random split; 0 disables")
parser.add_argument("--max_valid_samples", type=int, default=0, help="Cap valid split size after random split; 0 disables")
parser.add_argument("--max_test_samples", type=int, default=0, help="Cap test split size after random split; 0 disables")

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
parser.add_argument(
    "--report_nr_polish",
    action="store_true",
    help=(
        "After test eval, warm-start a Newton-Raphson solver from the model's "
        "predicted V and count iterations-to-convergence (tol=--nr_polish_tol). "
        "Reports model-warm-start vs flat-start iteration counts and convergence "
        "rate -- quantifies how much solver work the surrogate saves."
    ),
)
parser.add_argument(
    "--nr_polish_solver",
    type=str,
    default="own",
    choices=["own", "pandapower"],
    help=(
        "NR engine for --report_nr_polish. 'own' = your vectorized NR from "
        "ScenarioSynthesis_PPC/newton_raphson_improved.py (the exact solver "
        "used for data generation). 'pandapower' = the pypower-derived polar NR "
        "that pandapower is built on, with pandapower's exact convergence test "
        "(||F||_inf < tol); implemented standalone since pandapower's newtonpf "
        "cannot run without a full net/ppci object."
    ),
)
parser.add_argument("--nr_polish_tol", type=float, default=1e-8,
                    help="Convergence tolerance (pu) for NR-polish (default 1e-8, research-grade).")
parser.add_argument("--nr_polish_max_iter", type=int, default=30,
                    help="Max NR iterations for the polish solve.")
parser.add_argument("--nr_polish_max_cases", type=int, default=200,
                    help="Cap on number of test cases to polish (NR per case is costly).")
parser.add_argument(
    "--nr_impl_path",
    type=str,
    default="/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC",
    help="Directory containing newton_raphson_improved.py (for --nr_polish_solver own).",
)
parser.add_argument(
    "--convergence_tol_pu",
    type=float,
    default=1e-3,
    help=(
        "Per-CASE convergence tolerance (pu) for the convergence-rate metric. "
        "A case counts as 'converged' when BOTH max|ΔP| (PV+PQ) and max|ΔQ| "
        "(PQ) fall below this tol -- the same case-level criterion pandapower "
        "uses (||F||_inf < tol). Default 1e-3 pu is an engineering threshold; "
        "set tighter (e.g. 1e-6) to match research-grade NR."
    ),
)
parser.add_argument(
    "--skip_initial_eval",
    action="store_true",
    help="Skip the full train/valid evaluation before epoch 1; useful for large lazy parquet diagnostics",
)

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
parser.add_argument(
    "--run_name",
    type=str,
    default="",
    help="Optional short run name for log/checkpoint/plot filenames; defaults to the full configuration name.",
)

args = parser.parse_args()


# ------------------------------------------------------------------
# Effective configuration
# ------------------------------------------------------------------
SEED = args.seed_value

PINN = args.PINN
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
shortened_names = ['_'.join(name.split('_')[:6]) for name in parquet_filenames]
parquet_filename = '_and_'.join(shortened_names)

armijo_tag = "True" if args.use_armijo else "False"
target_sbase_tag = "rawS" if args.target_S_base is None else f"Sbase{args.target_S_base:g}"
start_tag = "flat" if any("manual_flat" in name or "flat" in name for name in parquet_filenames) else "dc"
if any("dc_compile" in name or "_dc_" in name for name in parquet_filenames):
    start_tag = "dc"
loss_mode_tag = "pinn" if args.PINN else "mse"
if args.PINN and args.mse_weight > 0.0:
    loss_mode_tag = f"pinn_mse{args.mse_weight:g}"
elif (not args.PINN) and args.mse_weight > 0.0:
    loss_mode_tag = f"mse_w{args.mse_weight:g}"
vn_tag = "vnfeat" if args.vn_feat else "novn"
grid_cache_tag = (
    ("sharegrid" if args.share_grid else "nosharegrid")
    + ("_sharey" if args.share_ybus else "_nosharey")
    + ("_nocacheY" if args.no_cache_dense_ybus else "_cacheY")
    + ("_lazy" if args.lazy_parquet else "_eager")
)
armijo_detail_tag = "noarmijo"
if args.use_armijo:
    armijo_detail_tag = (
        f"arm{args.armijo_mode}"
        f"_rho{args.armijo_rho:g}"
        f"_bt{args.armijo_max_backtracks}"
        f"_amin{args.armijo_min_alpha:g}"
    )
loss_tag = (
    f"_loss{loss_mode_tag}"
    f"_msew{args.mse_weight:g}"
    f"_ploss{args.physics_loss_form}"
    f"_pnorm{args.physics_residual_norm}"
    f"_pfinal{args.physics_final_weight:g}"
)
run_config_tag = (
    f"_{start_tag}"
    f"_{args.model}"
    f"_{target_sbase_tag}"
    f"_{args.dataset_complex_dtype}"
    f"_{vn_tag}"
    f"_{grid_cache_tag}"
    f"_{armijo_detail_tag}"
    f"_batch{args.BATCH}"
    f"_lr{args.LR:g}"
    f"_seed{args.seed_value}"
    f"_valid{args.valid_ratio:g}"
)


RUNNAME = (
    f"{parquet_filename}_K{args.K}_d{args.d}_dhi{args.d_hi}"
    f"_nheads{args.n_heads}_numattn{args.num_attn_layers}"
    f"_armijo{armijo_tag}{loss_tag}{run_config_tag}"
    f"_ep{args.EPOCHS}_TrainRatio{args.train_ratio:g}"
)
if args.model == "GNSMsg_EdgeSelfAttnKHop":
    RUNNAME = (
        f"{parquet_filename}_solverK{args.K}_khop{args.khop_K}_sigma{args.khop_sigma:g}"
        f"_khop{args.khop_norm}_{args.khop_source}_d{args.d}_dhi{args.d_hi}"
        f"_nheads{args.n_heads}_numattn{args.num_attn_layers}"
        f"_armijo{armijo_tag}{loss_tag}{run_config_tag}"
        f"_ep{args.EPOCHS}_TrainRatio{args.train_ratio:g}"
    )
if args.run_name:
    RUNNAME = args.run_name
os.makedirs(args.ckpt_dir, exist_ok=True)
BEST_CKPT_PATH = os.path.join(args.ckpt_dir, f"{RUNNAME}_{EPOCHS}_best_model.ckpt")
if args.log_to_file:
    os.makedirs(args.log_dir, exist_ok=True)
    log_filename = os.path.join(args.log_dir, f"{RUNNAME}_training_log.txt")

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    sys.stdout = TeeLogger(log_filename, original_stdout)
    sys.stderr = TeeLogger(log_filename, original_stderr)

    def _cleanup_logger():
        stdout_logger = sys.stdout
        stderr_logger = sys.stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        try:
            stdout_logger.flush()
        except Exception:
            pass
        try:
            stderr_logger.flush()
        except Exception:
            pass
        try:
            stdout_logger.close()
        except Exception:
            pass
        try:
            stderr_logger.close()
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
    f"physics_final_weight:{args.physics_final_weight}, DthetaMax:{args.DthetaMax}, DvmFrac:{args.DvmFrac}, "
    f"khop_K:{args.khop_K}, khop_sigma:{args.khop_sigma}, "
    f"khop_norm:{args.khop_norm}, khop_source:{args.khop_source}, "
    f"dataset_complex_dtype:{args.dataset_complex_dtype}"
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
    target_S_base=args.target_S_base,
    share_ybus=args.share_ybus,
    share_grid=args.share_grid,
    complex_dtype=args.dataset_complex_dtype,
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

def cap_subset(split, max_samples: int):
    if max_samples is None or max_samples <= 0 or len(split) <= max_samples:
        return split
    return Subset(split.dataset, split.indices[:max_samples])

train_ds = cap_subset(train_ds, args.max_train_samples)
val_ds = cap_subset(val_ds, args.max_valid_samples)
test_ds = cap_subset(test_ds, args.max_test_samples)
n_train, n_val, n_test = len(train_ds), len(val_ds), len(test_ds)

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
    if BATCH == 1:
        train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False)
        test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False)

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
        bus_feat_extra_dim=1 if args.vn_feat else 0,
    ).to(device)

elif args.model == "GNSMsg_EdgeSelfAttnKHop":
    model = GNSMsg_EdgeSelfAttnKHop(
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
        khop_K=args.khop_K,
        khop_sigma=args.khop_sigma,
        khop_norm=args.khop_norm,
        khop_source=args.khop_source,
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
    if Y.dim() == 2:
        Y = Y.unsqueeze(0)

    if Y.is_complex():
        complex_dtype = Y.dtype
        real_dtype = torch.float64 if complex_dtype == torch.complex128 else torch.float32
    else:
        real_dtype = Y.dtype
        complex_dtype = torch.complex128 if real_dtype == torch.float64 else torch.complex64

    v = Vpred[..., 0].to(dtype=real_dtype)
    th = Vpred[..., 1].to(dtype=real_dtype)
    Sset = Sset.to(device=Y.device, dtype=complex_dtype)

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
        "convergence_rate": 0.0,
        "convergence_tol_pu": 0.0,
        "n_converged": 0,
        "n_cases": 0,
    }


def _safe_quantile(t: torch.Tensor, q: float) -> float:
    """Compute quantile robustly regardless of tensor size.

    torch.quantile has a hard element limit (~16M). For larger tensors
    (e.g. 28800 samples × 722 buses = 20.8M elements) we fall back to
    numpy which has no such restriction. numpy.percentile on CPU is fast
    enough for end-of-epoch metric computation.
    """
    try:
        return torch.quantile(t, q).item()
    except RuntimeError:
        # numpy fallback — move to CPU first if needed
        arr = t.cpu().numpy() if t.is_cuda else t.numpy()
        return float(np.percentile(arr, q * 100))


def finalize_residual_distribution(dp_values, dq_values, *, tol_pu):
    dist = _empty_residual_distribution()

    if dp_values:
        dp = torch.cat(dp_values).float()
        dist.update({
            "mean_dp_pu":          dp.mean().item(),
            "median_dp_pu":        _safe_quantile(dp, 0.50),
            "p95_dp_pu":           _safe_quantile(dp, 0.95),
            "p99_dp_pu":           _safe_quantile(dp, 0.99),
            "rmse_dp_pu":          torch.sqrt((dp ** 2).mean()).item(),
            "frac_dp_below_tol":   (dp <= tol_pu).float().mean().item(),
            "n_dp":                int(dp.numel()),
        })

    if dq_values:
        dq = torch.cat(dq_values).float()
        dist.update({
            "mean_dq_pu":          dq.mean().item(),
            "median_dq_pu":        _safe_quantile(dq, 0.50),
            "p95_dq_pu":           _safe_quantile(dq, 0.95),
            "p99_dq_pu":           _safe_quantile(dq, 0.99),
            "rmse_dq_pu":          torch.sqrt((dq ** 2).mean()).item(),
            "frac_dq_below_tol":   (dq <= tol_pu).float().mean().item(),
            "n_dq":                int(dq.numel()),
        })

    return dist


def format_residual_distribution_compact(dist):
    return (
        f"(mean |ΔP| {dist['mean_dp_pu']:.3e}, |ΔQ| {dist['mean_dq_pu']:.3e} pu; "
        f"p95 |ΔP| {dist['p95_dp_pu']:.3e}, |ΔQ| {dist['p95_dq_pu']:.3e} pu; "
        f"tol≤ {dist['frac_dp_below_tol']:.2%} P, {dist['frac_dq_below_tol']:.2%} Q; "
        f"conv {dist.get('convergence_rate', 0.0):.2%}@{dist.get('convergence_tol_pu', 0.0):.0e})"
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
        f"Q {dist['frac_dq_below_tol']:.2%} ({dist['n_dq']} entries)\n"
        f"  convergence rate (per-case, max|ΔP|&max|ΔQ| < {dist.get('convergence_tol_pu',0.0):.1e} pu): "
        f"{dist.get('convergence_rate',0.0):.2%} "
        f"({dist.get('n_converged',0)}/{dist.get('n_cases',0)} cases)"
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
    n_converged = 0   # per-case: both max|ΔP| and max|ΔQ| < convergence_tol_pu

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
            vn_log = batch["vn_log"].to(device) if "vn_log" in batch else None

            Y_metric = Y
            Sstart_metric = Sstart
            S_base_metric = S_base
            Branch_tau_metric = Branch_tau
            Branch_shift_deg_metric = Branch_shift_deg
            Branch_y_series_from_metric = Branch_y_series_from
            Branch_y_series_to_metric = Branch_y_series_to
            Branch_y_series_ft_metric = Branch_y_series_ft
            Branch_y_shunt_from_metric = Branch_y_shunt_from
            Branch_y_shunt_to_metric = Branch_y_shunt_to
            Y_shunt_bus_metric = Y_shunt_bus

            # Keep the neural network on its normal float32/complex64 path.
            # The dataset can decode LVN admittances in complex128 so that
            # validation/test residual diagnostics avoid complex64 cancellation
            # artifacts, but PyTorch Linear layers here are float32.
            if args.dataset_complex_dtype == "complex128":
                Branch_tau = Branch_tau.float()
                Branch_shift_deg = Branch_shift_deg.float()
                Branch_y_series_from = Branch_y_series_from.to(torch.complex64)
                Branch_y_series_to = Branch_y_series_to.to(torch.complex64)
                Branch_y_series_ft = Branch_y_series_ft.to(torch.complex64)
                Branch_y_shunt_from = Branch_y_shunt_from.to(torch.complex64)
                Branch_y_shunt_to = Branch_y_shunt_to.to(torch.complex64)
                Y_shunt_bus = Y_shunt_bus.to(torch.complex64)
                if Y is not None:
                    Y = Y.to(torch.complex64)
                Sstart = Sstart.to(torch.complex64)
                Ustart = Ustart.to(torch.complex64)
                Vstart = Vstart.float()
                Vnewton = Vnewton.float()
                if vn_log is not None:
                    vn_log = vn_log.float()

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
                    vn_log=vn_log,
                )

                dmag = (Vpred[..., 0] - Vnewton[..., 0])
                dang = torch.atan2(
                    torch.sin(Vpred[..., 1] - Vnewton[..., 1]),
                    torch.cos(Vpred[..., 1] - Vnewton[..., 1])
                )
                mse_mag = torch.mean(dmag ** 2)
                mse_ang = torch.mean(dang ** 2)
                mse = mse_mag + mse_ang
                # Combined loss: L = L_phys + w * L_MSE
                # w=0 (default) → pure PINN (original behaviour)
                # w>0 → MSE anchor pulls V toward V_newton while PINN
                #        satisfies physics; stabilizes training on stiff/
                #        multi-voltage grids where pure PINN converges to
                #        a non-V_newton physics-feasible solution.
                if args.mse_weight > 0.0:
                    loss = loss_phys + args.mse_weight * mse
                else:
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
                    vn_log=vn_log,
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

                # Zero-grad guard for pure-MSE mode (pinn=False).
                # Root cause: when Armijo rejects all K steps for every
                # iteration, Vpred == V_start (unchanged), which is a
                # constant with no grad_fn. The MSE against V_newton is
                # then detached and loss.backward() would crash.
                # The same guard exists in the PINN branch above; mirror
                # it here so pure-MSE mode is equally robust.
                if train and not loss.requires_grad:
                    p0 = next(model.parameters())
                    loss = loss + 0.0 * p0.norm()
                    print("[warn] MSE loss detached (Armijo rejected all steps); applied zero-grad guard.")

            residual_metrics = compute_power_flow_residual_metrics(
                ensure_dense_y_for_metrics(
                    Y_metric,
                    bus_type,
                    Branch_f_bus,
                    Branch_t_bus,
                    Branch_status,
                    Branch_tau_metric,
                    Branch_shift_deg_metric,
                    Branch_y_series_from_metric,
                    Branch_y_series_to_metric,
                    Branch_y_series_ft_metric,
                    Branch_y_shunt_from_metric,
                    Branch_y_shunt_to_metric,
                    Y_shunt_bus_metric,
                ),
                Vpred,
                Sstart_metric,
                bus_type,
                n_nodes_per_graph=n_nodes_per_graph,
                S_base=S_base_metric,
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

            # Per-case convergence: pandapower-style ||F||_inf < tol applied
            # per case (both masked maxima below the engineering tolerance).
            _ctol = args.convergence_tol_pu
            _conv = (residual_metrics["max_dp_pu"] < _ctol) & (residual_metrics["max_dq_pu"] < _ctol)
            n_converged += int(_conv.sum().item())

    mean_loss = sum_loss / max(n_graphs_total, 1)
    convergence_rate = n_converged / max(n_graphs_total, 1)
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
    # Per-case convergence rate (pandapower-style ||F||_inf < tol, per case).
    # Carried inside residual_dist so no call-site signatures change.
    residual_dist["convergence_rate"] = convergence_rate
    residual_dist["convergence_tol_pu"] = args.convergence_tol_pu
    residual_dist["n_converged"] = n_converged
    residual_dist["n_cases"] = n_graphs_total
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

    if args.skip_initial_eval:
        print("Initial metrics before training: skipped (--skip_initial_eval)")
    else:
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
# NR-polish: warm-start a Newton-Raphson solver from the model prediction
# and count iterations to convergence. Two engines selectable.
# ------------------------------------------------------------------
def _busmap_own(bus_type_np):
    """Remap to the own-NR convention (1=slack, 2=PV, 3=PQ) regardless of how
    PQ is coded in the source (0 or 3)."""
    bt = np.full(bus_type_np.shape, 3, dtype=np.int64)
    bt[bus_type_np == 1] = 1
    bt[bus_type_np == 2] = 2
    return bt


def _nr_polish_own(Ybus, Sbus, V0, bus_type_np, tol, max_iter):
    """Use the data-generation NR (newton_raphson_improved.newtonrapson)."""
    import sys as _sys
    if args.nr_impl_path not in _sys.path:
        _sys.path.insert(0, args.nr_impl_path)
    from newton_raphson_improved import newtonrapson
    bt = _busmap_own(bus_type_np)
    _, _, _, diag = newtonrapson(
        bt, Ybus.astype(np.complex128), Sbus.astype(np.complex128),
        V0.astype(np.complex128),
        K=int(max_iter), diagnose=True, return_diagnostics=True,
        convergence_mode="misinf", mismatch_tol=float(tol),
    )
    return int(diag.get("iterations", 0)), bool(diag.get("converged", False))


def _nr_polish_pypower(Ybus, Sbus, V0, bus_type_np, tol, max_iter):
    """Standalone polar Newton-Raphson identical to pypower/pandapower's
    newtonpf core, with pandapower's convergence test ||F||_inf < tol."""
    Ybus = Ybus.astype(np.complex128)
    Sbus = Sbus.astype(np.complex128)
    V = V0.astype(np.complex128).copy()
    Va = np.angle(V); Vm = np.abs(V)
    ref = np.where(bus_type_np == 1)[0]
    pv = np.where(bus_type_np == 2)[0]
    pq = np.where((bus_type_np != 1) & (bus_type_np != 2))[0]
    pvpq = np.r_[pv, pq]
    npvpq, npq = len(pvpq), len(pq)

    def mismatch(V):
        Scalc = V * np.conj(Ybus @ V)
        dS = Scalc - Sbus
        return np.r_[dS[pvpq].real, dS[pq].imag]

    F = mismatch(V)
    converged = np.linalg.norm(F, np.inf) < tol
    i = 0
    while (not converged) and i < int(max_iter):
        i += 1
        Ibus = Ybus @ V
        diagV = np.diag(V)
        diagIbus = np.diag(Ibus)
        diagVnorm = np.diag(V / np.abs(V))
        dS_dVm = diagV @ np.conj(Ybus @ diagVnorm) + np.conj(diagIbus) @ diagVnorm
        dS_dVa = 1j * diagV @ np.conj(diagIbus - Ybus @ diagV)
        J11 = dS_dVa[np.ix_(pvpq, pvpq)].real
        J12 = dS_dVm[np.ix_(pvpq, pq)].real
        J21 = dS_dVa[np.ix_(pq, pvpq)].imag
        J22 = dS_dVm[np.ix_(pq, pq)].imag
        J = np.block([[J11, J12], [J21, J22]])
        try:
            dx = -np.linalg.solve(J, F)
        except np.linalg.LinAlgError:
            return i, False
        Va[pvpq] += dx[:npvpq]
        Vm[pq] += dx[npvpq:npvpq + npq]
        V = Vm * np.exp(1j * Va)
        Va, Vm = np.angle(V), np.abs(V)
        F = mismatch(V)
        converged = np.linalg.norm(F, np.inf) < tol
    return i, bool(converged)


def run_nr_polish_eval():
    """Warm-start NR from model prediction vs flat start; report iterations."""
    polish = _nr_polish_own if args.nr_polish_solver == "own" else _nr_polish_pypower
    print(f"\n[nr-polish] solver={args.nr_polish_solver} tol={args.nr_polish_tol:.1e} "
          f"max_iter={args.nr_polish_max_iter} cap={args.nr_polish_max_cases} cases")

    # batch_size=1 single-graph loader so each case is one grid (no blockdiag split)
    polish_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                               collate_fn=collate_blockdiag)
    iters_pred, iters_flat = [], []
    conv_pred, conv_flat = 0, 0
    n_done = 0
    model.eval()
    with torch.no_grad():
        for batch in polish_loader:
            if n_done >= args.nr_polish_max_cases:
                break
            bus_type = batch["bus_type"].to(device)
            Branch_f_bus = batch["Branch_f_bus"].to(device)
            Branch_t_bus = batch["Branch_t_bus"].to(device)
            Branch_status = batch["Branch_status"].to(device)
            Branch_tau = batch["Branch_tau"].to(device)
            Branch_shift_deg = batch["Branch_shift_deg"].to(device)
            Branch_y_series_from = batch["Branch_y_series_from"].to(device)
            Branch_y_series_to = batch["Branch_y_series_to"].to(device)
            Branch_y_series_ft = batch["Branch_y_series_ft"].to(device)
            Branch_y_shunt_from = batch["Branch_y_shunt_from"].to(device)
            Branch_y_shunt_to = batch["Branch_y_shunt_to"].to(device)
            Is_trafo = batch["Is_trafo"].to(device)
            Y_shunt_bus = batch["Y_shunt_bus"].to(device)
            if BLOCK_DIAG and "sizes" in batch:
                n_nodes_per_graph = batch["sizes"].to(device)
            else:
                n_nodes_per_graph = None
            Sstart = batch["S_start"].to(device)
            Vstart = batch["V_start"].to(device)
            vn_log = batch["vn_log"].to(device) if "vn_log" in batch else None
            Y = batch.get("Ybus", None)
            if Y is not None:
                Y = Y.to(device)

            Yd = ensure_dense_y_for_metrics(
                Y, bus_type, Branch_f_bus, Branch_t_bus, Branch_status,
                Branch_tau, Branch_shift_deg, Branch_y_series_from,
                Branch_y_series_to, Branch_y_series_ft, Branch_y_shunt_from,
                Branch_y_shunt_to, Y_shunt_bus,
            )
            out = model(
                bus_type, Branch_f_bus, Branch_t_bus, Branch_status,
                Branch_tau, Branch_shift_deg, Branch_y_series_from,
                Branch_y_series_to, Branch_y_series_ft, Branch_y_shunt_from,
                Branch_y_shunt_to, Is_trafo, Y, Sstart, Vstart,
                n_nodes_per_graph=n_nodes_per_graph, Y_shunt_bus=Y_shunt_bus,
                vn_log=vn_log,
            )
            Vpred = out[0] if isinstance(out, tuple) else out

            Yb = (Yd[0] if Yd.dim() == 3 else Yd).cpu().numpy()
            S_np = Sstart[0].cpu().numpy()
            bt_np = bus_type[0].cpu().numpy()
            vp = Vpred[0].cpu().numpy()
            V0_pred = vp[:, 0] * np.exp(1j * vp[:, 1])
            vs = Vstart[0].cpu().numpy()
            V0_flat = vs[:, 0] * np.exp(1j * vs[:, 1])

            ip, cp = polish(Yb, S_np, V0_pred, bt_np, args.nr_polish_tol, args.nr_polish_max_iter)
            iff, cf = polish(Yb, S_np, V0_flat, bt_np, args.nr_polish_tol, args.nr_polish_max_iter)
            iters_pred.append(ip); conv_pred += int(cp)
            iters_flat.append(iff); conv_flat += int(cf)
            n_done += 1

    if n_done == 0:
        print("[nr-polish] no cases evaluated."); return
    ip = np.array(iters_pred); iff = np.array(iters_flat)
    print(f"[nr-polish] cases={n_done}")
    print(f"[nr-polish] model warm-start : iters mean {ip.mean():.2f}  median {np.median(ip):.0f}  "
          f"max {ip.max()}  converged {conv_pred}/{n_done} ({conv_pred/n_done:.1%})")
    print(f"[nr-polish] flat  start      : iters mean {iff.mean():.2f}  median {np.median(iff):.0f}  "
          f"max {iff.max()}  converged {conv_flat}/{n_done} ({conv_flat/n_done:.1%})")
    saved = iff.mean() - ip.mean()
    print(f"[nr-polish] iterations saved by surrogate (flat - model): {saved:.2f} "
          f"({saved/max(iff.mean(),1e-9):.1%} reduction)")


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

    if args.report_nr_polish:
        run_nr_polish_eval()

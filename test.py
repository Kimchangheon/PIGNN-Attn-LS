# test_only.py
import os, sys, time, math, argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np

# === Imports kept identical to your training script (order matters if names shadow) ===
from GNN.SimpleGNN.GNSNode import GNSNode
from GNN.SimpleGNN.GNSMsg import GNSMsg as GNSMsg
from GNN.SimpleGNN.GNSMsg_armijo import GNSMsg as GNSMsg_armijo
from GNN.SimpleGNN.GNSMsg_armijo_5think import GNSMsg as GNSMsg_armijo

from GNN.SimpleGNN.GNSMsg_SelfAttention_armijo import  GNSMsg_EdgeSelfAttn
from GNN.SimpleGNN.GNSMsg_SelfAttention_armijo_5think import  GNSMsg_EdgeSelfAttn

from GNN.SimpleGNN.GNSMsgGRU_armijo import GNSMsgGRU

# Datasets (second import intentionally shadows the first, same as your script)
from GNN.SimpleGNN.Dataset_optimized_complex_columns import ChanghunDataset
# from GNN.SimpleGNN.Dataset_optimized_complex_columns_no_remove import ChanghunDataset

from collate_blockdiag_optimized import *
from collate_blockdiag_optimized_complex_columns import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.makedirs("./results/ckpt", exist_ok=True)
os.makedirs("./results/plots", exist_ok=True)
os.makedirs("./results/logs", exist_ok=True)

# -----------------------------
# 0) Args (same defaults) + ckpt
# -----------------------------
parser = argparse.ArgumentParser(description="Test-only script with same hyperparameters")
parser.add_argument("--PINN", action="store_true")
parser.add_argument("--BLOCK_DIAG", action="store_true")
parser.add_argument("--NORMALIZE", action="store_true")
parser.add_argument("--PER_UNIT", action="store_true")
parser.add_argument("--float64", action="store_true")
parser.add_argument('--mode', type=str, default="test", help='forced to test-only')
parser.add_argument("--mag_ang_mse", action="store_true")

parser.add_argument('--model', type=str, default="GNSMsg_EdgeSelfAttn")
parser.add_argument("--d", type=int, default=8)
parser.add_argument("--d_hi", type=int, default=32)
parser.add_argument("--K", type=int, default=80)
parser.add_argument('--gamma', type=float, default=0.9)
parser.add_argument("--use_armijo", action="store_true")
parser.add_argument("--vlimit", action="store_true")
parser.add_argument('--DthetaMax', type=float, default=0.3)
parser.add_argument('--DvmFrac', type=float, default=0.1)
parser.add_argument('--train_ratio', type=float, default=0)
parser.add_argument('--valid_ratio', type=float, default=0)

parser.add_argument("--ADJ_MODE", type=str, default="cplx")

parser.add_argument('--weight_init', type=str, default="sd0.02")
parser.add_argument('--bias_init', type=float, default=0.0)
parser.add_argument('--weight_decay', type=float, default=1e-3)

parser.add_argument('--lr_scheduler', type=str, default="default")
parser.add_argument('--cosineRestartEpoch', type=int, default=20)

parser.add_argument("--BATCH", type=int, default=64)
parser.add_argument("--EPOCHS", type=int, default=20)  # kept for RUNNAME compatibility
parser.add_argument("--LR", type=float, default=1e-4)
parser.add_argument("--VAL_EVERY", type=int, default=1)

parser.add_argument("--PARQUET", type=str, nargs='+',
                    default=["./data/MVN_15000_armijo_4_to_32_buses.parquet"])
parser.add_argument("--seed_value", type=int, default=42)

# NEW: required checkpoint path
parser.add_argument("--ckpt", type=str, required=True, help="Path to model weights (.ckpt)")

args = parser.parse_args()

SEED = args.seed_value

# Extract filenames (same RUNNAME logic)
parquet_filenames = [os.path.splitext(os.path.basename(p))[0] for p in args.PARQUET]
shortened_names = ['_'.join(name.split('_')[:3]) for name in parquet_filenames]
parquet_filename = '_and_'.join(shortened_names)

RUNNAME = f"{args.model}_9al_b{args.BATCH}_{parquet_filename}_K{args.K}_d{args.d}_dhi{args.d_hi}_ep{args.EPOCHS}_TrainRatio{args.train_ratio}"
if args.use_armijo : RUNNAME += "_armijo"
if args.vlimit : RUNNAME += "_vlimit"

# Simple logger (writes to file + stdout)
class Logger:
    def __init__(self, filename="output.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message); self.log.flush()
    def flush(self): pass

log_filename = f"./results/logs/{RUNNAME}_test_log.txt"
sys.stdout = Logger(log_filename)

# Force same flags as your script does
PINN       = True
# BLOCK_DIAG = True
BLOCK_DIAG = args.BLOCK_DIAG
NORMALIZE  = False
PER_UNIT   = True
args.mag_ang_mse = True

MODEL     = args.model
ADJ_MODE  = args.ADJ_MODE
BATCH     = args.BATCH
EPOCHS    = args.EPOCHS  # unused for testing but kept
LR        = args.LR      # unused for testing but kept
PARQUET   = args.PARQUET
d         = args.d
d_hi      = args.d_hi
K         = args.K
GAMMA     = args.gamma
VLIMIT    = args.vlimit
DthetaMax = args.DthetaMax
DvmFrac   = args.DvmFrac

print(f"[Test-Only] MODEL:{MODEL}, PINN:{PINN}, Block:{BLOCK_DIAG}, d:{d}, d_hi:{d_hi}, K:{K}")
print(f"Runname:{RUNNAME}\nPARQUET:{PARQUET}\nBATCH:{BATCH}\nCKPT:{args.ckpt}")

# -----------------------------
# 1) Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# 2) Dataset & split (same logic; we only use test split)
# -----------------------------
full_ds = ChanghunDataset(PARQUET, per_unit=PER_UNIT, device=device)
n_total = len(full_ds)
n_train = int(args.train_ratio * n_total)
n_val   = int(((1-args.train_ratio)/2) * n_total)
n_test  = n_total - n_train - n_val

train_ds, val_ds, test_ds = random_split(
    full_ds,
    lengths=[n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(SEED)
)

if BATCH == 1:
    test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False)
else:
    if BLOCK_DIAG:
        test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, collate_fn=collate_blockdiag)
    else:
        sizes = [full_ds[i]["N"] for i in range(len(full_ds))]
        test_sampler = MultiBucketBatchSampler(
            sizes=np.take(sizes, test_ds.indices),
            batch_size=BATCH,
            shuffle=False
        )
        test_loader = DataLoader(test_ds, batch_sampler=test_sampler)

print(f"Dataset sizes  |  train {n_train}   valid {n_val}   test {n_test}")

# -----------------------------
# 3) Model (same init & weight init)
# -----------------------------
if args.model == "GNSMsg":
    model = GNSMsg_armijo(d=d, d_hi=d_hi, K=K, pinn=PINN, gamma=GAMMA, v_limit=VLIMIT, use_armijo=args.use_armijo).to(device)
elif args.model == "GNSMsg_EdgeSelfAttn":
    model = GNSMsg_EdgeSelfAttn(d=d, d_hi=d_hi, K=K, pinn=PINN, gamma=GAMMA, v_limit=VLIMIT, use_armijo=args.use_armijo).to(device)
elif args.model == "GNSMsgGRU":
    model = GNSMsgGRU(d=d, d_hi=d_hi, K=K, pinn=PINN, gamma=GAMMA, v_limit=VLIMIT, use_armijo=args.use_armijo).to(device)
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

if args.float64:
    model = model.double()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of parameters: {count_parameters(model)}")

# -----------------------------
# 4) Loss helper (same as script)
# -----------------------------
def run_epoch(loader, *, train: bool, pinn: bool):
    model.train() if train else model.eval()
    sum_loss = 0.0
    sum_mse  = 0.0
    sum_mse_mag = 0.0
    sum_mse_ang = 0.0
    n_samples = 0

    with torch.no_grad():
        for batch in loader:
            B = batch["bus_type"].size(0)
            n_samples += B

            if BLOCK_DIAG:
                n_nodes_per_graph = batch["sizes"].to(device)
            else:
                n_nodes_per_graph = None

            bus_type = batch["bus_type"].to(device)
            Line     = batch["Lines_connected"].to(device)
            Y        = batch["Ybus"].to(device)
            Ys       = batch["Y_Lines"].to(device)
            Yc       = batch["Y_C_Lines"].to(device)

            Sstart   = batch["S_start"].to(device)
            Vstart   = batch["V_start"].to(device)
            Vnewton  = batch["V_newton"].to(device)

            if pinn:
                Vpred, loss_phys = model(bus_type, Line, Y, Ys, Yc, Sstart, Vstart, n_nodes_per_graph)
                dmag = (Vpred[..., 0] - Vnewton[..., 0])
                dang = torch.atan2(torch.sin(Vpred[..., 1] - Vnewton[..., 1]),
                                   torch.cos(Vpred[..., 1] - Vnewton[..., 1]))
                mse_mag = torch.mean(dmag ** 2)
                mse_ang = torch.mean(dang ** 2)
                mse = mse_mag + mse_ang
                loss = loss_phys
            else:
                Vpred = model(bus_type, Line, Y, Ys, Yc, Sstart, Vstart, n_nodes_per_graph)
                dmag = (Vpred[..., 0] - Vnewton[..., 0])
                dang = torch.atan2(torch.sin(Vpred[..., 1] - Vnewton[..., 1]),
                                   torch.cos(Vpred[..., 1] - Vnewton[..., 1]))
                mse_mag = torch.mean(dmag ** 2)
                mse_ang = torch.mean(dang ** 2)
                mse = mse_mag + mse_ang
                loss = mse

            sum_loss += loss.item() * B
            sum_mse  += mse.item()  * B
            sum_mse_mag += mse_mag.item() * B
            sum_mse_ang += mse_ang.item() * B

    mean_loss = sum_loss / n_samples
    mean_mse  = sum_mse  / n_samples
    mean_mse_mag = sum_mse_mag / n_samples
    mean_mse_ang = sum_mse_ang / n_samples
    return mean_loss, mean_mse, mean_mse_mag, mean_mse_ang

# -----------------------------
# 5) Load weights & TEST
# -----------------------------
print(f"\nLoading checkpoint from: {args.ckpt}")
ckpt = torch.load(args.ckpt, map_location=device)
model.load_state_dict(ckpt, strict=True)
print("Checkpoint loaded.")

test_loss, test_mse, test_mse_mag, test_mse_ang = run_epoch(
    test_loader, train=False, pinn=PINN
)

test_rmse = math.sqrt(test_mse)
test_rmse_mag = math.sqrt(test_mse_mag)
test_rmse_ang_deg = math.sqrt(test_mse_ang) * (180.0 / math.pi)

if PINN:
    print(
        f"\n[Test results]\n"
        f" physics-loss : {test_loss:.4e}\n"
        f" total RMSE   : {test_rmse:.4e}\n"
        f" |V| RMSE     : {test_rmse_mag:.4e}\n"
        f" θ RMSE       : {test_rmse_ang_deg:.4e}°"
    )
else:
    print(
        f"\nFinal test-set RMSE : {test_rmse:.4e}"
        f" (|V|: {test_rmse_mag:.4e}, θ: {test_rmse_ang_deg:.4e}°)"
    )

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from torch.utils.data import DataLoader, random_split


from GNSMsg_SelfAttention_armijo import GNSMsg_EdgeSelfAttn
from GNSMsg_armijo import GNSMsg

from Dataset_optimized_complex_columns import ChanghunDataset


from helper import *
# from collate_blockdiag import *
from collate_blockdiag_optimized_complex_columns import *

import multiprocessing as mp


import os
import time



os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

os.makedirs("./results/ckpt", exist_ok=True)
os.makedirs("./results/plots", exist_ok=True)

# ------------------------------------------------------------------
# 0.  Hyper-parameters
# ------------------------------------------------------------------
import argparse

parser = argparse.ArgumentParser(description="Training script with configurable hyperparameters")

parser.add_argument("--PINN", action="store_true", help="Enable Physics-Informed Neural Networks")
parser.add_argument("--BLOCK_DIAG", action="store_true", help="Use block diagonal mode")
parser.add_argument("--NORMALIZE", action="store_true", help="Enable Physics-Informed Neural Networks")
parser.add_argument("--PER_UNIT", action="store_true", help="Enable Physics-Informed Neural Networks")
parser.add_argument("--float64", action="store_true", help="Enable Physics-Informed Neural Networks")
parser.add_argument('--mode', type=str, default="train_test", help='train_valid_test | train | valid | test')
parser.add_argument("--mag_ang_mse", action="store_true", help="normalised |V| + wrapped-angle loss ")

parser.add_argument('--model', type=str, default="GNSMsg_EdgeSelfAttn", help='GNSMsg, GNSMsgGRU, etc')
parser.add_argument("--d", type=int, default=4, help="model input dim")
parser.add_argument("--d_hi", type=int, default=16, help="model hidden dim")
parser.add_argument("--K", type=int, default=40, help="K")
parser.add_argument('--gamma', type=float, default=0.9, help='phys_loss decay over K')
parser.add_argument("--use_armijo", action="store_true", help="use_armijo")
parser.add_argument("--vlimit", action="store_true", help="vlimit disabled")
parser.add_argument('--DthetaMax', type=float, default=0.3, help='DthetaMax')
parser.add_argument('--DvmFrac', type=float, default=0.1, help='DvmFrac')
parser.add_argument('--train_ratio', type=float, default=0.8, help='train_ratio')
parser.add_argument('--valid_ratio', type=float, default=0.1, help='valid_ratio')


parser.add_argument("--ADJ_MODE", type=str, default="cplx", help="Adjacency mode: real | cplx | other")

parser.add_argument('--weight_init', type=str, default="sd0.02", help='Weight initialization method (None, He, sd0.02)')
parser.add_argument('--bias_init', type=float, default=0.0, help='Bias initialization value (0.01, 0)')
parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay rate')

parser.add_argument('--lr_scheduler', type=str, default="default", help='CosineAnnealingLR')
parser.add_argument('--cosineRestartEpoch', type=int, default=20, help='cosineRestartEpoch')

parser.add_argument("--BATCH", type=int, default=16, help="Batch size")
parser.add_argument("--EPOCHS", type=int, default=20, help="Number of training epochs")
parser.add_argument("--LR", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--VAL_EVERY", type=int, default=1, help="Validation frequency (in epochs)")
# parser.add_argument("--PARQUET", type=str, default="./data/MVN_15000_armijo_4_to_32_buses.parquet", help="Path to Parquet data file")
parser.add_argument("--PARQUET", type=str, nargs='+', default=["./data/HVN_15000_NR_plain_4_to_32_buses.parquet"], help="Path to Parquet data file(s)")
parser.add_argument("--seed_value", type=int, default=42, help="Batch size")

args = parser.parse_args()

SEED = args.seed_value
# Assign to variables if needed
PINN       = True

# Extract just the filename without extension
parquet_filenames = [os.path.splitext(os.path.basename(p))[0] for p in args.PARQUET]

# Keep only up to the 3rd underscore (split into max 3 parts, then rejoin)
shortened_names = ['_'.join(name.split('_')[:3]) for name in parquet_filenames]

# Join them with '_and_'
parquet_filename = '_and_'.join(shortened_names)

# Final RUNNAME
RUNNAME = f"{parquet_filename}_K{args.K}_d{args.d}_dhi{args.d_hi}_ep{args.EPOCHS}_TrainRatio{args.train_ratio}"



BLOCK_DIAG = args.BLOCK_DIAG
NORMALIZE = args.NORMALIZE
PER_UNIT = args.PER_UNIT

PINN       = True
BLOCK_DIAG = True
NORMALIZE = False
PER_UNIT = True
#
args.mag_ang_mse = True

MODEL = args.model
ADJ_MODE   = args.ADJ_MODE
BATCH      = args.BATCH
# BATCH=1
EPOCHS     = args.EPOCHS
LR         = args.LR
VAL_EVERY  = args.VAL_EVERY
PARQUET    = args.PARQUET
d = args.d
d_hi = args.d_hi
K = args.K
GAMMA = args.gamma
VLIMIT = args.vlimit
DthetaMax = args.DthetaMax
DvmFrac = args.DvmFrac

import sys
# Add this class definition after your imports
class Logger:
    def __init__(self, filename="output.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

# Then, after RUNNAME is defined (around line 125 in your code), add:
log_filename = f"./results/{RUNNAME}_training_log.txt"
sys.stdout = Logger(log_filename)


print(f"MODEL:{MODEL}, PINN:{PINN}, Block:{BLOCK_DIAG}, d:{d}, d_hi:{d_hi},K:{K}, Runname:{RUNNAME}, PARQUET:{PARQUET}, BATCH:{BATCH}, EP:{EPOCHS}, LR:{LR}")

# ------------------------------------------------------------------
# 1.  Device
# ------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------------------------------------------------------
# 2.  Dataset  →  Train/Valid/Test split
# ------------------------------------------------------------------
full_ds = ChanghunDataset(PARQUET, per_unit=PER_UNIT, device=device)
n_total = len(full_ds)
n_train = int(args.train_ratio * n_total)
n_val   = int(args.valid_ratio * n_total)
n_test  = n_total - n_train - n_val

train_ds, val_ds, test_ds = random_split(
        full_ds,
        lengths=[n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED))   # reproducible split


if BATCH ==1 :
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False)
else :
    if BLOCK_DIAG :
        train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,collate_fn=collate_blockdiag)
        val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False,collate_fn=collate_blockdiag)
        test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False,collate_fn=collate_blockdiag)
    else :
        sizes = [full_ds[i]["N"] for i in range(len(full_ds))]  # list of ints
        # train_loader = make_size_bucketing_loader(train_ds, BATCH, shuffle=True)
        # val_loader   = make_size_bucketing_loader(val_ds,   BATCH, shuffle=False)
        # test_loader  = make_size_bucketing_loader(test_ds,  BATCH, shuffle=False)

        # build bucketed loaders
        train_sampler = MultiBucketBatchSampler(
            sizes=np.take(sizes, train_ds.indices),
            batch_size=BATCH,
            shuffle=True)

        val_sampler = MultiBucketBatchSampler(
            sizes=np.take(sizes, val_ds.indices),
            batch_size=BATCH,
            shuffle=False)

        test_sampler = MultiBucketBatchSampler(
            sizes=np.take(sizes, test_ds.indices),
            batch_size=BATCH,
            shuffle=False)
        train_loader = DataLoader(train_ds, batch_sampler=train_sampler)
        val_loader = DataLoader(val_ds, batch_sampler=val_sampler)
        test_loader = DataLoader(test_ds, batch_sampler=test_sampler)

print(f"Dataset sizes  |  train {n_train}   valid {n_val}   test {n_test}")

# ------------------------------------------------------------------
# 3.  Model / Optim / Loss
# ------------------------------------------------------------------

if args.model =="GNSMsg" :
    model = GNSMsg(d=d, d_hi=d_hi, K=K, pinn=PINN, gamma=GAMMA, v_limit=VLIMIT, use_armijo=args.use_armijo).to(device)
elif args.model =="GNSMsg_EdgeSelfAttn" :
    model = GNSMsg_EdgeSelfAttn(d=d, d_hi=d_hi, K=K, pinn=PINN, gamma=GAMMA, v_limit=VLIMIT, use_armijo=args.use_armijo).to(device)
# model = model.double()

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


total_params = count_parameters(model)
print(f"Total number of parameters: {total_params}")


# optim  = torch.optim.Adam(model.parameters(), lr=LR)
if args.lr_scheduler == "CosineAnnealingLR":
    steps_per_epoch = len(train_loader)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=args.weight_decay)
    T_0 = args.cosineRestartEpoch * steps_per_epoch  # first cycle length in steps
    scheduler = CosineAnnealingWarmRestarts(optim, T_0=T_0, T_mult=1, eta_min=1e-6)
else :
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=args.weight_decay)
loss_f = torch.nn.MSELoss()

def mag_ang_mse(Vpred, Vref, w_mag=1.0, w_ang=1/torch.pi):
    """
    Scale-balanced MSE:
      • magnitude difference divided by w_mag
      • angle error wrapped to (-π,π] and divided by w_ang
    """
    dmag = (Vpred[..., 0] - Vref[..., 0]) / w_mag
    dang = torch.atan2(
        torch.sin(Vpred[..., 1] - Vref[..., 1]),
        torch.cos(Vpred[..., 1] - Vref[..., 1])
    ) / w_ang
    return torch.mean(dmag ** 2 + dang ** 2)

# ------------------------------------------------------------------
# 4.  Helper: one epoch pass (train=False → no grad)
# ------------------------------------------------------------------
# ------------------------------------------------------
# helper: one full pass over a loader
# ------------------------------------------------------
def run_epoch(loader, *, train: bool, pinn: bool):
    model.train() if train else model.eval()

    sum_loss = 0.0          # physics   (always)
    sum_mse  = 0.0          # |V|,δ MSE (only meaningful when pinn=True)

    sum_mse_mag = 0.0
    sum_mse_ang = 0.0

    n_samples = 0

    with torch.set_grad_enabled(train):
        for batch in loader:
            B = batch["bus_type"].size(0)        # current mini-batch size
            n_samples += B

            # move to device ------------------------------------------------
            if BLOCK_DIAG :
                n_nodes_per_graph = batch["sizes"].to(device)
            else :
                n_nodes_per_graph = None

            bus_type = batch["bus_type"].to(device)
            Line    = batch["Lines_connected"].to(device)
            # Y       = batch["Ybus"].to(device)
            Y_raw = batch.get("Ybus", None)
            Y = Y_raw.to(device, non_blocking=True) if isinstance(Y_raw, torch.Tensor) else None
            Ys       = batch["Y_Lines"].to(device)
            Yc       = batch["Y_C_Lines"].to(device)

            Sstart   = batch["S_start"].to(device)

            Ustart = batch["U_start"].to(device)
            Vstart = Ustart
            Vstart   = batch["V_start"].to(device)

            Vnewton    = batch["V_newton"].to(device)

            # forward -------------------------------------------------------
            if pinn:
                Vpred, loss_phys = model(bus_type, Line, Y, Ys, Yc, Sstart, Vstart, n_nodes_per_graph)
                assert isinstance(Vpred, torch.Tensor), "expected tensor from model"

                dmag = (Vpred[..., 0] - Vnewton[..., 0])
                dang = torch.atan2(
                    torch.sin(Vpred[..., 1] - Vnewton[..., 1]),
                    torch.cos(Vpred[..., 1] - Vnewton[..., 1])
                )
                mse_mag = torch.mean(dmag ** 2)
                mse_ang = torch.mean(dang ** 2)
                mse = mse_mag + mse_ang

                loss = loss_phys
            else:
                Vpred = model(bus_type, Line, Y, Ys, Yc, Sstart, Vstart, n_nodes_per_graph)
                assert isinstance(Vpred, torch.Tensor), "expected tensor from model"

                w_mag, w_ang = 1.0, 1.0
                dmag = (Vpred[..., 0] - Vnewton[..., 0])
                dang = torch.atan2(
                    torch.sin(Vpred[..., 1] - Vnewton[..., 1]),
                    torch.cos(Vpred[..., 1] - Vnewton[..., 1])
                )
                mse_mag = torch.mean(dmag ** 2)
                mse_ang = torch.mean(dang ** 2)
                mse = mse_mag*w_mag + mse_ang*w_ang

                loss = mse                          # same number for convenience

            # backward / optim ---------------------------------------------
            if train:
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optim.step()
                if args.lr_scheduler == "CosineAnnealingLR":  # Add this condition
                    scheduler.step()


                    # aggregate -----------------------------------------------------
            sum_loss += loss.item() * B
            sum_mse  += mse.item()  * B
            sum_mse_mag += mse_mag.item()  * B
            sum_mse_ang += mse_ang.item()  * B

    mean_loss = sum_loss / n_samples
    mean_mse  = sum_mse  / n_samples
    mean_mse_mag = sum_mse_mag / n_samples
    mean_mse_ang = sum_mse_ang / n_samples
    return mean_loss, mean_mse, mean_mse_mag, mean_mse_ang        # always a tuple


import math

# ---------------------------------------------------------------------
# 5.  Training / validation
# ---------------------------------------------------------------------
if "train" in args.mode:
    train_loss_hist, train_rmse_hist        = [], []
    train_rmse_mag_hist, train_rmse_ang_hist_deg = [], []

    val_loss_hist,   val_rmse_hist          = [], []
    val_rmse_mag_hist, val_rmse_ang_hist_deg     = [], []

    best_val_loss = float('inf')

    # Get initial losses and metrics before training
    print("Initial metrics before training:")
    train_loss, train_mse, train_mse_mag, train_mse_ang = \
        run_epoch(train_loader, train=False, pinn=PINN)  # Note train=False here

    train_rmse = math.sqrt(train_mse)
    train_rmse_mag = math.sqrt(train_mse_mag)
    train_rmse_ang_deg = math.sqrt(train_mse_ang) * (180.0 / math.pi)

    val_loss, val_mse, val_mse_mag, val_mse_ang = \
        run_epoch(val_loader, train=False, pinn=PINN)

    val_rmse = math.sqrt(val_mse)
    val_rmse_mag = math.sqrt(val_mse_mag)
    val_rmse_ang_deg = math.sqrt(val_mse_ang) * (180.0 / math.pi)

    print(f"Epoch   0 | "
          f"train loss {train_loss:.4e}  rmse {train_rmse:.4e} "
          f"(mag {train_rmse_mag:.4e}, ang {train_rmse_ang_deg:.4e}°) | "
          f"valid loss {val_loss:.4e}  rmse {val_rmse:.4e} "
          f"(mag {val_rmse_mag:.4e}, ang {val_rmse_ang_deg:.4e}°)")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # -------- train pass ------------------------------------------
        train_loss, train_mse, train_mse_mag, train_mse_ang = \
            run_epoch(train_loader, train=True, pinn=PINN)

        train_rmse = math.sqrt(train_mse)
        train_rmse_mag = math.sqrt(train_mse_mag)
        train_rmse_ang_deg = math.sqrt(train_mse_ang) * (180.0 / math.pi)

        train_loss_hist.append(train_loss)
        train_rmse_hist.append(train_rmse)
        train_rmse_mag_hist.append(train_rmse_mag)
        train_rmse_ang_hist_deg.append(train_rmse_ang_deg)

        # -------- validation ------------------------------------------
        if epoch % VAL_EVERY == 0 or epoch == EPOCHS:
            val_loss, val_mse, val_mse_mag, val_mse_ang = \
                run_epoch(val_loader, train=False, pinn=PINN)

            val_rmse = math.sqrt(val_mse)
            val_rmse_mag = math.sqrt(val_mse_mag)
            val_rmse_ang_deg = math.sqrt(val_mse_ang) * (180.0 / math.pi)

            val_loss_hist.append(val_loss)
            val_rmse_hist.append(val_rmse)
            val_rmse_mag_hist.append(val_rmse_mag)
            val_rmse_ang_hist_deg.append(val_rmse_ang_deg)

            print(f"Epoch {epoch:3d} | "
                  f"train loss {train_loss:.4e}  rmse {train_rmse:.4e} "
                  f"(mag {train_rmse_mag:.4e}, ang {train_rmse_ang_deg:.4e}°) | "
                  f"valid loss {val_loss:.4e}  rmse {val_rmse:.4e} "
                  f"(mag {val_rmse_mag:.4e}, ang {val_rmse_ang_deg:.4e}°) | "
                  f"time {time.time()-t0:.2f}s")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = f"./results/ckpt/{RUNNAME}_{EPOCHS}_best_model.ckpt"
                torch.save(model.state_dict(), ckpt_path)
                print(f"  ↳ checkpoint saved to {ckpt_path}")
        else:
            print(f"Epoch {epoch:3d} | "
                  f"train loss {train_loss:.4e}  rmse {train_rmse:.4e} "
                  f"(mag {train_rmse_mag:.4e}, ang {train_rmse_ang_deg:.4e}°) | "
                  f"time {time.time()-t0:.2f}s")



    # -----------------------------------------------------------------
    # 6.  Curves
    # -----------------------------------------------------------------
    import matplotlib.pyplot as plt, os
    os.makedirs("./results/plots", exist_ok=True)

    epochs = range(1, len(train_loss_hist) + 1)

    if PINN:
        plt.figure(figsize=(6,4))
        plt.plot(epochs, train_loss_hist, label="Train Physics Loss")
        plt.plot(epochs[:len(val_loss_hist)], val_loss_hist,
                 label="Validation Physics Loss")
        plt.yscale("log"); plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.title("PINN: Physics Loss"); plt.legend(); plt.tight_layout()
        plt.savefig(f"./results/plots/{RUNNAME}_physics_loss.png"); plt.clf()

    # --- RMSE total ---------------------------------------------------
    plt.figure(figsize=(6,4))
    plt.plot(epochs, train_rmse_hist, label="Train RMSE (phasor/all)")
    plt.plot(epochs[:len(val_rmse_hist)], val_rmse_hist,
             label="Val RMSE (phasor/all)")
    plt.yscale("log"); plt.xlabel("Epoch"); plt.ylabel("RMSE")
    plt.title("Supervised RMSE"); plt.legend(); plt.tight_layout()
    plt.savefig(f"./results/plots/{RUNNAME}_rmse_total.png"); plt.clf()

    # --- Magnitude vs Angle RMSE --------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(10,4))

    ax[0].plot(epochs, train_rmse_mag_hist, label="Train |V|")
    ax[0].plot(epochs[:len(val_rmse_mag_hist)], val_rmse_mag_hist,
               label="Val |V|")
    ax[0].set_title("Magnitude RMSE"); ax[0].set_yscale("log")
    ax[0].set_xlabel("Epoch"); ax[0].legend()

    ax[1].plot(epochs, train_rmse_ang_hist_deg, label="Train θ (deg)")
    ax[1].plot(epochs[:len(val_rmse_ang_hist_deg)], val_rmse_ang_hist_deg,
               label="Val θ (deg)")
    ax[1].set_title("Angle RMSE (degrees)"); ax[1].set_yscale("log")
    ax[1].set_xlabel("Epoch"); ax[1].legend()

    fig.suptitle("Magnitude vs Angle RMSE"); fig.tight_layout()
    fig.savefig(f"./results/plots/{RUNNAME}_rmse_components.png"); plt.close(fig)

# ------------------------------------------------------------------
# 7.  Final test evaluation
# ------------------------------------------------------------------
if "test" in args.mode:
    # best_ckpt_path = f"./results/ckpt/{RUNNAME}_{EPOCHS}_best_model.ckpt"
    # print(f"\nLoading checkpoint from: {best_ckpt_path}")
    # model.load_state_dict(torch.load(best_ckpt_path))

    test_loss, test_mse, test_mse_mag, test_mse_ang = \
        run_epoch(test_loader, train=False, pinn=PINN)

    test_rmse = math.sqrt(test_mse)
    test_rmse_mag = math.sqrt(test_mse_mag)
    test_rmse_ang_deg = math.sqrt(test_mse_ang) * (180.0 / math.pi)

    if PINN:
        print(
            f"\nTest physics-loss : {test_loss:.4e}"
            f" | total RMSE : {test_rmse:.4e}"
            f" | |V| RMSE : {test_rmse_mag:.4e}"
            f" | θ RMSE : {test_rmse_ang_deg:.4e}°"
        )
    else:
        print(
            f"\nFinal test-set RMSE : {test_rmse:.4e}"
            f" (|V|: {test_rmse_mag:.4e}, θ: {test_rmse_ang_deg:.4e}°)"
        )
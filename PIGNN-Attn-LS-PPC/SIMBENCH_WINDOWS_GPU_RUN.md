# Running the SimBench PIGNN-Attn-LS Model on a Windows GPU Machine

This guide explains how to train and evaluate the SimBench snapshot-envelope
manual-flat model locally on a single Windows machine with an NVIDIA GPU.

The recommended configuration is the best SimBench run from the June 2026
experiment:

- model: `GNSMsg_EdgeSelfAttn`
- dataset: `SimBench_snapshot_envelope_manual_flat_100MVA_coupled_ppcY_manual_flat_siNR_36000_NR_branchrows_directSI.parquet`
- buses / branches: 94 buses, 131 branches
- loss: PINN physics loss only
- Armijo mode: restored fixed mode
- best attention depth: `num_attn_layers=10`
- voltage-level feature: disabled, i.e. no `--vn_feat`

## 1. Repository Layout

Run commands from the PPC directory:

```powershell
cd C:\path\to\PIGNN-Attn-LS\PIGNN-Attn-LS-PPC
```

Expected important files:

```text
train_valid_test.py
GNSMsg_SelfAttention_armijo.py
Dataset_optimized_complex_columns.py
collate_blockdiag_optimized_complex_columns.py
read_npy_columns_optimized.py
helper.py
```

Put the SimBench parquet somewhere local, for example:

```text
C:\data\SimBench_snapshot_envelope_manual_flat_100MVA_coupled_ppcY_manual_flat_siNR_36000_NR_branchrows_directSI.parquet
```

## 2. Download Data and Stored Checkpoints

The SimBench parquet, trained checkpoints, and training logs are also stored on:

```text
chkim@131.188.35.62
```

You must be connected to the FAU university VPN before using `scp`.

On the server, the parquets are available under:

```text
~/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out/
```

Verified parquet filenames:

```text
SimBench_snapshot_envelope_manual_flat_coupled_ppcY_manual_flat_siNR_36000_NR_branchrows_directSI.parquet
SimBench_snapshot_envelope_manual_flat_100MVA_coupled_ppcY_manual_flat_siNR_36000_NR_branchrows_directSI.parquet
```

For this guide, use the `100MVA` parquet:

```powershell
New-Item -ItemType Directory -Force C:\data | Out-Null

scp chkim@131.188.35.62:~/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out/SimBench_snapshot_envelope_manual_flat_100MVA_coupled_ppcY_manual_flat_siNR_36000_NR_branchrows_directSI.parquet C:\data\
```

The trained SimBench checkpoints and logs have been uploaded to:

```text
~/PIGNN-Attn-LS/ScenarioSynthesis_PPC/results/ckpt/simbench_manual_flat_fixed_armijo_20260615_190832/
~/PIGNN-Attn-LS/ScenarioSynthesis_PPC/results/logs/simbench_manual_flat_fixed_armijo_20260615_190832/
```

Download them locally:

```powershell
New-Item -ItemType Directory -Force .\results\ckpt | Out-Null
New-Item -ItemType Directory -Force .\results\logs | Out-Null

scp -r chkim@131.188.35.62:~/PIGNN-Attn-LS/ScenarioSynthesis_PPC/results/ckpt/simbench_manual_flat_fixed_armijo_20260615_190832 .\results\ckpt\

scp -r chkim@131.188.35.62:~/PIGNN-Attn-LS/ScenarioSynthesis_PPC/results/logs/simbench_manual_flat_fixed_armijo_20260615_190832 .\results\logs\
```

The best checkpoint is:

```text
.\results\ckpt\simbench_manual_flat_fixed_armijo_20260615_190832\simbench_flat_fix_pinn_b16_lr3e-6_K40_dhi16_h8_attn10_novn_40_best_model.ckpt
```

The matching training log is:

```text
.\results\logs\simbench_manual_flat_fixed_armijo_20260615_190832\simbench_flat_fix_pinn_b16_lr3e-6_K40_dhi16_h8_attn10_novn_training_log.txt
```

If `scp` asks for a password, use the server account password provided by the
project owner.

## 3. Python Environment

Recommended:

- Windows 10/11
- NVIDIA GPU with CUDA support
- Python 3.10 or 3.11
- A CUDA-enabled PyTorch install

Create and activate a virtual environment:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

Install PyTorch. Example for CUDA 12.1:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install `torch_scatter`. The URL must match your PyTorch and CUDA version.
For PyTorch `2.4.x` with CUDA 12.1, for example:

```powershell
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
```

If your PyTorch version is different, replace the wheel URL accordingly, e.g.
`torch-2.5.0+cu121.html`, `torch-2.6.0+cu121.html`, etc.

Install the remaining packages:

```powershell
pip install numpy pandas pyarrow matplotlib
```

Quick environment check:

```powershell
python -c "import torch, numpy, pandas, pyarrow, torch_scatter; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'); print('numpy:', numpy.__version__); print('pandas:', pandas.__version__); print('pyarrow:', pyarrow.__version__)"
```

## 4. Use the Stored Checkpoint

After downloading the checkpoint directory above, evaluate the stored best model
without retraining:

```powershell
python -u train_valid_test.py `
  --mode test `
  --EPOCHS=40 `
  --BATCH=16 `
  --LR=3e-6 `
  --seed_value=42 `
  --train_ratio=0.3333 `
  --valid_ratio=0.3333 `
  --lr_scheduler=CosineAnnealingLR `
  --PARQUET "C:\data\SimBench_snapshot_envelope_manual_flat_100MVA_coupled_ppcY_manual_flat_siNR_36000_NR_branchrows_directSI.parquet" `
  --PER_UNIT `
  --target_S_base 1e8 `
  --share_grid `
  --dataset_complex_dtype complex128 `
  --vlimit `
  --model GNSMsg_EdgeSelfAttn `
  --armijo_mode=fixed `
  --armijo_rho=0.5 `
  --armijo_max_backtracks=5 `
  --armijo_min_alpha=0.0625 `
  --PINN `
  --log_to_file `
  --log_dir ".\results\logs\simbench_local_eval_attn10_novn" `
  --ckpt_dir ".\results\ckpt\simbench_manual_flat_fixed_armijo_20260615_190832" `
  --run_name simbench_flat_fix_pinn_b16_lr3e-6_K40_dhi16_h8_attn10_novn `
  --d=4 `
  --d_hi=16 `
  --n_heads=8 `
  --num_attn_layers=10 `
  --K=40 `
  --use_armijo
```

Important: `train_valid_test.py` finds the checkpoint by combining
`--ckpt_dir`, `--run_name`, and `--EPOCHS`. Therefore these three values must
match the stored checkpoint filename:

```text
simbench_flat_fix_pinn_b16_lr3e-6_K40_dhi16_h8_attn10_novn_40_best_model.ckpt
```

## 5. Recommended Training Command

This is the best-performing SimBench setting from the finished batch:

```powershell
python -u train_valid_test.py `
  --EPOCHS=40 `
  --BATCH=16 `
  --LR=3e-6 `
  --seed_value=42 `
  --train_ratio=0.3333 `
  --valid_ratio=0.3333 `
  --lr_scheduler=CosineAnnealingLR `
  --PARQUET "C:\data\SimBench_snapshot_envelope_manual_flat_100MVA_coupled_ppcY_manual_flat_siNR_36000_NR_branchrows_directSI.parquet" `
  --PER_UNIT `
  --target_S_base 1e8 `
  --share_grid `
  --dataset_complex_dtype complex128 `
  --vlimit `
  --model GNSMsg_EdgeSelfAttn `
  --armijo_mode=fixed `
  --armijo_rho=0.5 `
  --armijo_max_backtracks=5 `
  --armijo_min_alpha=0.0625 `
  --PINN `
  --log_to_file `
  --log_dir ".\results\logs\simbench_local_attn10_novn" `
  --ckpt_dir ".\results\ckpt\simbench_local_attn10_novn" `
  --run_name simbench_flat_fix_pinn_b16_lr3e-6_K40_dhi16_h8_attn10_novn `
  --d=4 `
  --d_hi=16 `
  --n_heads=8 `
  --num_attn_layers=10 `
  --K=40 `
  --use_armijo
```

The script will:

1. split the 36,000 cases into train / validation / test,
2. train for 40 epochs,
3. save the best validation checkpoint under `.\results\ckpt\...`,
4. reload the best checkpoint,
5. report final test metrics.

## 6. Model Configuration

| Item | Value |
|---|---:|
| model | `GNSMsg_EdgeSelfAttn` |
| graph input | branch-row direct-SI parquet |
| per-unit mode | enabled |
| target S base | `1e8 VA` = 100 MVA |
| dataset complex dtype | `complex128` |
| shared grid cache | enabled via `--share_grid` |
| loss | PINN physics loss only |
| MSE loss weight | `0.0` |
| K iterative steps | `40` |
| hidden size `d_hi` | `16` |
| node/message dimension `d` | `4` |
| attention heads | `8` |
| attention layers | `10` |
| Armijo mode | `fixed` |
| Armijo rho | `0.5` |
| Armijo max backtracks | `5` |
| Armijo min alpha | `0.0625` |
| batch size | `16` |
| learning rate | `3e-6` |
| epochs | `40` |
| scheduler | `CosineAnnealingLR` |
| voltage magnitude limit | enabled via `--vlimit` |
| `vn_feat` | disabled for best run |

## 7. Smaller-GPU Options

If your GPU runs out of memory, lower the batch size first:

```powershell
--BATCH=8
```

If it is still too slow, test the smaller attention-depth version:

```powershell
--num_attn_layers=4
```

The `attn4` run is faster, but it was less accurate than `attn10` in this
experiment.

## 8. Completed SimBench Results

All results below used:

```text
K=40, d=4, d_hi=16, n_heads=8, BATCH=16, LR=3e-6,
PINN only, fixed Armijo, dataset_complex_dtype=complex128,
PER_UNIT target_S_base=100 MVA, share_grid=True
```

### Validation Metrics at Epoch 40

| Case | Val \|V\| RMSE | Val theta RMSE | Val Delta P inf | Val Delta Q inf | mean \|Delta P\| | mean \|Delta Q\| | p95 \|Delta P\| | p95 \|Delta Q\| |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| attn10 no-vn | 4.938e-2 | 9.156 deg | 3.304e-1 | 1.886e-1 | 1.053e-1 | 2.497e-2 | 2.967e-1 | 8.796e-2 |
| attn8 no-vn | 6.795e-2 | 15.043 deg | 5.167e-1 | 3.133e-1 | 1.884e-1 | 5.456e-2 | 4.564e-1 | 1.921e-1 |
| attn4 no-vn | 3.972e-2 | 16.300 deg | 8.353e-1 | 1.368e-1 | 2.160e-1 | 3.993e-2 | 4.858e-1 | 1.113e-1 |
| attn8 vn | 2.520e-2 | 17.348 deg | 8.807e-1 | 6.239e-1 | 2.505e-1 | 8.171e-2 | 6.019e-1 | 2.089e-1 |
| attn10 vn | 4.537e-2 | 18.121 deg | 1.031e0 | 4.890e-1 | 2.688e-1 | 6.041e-2 | 8.473e-1 | 2.143e-1 |
| attn4 vn | 6.602e-2 | 17.874 deg | 1.194e0 | 7.232e-1 | 2.747e-1 | 6.831e-2 | 8.032e-1 | 2.651e-1 |

### Test Metrics

| Case | Test \|V\| RMSE | Test theta RMSE | Test Delta P inf | Test Delta Q inf | mean \|Delta P\| | mean \|Delta Q\| | p95 \|Delta P\| | p95 \|Delta Q\| |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| attn10 no-vn | 4.938e-2 | 9.159 deg | 3.302e-1 | 1.908e-1 | 1.052e-1 | 2.499e-2 | 2.967e-1 | 8.798e-2 |
| attn8 no-vn | 6.831e-2 | 15.047 deg | 5.171e-1 | 3.219e-1 | 1.884e-1 | 5.510e-2 | 4.568e-1 | 1.941e-1 |
| attn8 vn | 3.945e-2 | 15.727 deg | 7.788e-1 | 3.065e-1 | 2.268e-1 | 5.257e-2 | 5.178e-1 | 1.744e-1 |
| attn4 no-vn | 3.971e-2 | 16.301 deg | 8.354e-1 | 1.365e-1 | 2.160e-1 | 3.990e-2 | 4.858e-1 | 1.112e-1 |
| attn10 vn | 4.058e-2 | 17.811 deg | 1.047e0 | 5.282e-1 | 2.655e-1 | 6.209e-2 | 8.223e-1 | 2.075e-1 |
| attn4 vn | 6.607e-2 | 17.869 deg | 1.194e0 | 7.197e-1 | 2.747e-1 | 6.872e-2 | 8.041e-1 | 2.654e-1 |

## 9. Interpretation

The best setting is `attn10 no-vn`.

It has the best overall physical residual profile:

- lowest test `Delta P inf`: `3.302e-1 pu`
- low test `Delta Q inf`: `1.908e-1 pu`
- lowest mean active-power residual: `1.052e-1 pu`
- lowest p95 active-power residual: `2.967e-1 pu`
- best angle RMSE: `9.159 deg`

The `vn_feat` variants were worse for this SimBench run. That is different
from the LVN Heo1 experiments, where voltage-level information was important.
For this smaller 94-bus SimBench grid, the extra voltage nominal feature did
not help under the PINN-only fixed-Armijo recipe.

## 10. Output Files

With the recommended command, logs and checkpoints are written to:

```text
.\results\logs\simbench_local_attn10_novn
.\results\ckpt\simbench_local_attn10_novn
```

The best checkpoint filename follows:

```text
simbench_flat_fix_pinn_b16_lr3e-6_K40_dhi16_h8_attn10_novn_40_best_model.ckpt
```

At the end of training, `train_valid_test.py` automatically reloads this best
checkpoint and reports test metrics.

## 11. Common Issues

### `ModuleNotFoundError: torch_scatter`

Install the `torch-scatter` wheel that exactly matches your PyTorch and CUDA
version. The package is not usually installed by plain `pip install
torch-scatter` on Windows.

### CUDA is not detected

Check:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

If it prints `False`, reinstall PyTorch with a CUDA wheel.

### Out of memory

Use:

```powershell
--BATCH=8
```

or, for a quick smoke test:

```powershell
--max_train_samples=512 --max_valid_samples=512 --max_test_samples=512 --EPOCHS=2
```

### Training is slow

The recommended run uses `complex128` for accurate residual metrics. If you
only want a quick development test, use:

```powershell
--dataset_complex_dtype complex64
```

For final reporting, use `complex128`.

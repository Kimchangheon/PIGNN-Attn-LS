# PIGNN-Attn-LS / Dataset: LVN 

The default config trains the `GNSMsg_EdgeSelfAttn_VnFeat` variant — same
backbone as PIGNN-Attn-LS plus a per-bus `vn_log` scalar so the model can
distinguish 3 kV / 20 kV / 110 kV / 380 kV buses in multi-voltage LVN grids.

## Layout

```
.
├── configs/
│   └── lvn_pignn_attn_ls.yaml      # YAML config (LVN VnFeat recipe)
├── data_loading/                   # ChanghunDataset, collator, samplers
├── datasets/                       # drop LVN parquet here (see datasets/README.md)
├── models/
│   ├── registry.py
│   └── edge_selfattn/              # PIGNN-Attn-LS implementation
├── scripts/
│   ├── convert_lvn_to_hvn_schema.py  # raw LVN snapshot -> HVN-schema parquet
│   └── train_lvn_pignn_attn_ls.sh    # convenience launcher
├── train/                          # CLI, loop, metrics, MLflow plumbing
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate

# 1. Install torch + torch_scatter matched to your CUDA toolchain.
#    Example (CUDA 12.1):
pip install torch==2.4.* --index-url https://download.pytorch.org/whl/cu121
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

# 2. Install the rest.
pip install -r requirements.txt

# 3. Install this bundle so `python -m train` resolves the packages.
pip install -e .
```

CPU-only works too — swap the index-URL above for the CPU wheel index.

## Data prep

See `datasets/README.md`. Either drop a pre-converted
`LVN_converted_n36000_v2.parquet` into `datasets/`, or run
`scripts/convert_lvn_to_hvn_schema.py` against the raw LVN snapshot.

## Train

```bash
# Recommended: YAML-driven launch
python -m train --config configs/lvn_pignn_attn_ls.yaml

# Or use the wrapper (forwards extra args)
./scripts/train_lvn_pignn_attn_ls.sh

# Override anything on the CLI — values beat the YAML
python -m train --config configs/lvn_pignn_attn_ls.yaml \
    --EPOCHS 60 --BATCH 64 --seed_value 43
```

The recipe baked into the YAML reproduces the documented production run:

| flag                | value                                           |
|---------------------|-------------------------------------------------|
| model               | `GNSMsg_EdgeSelfAttn_VnFeat`                    |
| d / d_hi            | 4 / 16                                          |
| K                   | 10                                              |
| num_attn_layers     | 1                                               |
| DthetaMax / DvmFrac | 0.30 / 0.10                                     |
| batch / epochs / lr | 32 / 30 / 1e-4                                  |
| split               | ratio (0.8 / 0.1 / 0.1), seed 42                |
| loss                | `mag_ang_mse` + physics residual (`PINN`)       |

## Outputs

Each run writes to `results/runs/<timestamp>_<slug>/`:

- `train.log`              -- full stdout
- `artifacts/history.csv`  -- per-epoch loss / RMSE (mag, ang) (written at end of run)
- `ckpt/best.ckpt`         -- best checkpoint by `val_rmse_mag + val_rmse_ang_deg`
- `artifacts/code_snapshot.zip` -- frozen copy of code used
- `plots/*.png`            -- loss curves

Flip `mlflow.enabled: true` in the YAML (or pass `--mlflow`) to also stream
metrics + the run directory into MLflow.

## Provenance

Sourced from `SimpleGNN/` commit `4bdaa1e` of the
`feat/iem-framework` branch. Only the `edge_selfattn` model package is
included; `pe_deq_pf` / `hyperdeq_pf_pilot` from the parent repo are not
needed and are excluded.

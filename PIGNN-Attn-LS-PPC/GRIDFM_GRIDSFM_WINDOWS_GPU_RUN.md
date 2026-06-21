# Running GridFM and GridSFM on SimBench, LVN Heo1, and ENTSO-E

This guide explains how to reproduce the June 2026 GridFM/GridSFM experiments
with the existing PPC parquet data loader and complex128 residual evaluation.
It covers SimBench, LVN Heo1, and the 6,051-bus ENTSO-E RealGridTest case,
and points to the packaged checkpoints
and logs that were copied from Alex.

The important idea is that the data pipeline is unchanged:

- `Dataset_optimized_complex_columns.py`
- `collate_blockdiag_optimized_complex_columns.py`
- branch-row direct-SI parquet input
- per-unit rebasing to 100 MVA with `--target_S_base 1e8`
- complex128 residual metrics with `--dataset_complex_dtype complex128`

Only the surrogate model is changed:

- GridFM: `train_valid_test_gridfm.py`
- GridSFM: `train_valid_test_gridsfm.py`

## 1. Local Repository Layout

Run commands from the PPC directory:

```powershell
cd C:\path\to\PIGNN-Attn-LS\PIGNN-Attn-LS-PPC
```

Expected important files:

```text
train_valid_test_gridfm.py
train_valid_test_gridsfm.py
Dataset_optimized_complex_columns.py
collate_blockdiag_optimized_complex_columns.py
```

GridFM/GridSFM dependencies also need their corresponding packages installed:

```text
gridfm_graphkit
gridsfm
torch
torch_geometric
torch_scatter
numpy
pandas
pyarrow
```

On Alex these experiments used:

```text
/home/hpc/iwi5/iwi5295h/conda-envs/gridfm-py312/bin/python
```

For Windows, a CUDA-enabled Python 3.12 conda environment is recommended. If
native Windows PyG wheels are difficult, use WSL2 with an NVIDIA CUDA-enabled
PyTorch install.

## 2. Parquet Inputs

Put the parquets somewhere local, for example:

```text
C:\data\
```

Recommended parquets:

```text
SimBench_snapshot_envelope_manual_flat_100MVA_coupled_ppcY_manual_flat_siNR_36000_NR_branchrows_directSI.parquet
SimBench_snapshot_envelope_dc_compile_100MVA_coupled_ppcY_dc_compile_siNR_36000_NR_branchrows_directSI.parquet
LVN_snapshot_envelope_manual_flat_coupled_ppcY_manual_flat_siNR_36000_NR_branchrows_directSI.parquet
LVN_snapshot_envelope_dc_compile_coupled_ppcY_dc_compile_siNR_36000_NR_branchrows_directSI.parquet
ENTSO_E_RealGridTest_flat_start_ppcY_A_manual_flat_siNR_36000_NR_branchrows_directSI.parquet
ENTSO_E_RealGridTest_dc_init_ppcY_A_dc_compile_siNR_36000_NR_branchrows_directSI.parquet
```

The two ENTSO-E parquets each contain 36,000 converged cases on a fixed grid
with 6,051 buses and 8,202 branches. Their stored base is 1 MVA; the commands
below intentionally rebase them to 100 MVA with `--target_S_base 1e8`.

On the transfer server, parquets are expected under:

```text
~/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out/
```

Example download:

```powershell
New-Item -ItemType Directory -Force C:\data | Out-Null

scp chkim@131.188.35.62:~/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out/SimBench_snapshot_envelope_dc_compile_100MVA_coupled_ppcY_dc_compile_siNR_36000_NR_branchrows_directSI.parquet C:\data\
```

## 3. Packaged Checkpoints and Logs

The best available checkpoints and all matching logs were packaged locally under:

```text
results\ckpt\gridfm_gridsfm_best_20260617\
results\logs\gridfm_gridsfm_best_20260617\
```

They were also uploaded to the transfer server under:

```text
~/PIGNN-Attn-LS/ScenarioSynthesis_PPC/results/ckpt/gridfm_gridsfm_best_20260617/
~/PIGNN-Attn-LS/ScenarioSynthesis_PPC/results/logs/gridfm_gridsfm_best_20260617/
~/PIGNN-Attn-LS/ScenarioSynthesis_PPC/results/ckpt/entsoe_gridfm_best_20260620/
~/PIGNN-Attn-LS/ScenarioSynthesis_PPC/results/logs/entsoe_gridfm_best_20260620/
```

Download them on a Windows machine:

```powershell
New-Item -ItemType Directory -Force .\results\ckpt | Out-Null
New-Item -ItemType Directory -Force .\results\logs | Out-Null

scp -r chkim@131.188.35.62:~/PIGNN-Attn-LS/ScenarioSynthesis_PPC/results/ckpt/gridfm_gridsfm_best_20260617 .\results\ckpt\

scp -r chkim@131.188.35.62:~/PIGNN-Attn-LS/ScenarioSynthesis_PPC/results/logs/gridfm_gridsfm_best_20260617 .\results\logs\

scp -r chkim@131.188.35.62:~/PIGNN-Attn-LS/ScenarioSynthesis_PPC/results/ckpt/entsoe_gridfm_best_20260620 .\results\ckpt\

scp -r chkim@131.188.35.62:~/PIGNN-Attn-LS/ScenarioSynthesis_PPC/results/logs/entsoe_gridfm_best_20260620 .\results\logs\
```

Available checkpoint groups:

| Group | Available best checkpoints |
|---|---|
| `lvn_gridfm_py312_ft_a10080_20260615_195321` | GridFM LVN DC pretrain, flat scratch, flat fine-tuned from DC |
| `lvn_gridsfm_a10080_20260616_0945` | GridSFM LVN pre-FT flat/DC, scratch flat |
| `simbench_gridfm_gridsfm_a40_20260616_1025` | GridFM SimBench flat, GridSFM SimBench flat pre-FT/scratch |
| `simbench_dc_gridfm_gridsfm_a40_20260616_200625` | GridFM SimBench DC |
| `entsoe_gridfm_best_20260620` | GridFM ENTSO-E flat/DC scratch, best checkpoints after 12 epochs |

Note: the LVN GridSFM scratch-DC log is present, but its checkpoint was not in
the copied-back checkpoint directory at packaging time.

## 4. Result Summary

Metrics are per-unit on a 100 MVA base. Multiply residuals by 100 for MW/MVAr.
Rows marked `best valid` are validation snapshots; `final test` rows are
completed test-set evaluations.

| Grid | Model | Run | Row | RMSE | theta deg | dPinf | dQinf | mean P | mean Q | p95 P | p95 Q |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LVN Heo1 | GridFM | py312 DC pretrain | final test | 2.204e-2 | 0.176 | 3.071 | 11.023 | 2.28e-2 | 8.14e-2 | 3.85e-2 | 2.88e-1 |
| LVN Heo1 | GridFM | py312 flat FT from DC | final test | 2.457e-2 | 0.841 | 4.086 | 142.52 | 5.43e-2 | 2.55e-1 | 1.54e-1 | 1.48e-1 |
| LVN Heo1 | GridFM | py312 flat scratch | final test | 2.571e-2 | 0.914 | 1202 | 5237 | 1.713 | 7.481 | 1.43e-1 | 2.05e-1 |
| LVN Heo1 | GridSFM | scratch DC | best valid ep25 | 3.551e-2 | 1.549 | 3.907 | 1.887 | 3.63e-2 | 5.76e-2 | 8.07e-2 | 3.25e-1 |
| LVN Heo1 | GridSFM | scratch flat | best valid ep38 | 3.663e-2 | 1.706 | 3.418 | 1.869 | 3.76e-2 | 5.27e-2 | 8.19e-2 | 2.57e-1 |
| LVN Heo1 | GridSFM | pretrained FT DC | best valid ep37 | 6.542e-2 | 3.707 | 4.605 | 0.979 | 3.27e-2 | 3.59e-2 | 7.63e-2 | 1.62e-1 |
| SimBench | GridFM | DC scratch | final test | 6.172e-4 | 0.020 | 1.388e-2 | 3.070e-2 | 1.59e-3 | 4.07e-3 | 5.49e-3 | 1.36e-2 |
| SimBench | GridFM | flat scratch | final test | 1.741e-3 | 0.093 | 4.952e-2 | 2.341e-2 | 8.73e-3 | 4.46e-3 | 2.90e-2 | 1.55e-2 |
| SimBench | GridSFM | flat pre-FT | final test | 2.869e-1 | 16.434 | 8.714 | 0.782 | 6.47e-1 | 1.48e-1 | 2.842 | 4.58e-1 |
| SimBench | GridSFM | flat scratch | final test | 4.068e-1 | 23.304 | 3.795 | 0.491 | 3.79e-1 | 1.17e-1 | 1.302 | 3.04e-1 |
| SimBench | GridSFM | DC pre-FT | best valid ep10 | 4.144e-1 | 23.737 | 3.795 | 0.445 | 4.08e-1 | 1.18e-1 | 1.336 | 2.91e-1 |
| SimBench | GridSFM | DC scratch | best valid ep1 | 4.059e-1 | 23.213 | 3.967 | 0.471 | 3.80e-1 | 6.38e-2 | 1.350 | 2.27e-1 |
| ENTSO-E | GridFM | flat scratch, 12/40 epochs | best valid ep9 | 4.107e-2 | 2.318 | 17.70 | 15.35 | 3.04e-1 | 1.33e-1 | 1.078 | 3.99e-1 |
| ENTSO-E | GridFM | DC scratch, 12/40 epochs | best valid ep12 | 1.858e-2 | 1.000 | 12.62 | 12.48 | 2.08e-1 | 1.01e-1 | 7.41e-1 | 2.87e-1 |
| ENTSO-E | GridFM | flat scratch, best checkpoint | final test | 4.155e-2 | 2.346 | 17.74 | 15.43 | 3.04e-1 | 1.33e-1 | 1.080 | 4.00e-1 |
| ENTSO-E | GridFM | DC scratch, best checkpoint | final test | 1.853e-2 | 0.997 | 12.66 | 12.54 | 2.07e-1 | 1.01e-1 | 7.37e-1 | 2.89e-1 |

Best current takeaways:

- Best SimBench result: GridFM DC scratch.
- Best completed LVN GridFM result: GridFM py312 DC pretrain.
- Best low-residual LVN GridSFM snapshot: GridSFM scratch flat/DC or pre-FT DC,
  depending whether active or reactive infinity residual is prioritized.
- GridSFM reduces residuals after training, but its voltage-angle RMSE remains
  much worse than GridFM on both SimBench and LVN Heo1.
- The ENTSO-E jobs reached the 24-hour walltime after epoch 12, so they are
  partial training results rather than converged 40-epoch comparisons. DC
  initialization was clearly stronger than flat initialization at that point.

## 5. GridFM Training Command

Use this template for either SimBench or LVN by changing `--PARQUET`,
`--run_name`, `--log_dir`, and `--ckpt_dir`.

```powershell
python -u train_valid_test_gridfm.py `
  --PARQUET "C:\data\SimBench_snapshot_envelope_dc_compile_100MVA_coupled_ppcY_dc_compile_siNR_36000_NR_branchrows_directSI.parquet" `
  --run_name gridfm312_simbench_dc_scratch_L12_h48_nh8_b8 `
  --log_to_file --log_dir ".\results\logs\gridfm312_simbench_dc_scratch" `
  --ckpt_dir ".\results\ckpt\gridfm312_simbench_dc_scratch" `
  --PER_UNIT --target_S_base 1e8 --share_grid --lazy_parquet --row_group_cache_size 2 `
  --dataset_complex_dtype complex128 `
  --BATCH 8 --EPOCHS 40 --LR 5e-4 `
  --train_ratio 0.3333 --valid_ratio 0.3333 --seed_value 42 `
  --hidden_size 48 --num_layers 12 --n_heads 8 `
  --zero_init_head --vn_feature_mode log --feature_transform signed_log `
  --mse_weight 1.0 --physics_weight 1e-2 `
  --physics_loss_form logcosh --VAL_EVERY 1
```

To warm-start GridFM from a saved checkpoint:

```powershell
python -u train_valid_test_gridfm.py `
  --PARQUET "C:\data\LVN_snapshot_envelope_manual_flat_coupled_ppcY_manual_flat_siNR_36000_NR_branchrows_directSI.parquet" `
  --run_name gridfm312_flat_ft_from_dc_L12_h48_nh8_b8 `
  --init_checkpoint ".\results\ckpt\gridfm_gridsfm_best_20260617\lvn_gridfm_py312_ft_a10080_20260615_195321\gridfm312_dc_pretrain_L12_h48_nh8_b8_best.pt" `
  --log_to_file --log_dir ".\results\logs\gridfm_lvn_flat_ft" `
  --ckpt_dir ".\results\ckpt\gridfm_lvn_flat_ft" `
  --PER_UNIT --target_S_base 1e8 --share_grid --lazy_parquet --row_group_cache_size 2 `
  --dataset_complex_dtype complex128 `
  --BATCH 8 --EPOCHS 40 --LR 5e-4 `
  --train_ratio 0.3333 --valid_ratio 0.3333 --seed_value 42 `
  --hidden_size 48 --num_layers 12 --n_heads 8 `
  --zero_init_head --vn_feature_mode log --feature_transform signed_log `
  --mse_weight 1.0 --physics_weight 1e-2 `
  --physics_loss_form logcosh --VAL_EVERY 1
```

## 6. GridSFM Training Commands

GridSFM uses batch size 4 in the current experiments.

Pretrained fine-tuning:

```powershell
python -u train_valid_test_gridsfm.py `
  --PARQUET "C:\data\SimBench_snapshot_envelope_manual_flat_100MVA_coupled_ppcY_manual_flat_siNR_36000_NR_branchrows_directSI.parquet" `
  --run_name gridsfm_simbench_pre_ft_flat_b4 `
  --log_to_file --log_dir ".\results\logs\gridsfm_simbench_pre_ft_flat" `
  --ckpt_dir ".\results\ckpt\gridsfm_simbench_pre_ft_flat" `
  --PER_UNIT --target_S_base 1e8 --share_grid --lazy_parquet --row_group_cache_size 2 `
  --dataset_complex_dtype complex128 `
  --BATCH 4 --EPOCHS 40 --LR 1e-4 `
  --train_ratio 0.3333 --valid_ratio 0.3333 --seed_value 42 `
  --mse_weight 1.0 --physics_weight 1e-2 `
  --physics_loss_form logcosh --VAL_EVERY 1 `
  --treat_voltage_mismatch_as_transformer `
  --init_mode pretrained `
  --pretrained_checkpoint ".\checkpoints\gridsfm_open_v1.1.pt"
```

Scratch training:

```powershell
python -u train_valid_test_gridsfm.py `
  --PARQUET "C:\data\LVN_snapshot_envelope_manual_flat_coupled_ppcY_manual_flat_siNR_36000_NR_branchrows_directSI.parquet" `
  --run_name gridsfm_scratch_flat_b4 `
  --log_to_file --log_dir ".\results\logs\gridsfm_lvn_scratch_flat" `
  --ckpt_dir ".\results\ckpt\gridsfm_lvn_scratch_flat" `
  --PER_UNIT --target_S_base 1e8 --share_grid --lazy_parquet --row_group_cache_size 2 `
  --dataset_complex_dtype complex128 `
  --BATCH 4 --EPOCHS 40 --LR 1e-4 `
  --train_ratio 0.3333 --valid_ratio 0.3333 --seed_value 42 `
  --mse_weight 1.0 --physics_weight 1e-2 `
  --physics_loss_form logcosh --VAL_EVERY 1 `
  --treat_voltage_mismatch_as_transformer `
  --init_mode scratch
```

## 7. Recommended Runs

For SimBench:

1. GridFM DC scratch: best current result.
2. GridFM flat scratch: strong flat-start baseline.
3. GridSFM flat scratch/pretrained fine-tune only as comparison baselines.

For LVN Heo1:

1. GridFM DC pretrain: best completed balanced GridFM result.
2. GridFM flat fine-tune from DC: useful warm-start comparison.
3. GridSFM scratch flat/DC: useful low-residual validation snapshots, but angle
   error remains larger than GridFM.

For ENTSO-E RealGridTest:

1. Prefer the DC-start GridFM checkpoint: it has the best voltage and residual
   metrics of the two partial runs.
2. Use `BATCH 1` as the conservative reproducible setting for the 6,051-bus
   graph. Larger batches should be profiled before committing a long job.
3. Split with seed 42 and ratios 0.3333/0.3333 gives 11,998 training, 11,998
   validation, and 12,004 test cases.

Example ENTSO-E DC-start command:

```powershell
python -u train_valid_test_gridfm.py `
  --PARQUET "C:\data\ENTSO_E_RealGridTest_dc_init_ppcY_A_dc_compile_siNR_36000_NR_branchrows_directSI.parquet" `
  --run_name gridfm312_entsoe_dc_scratch_L12_h48_nh8_b1_sbase100M `
  --log_to_file --log_dir ".\results\logs\entsoe_gridfm_dc" `
  --ckpt_dir ".\results\ckpt\entsoe_gridfm_dc" `
  --PER_UNIT --target_S_base 1e8 --share_grid --lazy_parquet --row_group_cache_size 1 `
  --dataset_complex_dtype complex128 `
  --BATCH 1 --EPOCHS 40 --LR 5e-4 `
  --train_ratio 0.3333 --valid_ratio 0.3333 --seed_value 42 `
  --hidden_size 48 --num_layers 12 --n_heads 8 `
  --zero_init_head --vn_feature_mode log --feature_transform signed_log `
  --mse_weight 1.0 --physics_weight 1e-2 `
  --physics_loss_form logcosh --VAL_EVERY 1
```

## 8. Notes on Checkpoint Reuse

GridFM supports direct checkpoint warm-start through `--init_checkpoint`.

GridSFM currently trains/evaluates in one run and reloads the best checkpoint at
the end of that run. The saved GridSFM checkpoints are included for archival and
future evaluation, but direct checkpoint-only evaluation would need adding an
`--init_checkpoint` argument to `train_valid_test_gridsfm.py`, similar to the
GridFM script.

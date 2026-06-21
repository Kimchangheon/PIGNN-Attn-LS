#!/bin/bash -l
#SBATCH --job-name=gridfm312_entsoe_flat_best_test
#SBATCH --output=/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/gridfm312_entsoe_flat_best_test.out
#SBATCH --error=/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/gridfm312_entsoe_flat_best_test.err
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

set -euo pipefail
: "${TMPDIR:?TMPDIR is not set by Slurm}"

BASE=/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
PYTHON=/home/hpc/iwi5/iwi5295h/conda-envs/gridfm-py312/bin/python
LOG_DIR=${BASE}/results/logs/entsoe_realgrid_gridfm_test_20260620
LOCAL_PARQUET=${TMPDIR}/gridfm312_entsoe_flat_best_test_${SLURM_JOB_ID}.parquet
LOCAL_CKPT_DIR=${TMPDIR}/gridfm312_entsoe_flat_best_test_ckpt_${SLURM_JOB_ID}
export PYTHONPATH=${BASE}:${PYTHONPATH:-}
mkdir -p "${LOG_DIR}" "${LOCAL_CKPT_DIR}"
cd "${BASE}"

cp /home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out/ENTSO_E_RealGridTest_flat_start_ppcY_A_manual_flat_siNR_36000_NR_branchrows_directSI.parquet "${LOCAL_PARQUET}"

srun "${PYTHON}" -u train_valid_test_gridfm.py \
  --PARQUET "${LOCAL_PARQUET}" \
  --run_name gridfm312_entsoe_flat_best_test \
  --log_to_file --log_dir "${LOG_DIR}" \
  --ckpt_dir "${LOCAL_CKPT_DIR}" \
  --init_checkpoint /home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/entsoe_realgrid_gridfm_a10080_20260618_154511/gridfm312_entsoe_flat_scratch_L12_h48_nh8_b1_sbase100M_best.pt --init_strict \
  --PER_UNIT --target_S_base 1e8 --share_grid --lazy_parquet --row_group_cache_size 1 \
  --dataset_complex_dtype complex128 \
  --BATCH 1 --EPOCHS 0 --LR 5e-4 \
  --train_ratio 0.3333 --valid_ratio 0.3333 --max_valid_samples 1 --seed_value 42 \
  --hidden_size 48 --num_layers 12 --n_heads 8 \
  --zero_init_head --vn_feature_mode log --feature_transform signed_log \
  --mse_weight 1.0 --physics_weight 1e-2 --physics_loss_form logcosh

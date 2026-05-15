#!/usr/bin/env bash
set -euo pipefail

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
ALEX_BASE="${ALEX_BASE:-/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC}"
PARQUET="${PARQUET:-/home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out/LVN_snapshot_envelope_chunk_1/LVN_snapshot_envelope_coupled_ppcY_dc_compile_siNR_2000_NR_branchrows_directSI.parquet}"
REMOTE_SBATCH_DIR="${ALEX_BASE}/sbatch/lvn_chunk_geometric_${STAMP}"
REMOTE_LOG_DIR="${ALEX_BASE}/results/logs/lvn_chunk_geometric_${STAMP}"
REMOTE_JOB_OUT_DIR="${ALEX_BASE}/sbatch/Job_out"
MANIFEST="${REMOTE_LOG_DIR}/manifest.tsv"
LABEL_PREFIX="${LABEL_PREFIX:-}"

EPOCHS="${EPOCHS:-40}"
BATCH="${BATCH:-8}"
LR="${LR:-3e-6}"
SEED="${SEED:-42}"
ROW_GROUP_CACHE_SIZE="${ROW_GROUP_CACHE_SIZE:-2}"
PARTITION="${PARTITION:-a100}"
GPU_TYPE="${GPU_TYPE:-a100}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
ARMIJO_RHO="${ARMIJO_RHO:-0.5}"
ARMIJO_MAX_BACKTRACKS="${ARMIJO_MAX_BACKTRACKS:-60}"
ARMIJO_MIN_ALPHA="${ARMIJO_MIN_ALPHA:-0.0}"

mkdir -p "${REMOTE_SBATCH_DIR}" "${REMOTE_LOG_DIR}" "${REMOTE_JOB_OUT_DIR}"
printf "label\tjob_id\tremote_script\tremote_log_dir\tjob_out\tjob_err\n" > "${MANIFEST}"

labels=(
  01_nhead8_dhi16_attn8_armijo
  02_nhead8_dhi16_attn10_armijo
  03_nhead8_dhi24_attn8_noarmijo
  04_nhead8_dhi24_attn8_armijo
  05_nhead8_dhi24_attn10_noarmijo
  06_nhead8_dhi24_attn10_armijo
)

args=(
  "--d=4 --d_hi=16 --n_heads=8 --num_attn_layers=8 --K=40 --use_armijo"
  "--d=4 --d_hi=16 --n_heads=8 --num_attn_layers=10 --K=40 --use_armijo"
  "--d=4 --d_hi=24 --n_heads=8 --num_attn_layers=8 --K=40"
  "--d=4 --d_hi=24 --n_heads=8 --num_attn_layers=8 --K=40 --use_armijo"
  "--d=4 --d_hi=24 --n_heads=8 --num_attn_layers=10 --K=40"
  "--d=4 --d_hi=24 --n_heads=8 --num_attn_layers=10 --K=40 --use_armijo"
)

echo "Dispatching ${#labels[@]} LVN chunk geometric experiments"
echo "PARQUET=${PARQUET}"
echo "REMOTE_LOG_DIR=${REMOTE_LOG_DIR}"
echo "LR=${LR}"
echo "ARMIJO_RHO=${ARMIJO_RHO}"
echo "ARMIJO_MAX_BACKTRACKS=${ARMIJO_MAX_BACKTRACKS}"
echo "ARMIJO_MIN_ALPHA=${ARMIJO_MIN_ALPHA}"

for idx in "${!labels[@]}"; do
  label="${LABEL_PREFIX}${labels[$idx]}"
  read -r -a extra_args <<< "${args[$idx]}"

  printf -v extra_args_joined ' %q' "${extra_args[@]}"
  printf -v parquet_quoted '%q' "${PARQUET}"
  printf -v base_quoted '%q' "${ALEX_BASE}"
  printf -v log_dir_quoted '%q' "${REMOTE_LOG_DIR}"

  script_path="${REMOTE_SBATCH_DIR}/${label}.sh"
  cat > "${script_path}" <<EOF2
#!/bin/bash -l
#SBATCH --job-name=${label}
#SBATCH --output=${REMOTE_JOB_OUT_DIR}/${label}.out
#SBATCH --error=${REMOTE_JOB_OUT_DIR}/${label}.err
#SBATCH --gres=gpu:${GPU_TYPE}:1
#SBATCH --time=${TIME_LIMIT}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --partition=${PARTITION}

set -euo pipefail

export PYTHONPATH=${base_quoted}:\${PYTHONPATH:-}
module load python
cd ${base_quoted}

srun python train_valid_test.py --EPOCHS=${EPOCHS} --BATCH=${BATCH} --LR=${LR} --seed_value=${SEED} --train_ratio=0.3333 --valid_ratio=0.3333 --lr_scheduler=CosineAnnealingLR --PARQUET ${parquet_quoted} --vlimit --model GNSMsg_EdgeSelfAttn --lazy_parquet --row_group_cache_size=${ROW_GROUP_CACHE_SIZE} --armijo_mode=geometric --armijo_rho=${ARMIJO_RHO} --armijo_max_backtracks=${ARMIJO_MAX_BACKTRACKS} --armijo_min_alpha=${ARMIJO_MIN_ALPHA} --log_to_file --log_dir ${log_dir_quoted}${extra_args_joined}
EOF2

  chmod +x "${script_path}"
  response="$(sbatch "${script_path}")"
  job_id="$(printf '%s\n' "${response}" | awk '/Submitted batch job/ {print $4; exit}')"
  echo "${response}"
  echo "script=${script_path}"
  echo "job_out=${REMOTE_JOB_OUT_DIR}/${label}.out"
  echo "job_err=${REMOTE_JOB_OUT_DIR}/${label}.err"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${label}" "${job_id}" "${script_path}" "${REMOTE_LOG_DIR}" "${REMOTE_JOB_OUT_DIR}/${label}.out" "${REMOTE_JOB_OUT_DIR}/${label}.err" >> "${MANIFEST}"
done

echo "Manifest: ${MANIFEST}"

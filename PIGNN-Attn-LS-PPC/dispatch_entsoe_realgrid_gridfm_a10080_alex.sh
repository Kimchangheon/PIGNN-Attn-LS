#!/usr/bin/env bash
set -euo pipefail

ALEX_HOST="${ALEX_HOST:-alex}"
SSH_CONFIG_FILE="${SSH_CONFIG_FILE:-}"
ALEX_BASE="${ALEX_BASE:-/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC}"
VAULT_BASE="${VAULT_BASE:-/home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC}"
PYTHON_BIN="${PYTHON_BIN:-/home/hpc/iwi5/iwi5295h/conda-envs/gridfm-py312/bin/python}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
GROUP="${GROUP:-entsoe_realgrid_gridfm_a10080_${STAMP}}"
REMOTE_SBATCH_DIR="${ALEX_BASE}/sbatch/${GROUP}"
REMOTE_LOG_DIR="${ALEX_BASE}/results/logs/${GROUP}"
REMOTE_JOB_OUT_DIR="${ALEX_BASE}/sbatch/Job_out"
REMOTE_CKPT_DIR="${VAULT_BASE}/results/ckpt/${GROUP}"
LOCAL_SBATCH_DIR="${ALEX_BASE_LOCAL:-$(pwd)}/sbatch/${GROUP}"

PARQUET_FLAT="${PARQUET_FLAT:-/home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out/ENTSO_E_RealGridTest_flat_start_ppcY_A_manual_flat_siNR_36000_NR_branchrows_directSI.parquet}"
PARQUET_DC="${PARQUET_DC:-/home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out/ENTSO_E_RealGridTest_dc_init_ppcY_A_dc_compile_siNR_36000_NR_branchrows_directSI.parquet}"

BATCH="${BATCH:-1}"
EPOCHS="${EPOCHS:-40}"
LR="${LR:-5e-4}"
PHYSICS_WEIGHT="${PHYSICS_WEIGHT:-1e-2}"
MSE_WEIGHT="${MSE_WEIGHT:-1.0}"
HIDDEN_SIZE="${HIDDEN_SIZE:-48}"
NUM_LAYERS="${NUM_LAYERS:-12}"
N_HEADS="${N_HEADS:-8}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
PARTITION="${PARTITION:-a100}"
GPU_TYPE="${GPU_TYPE:-a100}"
SBATCH_CONSTRAINT="${SBATCH_CONSTRAINT:-a100_80}"
TARGET_S_BASE="${TARGET_S_BASE:-1e8}"
ROW_GROUP_CACHE_SIZE="${ROW_GROUP_CACHE_SIZE:-1}"

SSH_CMD=(ssh)
SCP_CMD=(scp)
if [[ -n "${SSH_CONFIG_FILE}" ]]; then
  SSH_CMD+=(-F "${SSH_CONFIG_FILE}")
  SCP_CMD+=(-F "${SSH_CONFIG_FILE}")
fi

run_ssh() {
  local attempt
  for attempt in 1 2 3 4 5; do
    if "${SSH_CMD[@]}" "${ALEX_HOST}" "$@"; then
      return 0
    fi
    echo "[ssh] attempt ${attempt} failed; retrying..." >&2
    sleep 20
  done
  return 1
}

run_scp() {
  local attempt
  for attempt in 1 2 3 4 5; do
    if "${SCP_CMD[@]}" "$@"; then
      return 0
    fi
    echo "[scp] attempt ${attempt} failed; retrying..." >&2
    sleep 20
  done
  return 1
}

echo "GROUP=${GROUP}"
echo "ALEX_HOST=${ALEX_HOST}"
if [[ -n "${SSH_CONFIG_FILE}" ]]; then
  echo "SSH_CONFIG_FILE=${SSH_CONFIG_FILE}"
fi
echo "PARQUET_FLAT=${PARQUET_FLAT}"
echo "PARQUET_DC=${PARQUET_DC}"
echo "BATCH=${BATCH}"
echo "TARGET_S_BASE=${TARGET_S_BASE}"
echo "REMOTE_LOG_DIR=${REMOTE_LOG_DIR}"
echo "REMOTE_CKPT_DIR=${REMOTE_CKPT_DIR}"

mkdir -p "${LOCAL_SBATCH_DIR}"

make_gridfm_job() {
  local label="$1"
  local parquet="$2"
  local local_script="${LOCAL_SBATCH_DIR}/${label}.sh"
  cat > "${local_script}" <<EOF
#!/bin/bash -l
#SBATCH --job-name=${label}
#SBATCH --output=${REMOTE_JOB_OUT_DIR}/${label}.out
#SBATCH --error=${REMOTE_JOB_OUT_DIR}/${label}.err
#SBATCH --gres=gpu:${GPU_TYPE}:1
#SBATCH --time=${TIME_LIMIT}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --partition=${PARTITION}
#SBATCH --constraint=${SBATCH_CONSTRAINT}

set -euo pipefail
: "\${TMPDIR:?TMPDIR is not set by Slurm}"

export PYTHONPATH=${ALEX_BASE}:\${PYTHONPATH:-}
PYTHON_CMD=${PYTHON_BIN}
cd ${ALEX_BASE}

LOCAL_PARQUET="\${TMPDIR}/${label}_\${SLURM_JOB_ID}.parquet"
LOCAL_CKPT_DIR="\${TMPDIR}/${label}_ckpt_\${SLURM_JOB_ID}"
mkdir -p "\${LOCAL_CKPT_DIR}" "${REMOTE_LOG_DIR}" "${REMOTE_CKPT_DIR}"

copy_back_outputs() {
  status=\$?
  mkdir -p "${REMOTE_CKPT_DIR}"
  cp -a "\${LOCAL_CKPT_DIR}/." "${REMOTE_CKPT_DIR}/" 2>/dev/null || true
  exit \${status}
}
trap copy_back_outputs EXIT

"\${PYTHON_CMD}" - <<'PY'
import torch, torch_geometric, torch_scatter, gridfm_graphkit
print("[env] imports ok:", "torch", torch.__version__, "pyg", torch_geometric.__version__)
PY

echo "[stage] copying parquet to \${LOCAL_PARQUET}"
stage_start=\$(date +%s)
cp ${parquet} "\${LOCAL_PARQUET}"
stage_end=\$(date +%s)
echo "[stage] parquet staging complete in \$((stage_end - stage_start)) seconds"

srun "\${PYTHON_CMD}" -u train_valid_test_gridfm.py \\
  --PARQUET "\${LOCAL_PARQUET}" \\
  --run_name ${label} \\
  --log_to_file --log_dir ${REMOTE_LOG_DIR} \\
  --ckpt_dir "\${LOCAL_CKPT_DIR}" \\
  --PER_UNIT --target_S_base ${TARGET_S_BASE} --share_grid --lazy_parquet --row_group_cache_size ${ROW_GROUP_CACHE_SIZE} \\
  --dataset_complex_dtype complex128 \\
  --BATCH ${BATCH} --EPOCHS ${EPOCHS} --LR ${LR} \\
  --train_ratio 0.3333 --valid_ratio 0.3333 --seed_value 42 \\
  --hidden_size ${HIDDEN_SIZE} --num_layers ${NUM_LAYERS} --n_heads ${N_HEADS} \\
  --zero_init_head --vn_feature_mode log --feature_transform signed_log \\
  --mse_weight ${MSE_WEIGHT} --physics_weight ${PHYSICS_WEIGHT} \\
  --physics_loss_form logcosh --VAL_EVERY 1
EOF
  chmod +x "${local_script}"
}

FLAT_LABEL="gridfm312_entsoe_flat_scratch_L${NUM_LAYERS}_h${HIDDEN_SIZE}_nh${N_HEADS}_b${BATCH}_sbase100M"
DC_LABEL="gridfm312_entsoe_dc_scratch_L${NUM_LAYERS}_h${HIDDEN_SIZE}_nh${N_HEADS}_b${BATCH}_sbase100M"

make_gridfm_job "${FLAT_LABEL}" "${PARQUET_FLAT}"
make_gridfm_job "${DC_LABEL}" "${PARQUET_DC}"

SUBMIT_LOCAL="${LOCAL_SBATCH_DIR}/submit_all.sh"
cat > "${SUBMIT_LOCAL}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
DIR='${REMOTE_SBATCH_DIR}'
for script in '${FLAT_LABEL}.sh' '${DC_LABEL}.sh'; do
  jid=\$(sbatch --parsable "\${DIR}/\${script}")
  printf '%s\t%s\n' "\${jid}" "\${script%.sh}"
done
EOF
chmod +x "${SUBMIT_LOCAL}"

run_ssh "mkdir -p '${REMOTE_SBATCH_DIR}' '${REMOTE_LOG_DIR}' '${REMOTE_JOB_OUT_DIR}' '${REMOTE_CKPT_DIR}'"
run_scp train_valid_test_gridfm.py collate_blockdiag_optimized_complex_columns.py "${ALEX_HOST}:${ALEX_BASE}/"
run_scp "${LOCAL_SBATCH_DIR}"/*.sh "${ALEX_HOST}:${REMOTE_SBATCH_DIR}/"

echo "Submitted jobs:"
run_ssh "bash '${REMOTE_SBATCH_DIR}/submit_all.sh'"
echo "Logs: ${REMOTE_LOG_DIR}"
echo "Checkpoints: ${REMOTE_CKPT_DIR}"

#!/bin/bash -l
#SBATCH --job-name=case30_A
#SBATCH --output=./Job_out/case30_A.out
#SBATCH --error=./Job_out/case30_A.err
#SBATCH --partition=singlenode              # Meggie queues: devel | work | big (big = by request)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72            # 20 physical cores per node on Meggie
#SBATCH --time=24:00:00
#SBATCH --export=NONE                 # follow Meggie examples

# Meggie recommends unsetting the default env export
unset SLURM_EXPORT_ENV

# (Optional) load your Python/solver modules here
module load python
conda activate
conda activate $WORK/conda-envs/Python3.12_PandaPower3.4.0
# module load petsc        # if you use PETSc via mpi4py

# Your project environment
export PYTHONPATH=/home/hpc/iwi5/iwi5295h/master_master:$PYTHONPATH
cd ..

# Run your application strictly on CPU, binding ranks/threads to cores
srun python main_datagen_multiproc_improved.py \
  --preset case30 \
  --ybus_mode ppcY \
  --runs 36000 \
  --rows_per_task 1500 \
  --save_steps 3000 \
  --no_save_y_matrix \
  --scenario_level A \
  --K 40 \
  --use_force_shunt_when_no_trafo \
  --start_mode dc_compile \
  --diagnose_nr \
  --convergence_mode misinf \
  --save_path /home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out \
  --overwrite

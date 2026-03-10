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
# module load petsc        # if you use PETSc via mpi4py

# Your project environment
export PYTHONPATH=/home/hpc/iwi5/iwi5295h/master_master:$PYTHONPATH
cd ..

# Run your application strictly on CPU, binding ranks/threads to cores
srun python main_datagen_multiproc_improved.py \
  --preset case30 \
  --ybus_mode ppcY \
  --runs 36000 \
  --workers 0 \
  --rows_per_task 2000 \
  --save_steps 6000 \
  --jitter_load 0.10 \
  --pv_vset_lo 0.98 \
  --pv_vset_hi 1.04 \
  --rand_u_start \
  --angle_jitter_deg 5 \
  --mag_jitter_pq 0.02 \
  --save_path ./out \
  --overwrite

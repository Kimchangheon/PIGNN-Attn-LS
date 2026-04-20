#!/bin/bash

#SBATCH --job-name=case14_36000_d4_dhi16_h4_attn1_k40_NR_plain
#SBATCH --output=./Job_out/case14_36000_d4_dhi16_h4_attn1_k40_NR_plain.out
#SBATCH --gres=gpu:a40:1                 # Request 1 A40 GPU
#SBATCH --time=24:00:00                  # Request 24 hours of runtime
#SBATCH --ntasks=1                       # Run a single task
#SBATCH --cpus-per-task=16               # Request 16 CPU coresk99k9vs
#SBATCH --partition=a40                  # Use the A40 partition
#SBATCH --error=./Job_out/case14_36000_d4_dhi16_h4_attn1_k40_NR_plain.err

export PYTHONPATH=/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC:$PYTHONPATH
module load python

cd ..

# Run your application or script
srun python train_valid_test.py --d=4 --d_hi=16 --num_attn_layers=1 --n_heads=4 --K=40 --EPOCHS=100 --lr_scheduler=CosineAnnealingLR --PARQUET ../ScenarioSynthesis_PPC/out/case145_ppcY_A_dc_compile_siNR_36000_NR_branchrows_directSI.parquet --vlimit --model GNSMsg_EdgeSelfAttn
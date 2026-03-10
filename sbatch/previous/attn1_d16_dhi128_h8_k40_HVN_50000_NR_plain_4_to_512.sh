#!/bin/bash

#SBATCH --job-name=attn1_d16_dhi128_h8_k40_HVN_50000_NR_plain_4_to_512
#SBATCH --output=./Job_out/attn1_d16_dhi128_h8_k40_HVN_50000_NR_plain_4_to_512.out
#SBATCH --gres=gpu:a100:1 -C a100_80                 # Request 1 A40 GPU
#SBATCH --time=24:00:00                  # Request 24 hours of runtime
#SBATCH --ntasks=1                       # Run a single task
#SBATCH --cpus-per-task=16               # Request 16 CPU coresk99k9vs
#SBATCH --partition=a100                  # Use the A40 partition
#SBATCH --error=./Job_out/attn1_d16_dhi128_h8_k40_HVN_50000_NR_plain_4_to_512.err

export PYTHONPATH=/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS:$PYTHONPATH
module load python

cd ..

# Run your application or script
srun python train_valid_test.py --d=16 --d_hi=128 --n_heads=8 --num_attn_layers=1 --K=40 --EPOCHS=100 --lr_scheduler=CosineAnnealingLR --PARQUET /home/woody/iwi5/iwi5295h/large/HVN_50000_NR_plain_4_to_512_buses.parquet --vlimit --use_armijo --model GNSMsg_EdgeSelfAttn
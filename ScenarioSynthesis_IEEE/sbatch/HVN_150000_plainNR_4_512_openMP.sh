#!/bin/bash -l
#SBATCH --job-name=HVN_150000_plainNR_4_512_openMP
#SBATCH --output=./Job_out/HVN_150000_plainNR_4_512_openMP.out
#SBATCH --error=./Job_out/HVN_150000_plainNR_4_512_openMP.err
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
srun python main_datagen_multiproc_improved.py --no_save_y_matrix --runs 150000 --save_steps 1000 --rows_per_task 200 --min_number_of_buses 4 --max_number_of_buses 512 --gridtype HVN --save_path /home/woody/iwi5/iwi5295h/large
python main_datagen_multiproc_improved.py --no_save_y_matrix --runs 15000 --save_steps 1000 --rows_per_task 200 --min_number_of_buses 4 --max_number_of_buses 32 --gridtype HVN --save_path /home/woody/iwi5/iwi5295h/large

#python main_datagen_multiproc_improved.py --runs 1500 --save_steps 100 --rows_per_task 20 --min_number_of_buses 4 --max_number_of_buses 512 --gridtype HVN --save_path /home/woody/iwi5/iwi5295h/large


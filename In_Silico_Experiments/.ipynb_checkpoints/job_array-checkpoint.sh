#!/bin/bash
#SBATCH --output=logs/experiment_%A_%a.out # Standard output log (%A - job ID, %a - array task ID)
#SBATCH --error=logs/experiment_%A_%a.err  # Standard error log
#SBATCH --nodes=1                          # One node per job
#SBATCH --ntasks=1                         # One task per job
#SBATCH --cpus-per-task=4                  # Number of CPU cores per task (adjust as needed)
#SBATCH --mem=16G                         # Memory per node (adjust as needed)
#SBATCH --time=03:20:00                    # Time limit (adjust as needed)
#SBATCH --gres=gpu:1                       # Request 1 GPU per job
#SBATCH --array=1-2268%100            

# Get the parameters for this task from args.txt
PARAMS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" args.txt)

# Run the experiment with the parameters for this task
python3 run_experiment.py $PARAMS


#!/bin/bash

#SBATCH --job-name=individualized_calibration_for_language
#SBATCH --account=nlp
#SBATCH --partition=nlp

#SBATCH -N 1                         # number of minimum nodes
#SBATCH --gres=gpu:4                 # Request n gpus
#SBATCH --cpus-per-task=20

#SBATCH -o sweeps/slurm_%N_%j_out.txt       # stdout goes here
#SBATCH -e sweeps/slurm_%N_%j_err.txt       # stderr goes here
#SBATCH --mail-type=fail                    # send email if job fails
#SBATCH --mail-user=zachary@campus.technion.ac.il

# Get the sweep name as a parameter
SWEEP_NAME=$1

source activate individualized_calibration_for_language
echo "Running sweep $SWEEP_NAME"

wandb login
export WANDB_SERVICE_WAIT=300


# Run the sweep
wandb agent "$SWEEP_NAME"
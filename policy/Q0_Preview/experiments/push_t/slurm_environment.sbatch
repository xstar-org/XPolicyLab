#!/bin/bash -l
#SBATCH --job-name=q0-sol-push-t
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=7-00:00:00
set -euo pipefail
umask 077
: "${Q0_EXPERIMENT_ROOT:?Set a new absolute experiment root}"
: "${Q0_EXPERIMENT_CODE:?Set the experiment script directory}"
: "${Q0_SIM_PYTHON:?Set the existing Isaac Sim Python launcher}"
exec python3 -u "$Q0_EXPERIMENT_CODE/slurm_worker.py" --root "$Q0_EXPERIMENT_ROOT" \
    --sim-python "$Q0_SIM_PYTHON"

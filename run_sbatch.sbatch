#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH -N 1

#SBATCH --job-name=hmd_example
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

#SBATCH --partition edu-thesis
#SBATCH -t 0:15:0

HMD_ROOT="/data/hmd_2024/"
ENV_FILE="$HMD_ROOT/env_file"
VENV_PATH="$HMD_ROOT/venv"
PYTHON="$VENV_PATH/bin/python"

export HF_HOME=$HMD_ROOT

# script using during the labs
PYTHONPATH=$HMD_ROOT/code $PYTHON pipeline.py "$@"
#!/bin/bash
#SBATCH --job-name=re-smoke
#SBATCH --partition=rtx-devel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/kamarthi_v_neu/re_smoke_%j.out
#SBATCH --error=/scratch/kamarthi_v_neu/re_smoke_%j.err
set -euo pipefail
echo "node=$(hostname) job=$SLURM_JOB_ID"
~/envs/trope/bin/python ~/research-engine-smoke/score_box.py /scratch/kamarthi_v_neu/re_smoke_effect.npy
echo "SMOKE_DONE"

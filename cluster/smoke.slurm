#!/bin/bash
#SBATCH --job-name=re-smoke
#SBATCH --partition=rtx-devel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/work/neu/p2026_0016_neu/re_smoke_%j.out
#SBATCH --error=/work/neu/p2026_0016_neu/re_smoke_%j.err
set -euo pipefail
echo "node=$(hostname) job=$SLURM_JOB_ID"
~/envs/trope/bin/python ~/research-engine-smoke/score_box.py /work/neu/p2026_0016_neu/re_smoke_effect.npy
echo "SMOKE_DONE"

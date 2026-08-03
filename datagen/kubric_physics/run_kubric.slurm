#!/bin/bash
#SBATCH --job-name=kubric-gen
#SBATCH --partition=rtx-batch
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --account=p2026_0016_neu
#
# Paired possible/impossible physics-video generation via the kubruntu Apptainer image.
# Blender renders headless (background mode) with NO display, CPU Cycles for correctness.
#
# Usage: sbatch run_kubric.slurm <OUT_DIR> <SEEDS> [VIOLATIONS] [LIMIT] [SHEETS]
#   e.g. sbatch run_kubric.slurm /work/neu/p2026_0016_neu/kubric/out 0-19
#   SHEETS=1 also writes per-pair contact-sheet PNGs (human-perceptibility check).

set -euo pipefail
BASE=/work/neu/p2026_0016_neu/kubric
SIF=$BASE/kubruntu.sif
CODE=$BASE/datagen
OUT=${1:-$BASE/out}
SEEDS=${2:-0-19}
VIOL=${3:-solidity,permanence,immutability}
LIMIT=${4:-0}
SHEETS=${5:-0}
SHEETS_FLAG=""
[ "$SHEETS" = "1" ] && SHEETS_FLAG="--sheets"

mkdir -p "$BASE/logs" "$OUT"
cd "$BASE"

# Blender must never touch a display; force offscreen/background.
export DISPLAY=""
export APPTAINERENV_KUBRIC_USE_GPU=0

echo "host=$(hostname) sif=$SIF out=$OUT seeds=$SEEDS viol=$VIOL limit=$LIMIT sheets=$SHEETS"
# NO --nv: injecting host NVIDIA GL libs pulls libGLX requiring GLIBC_2.34, which the container's
# older GLIBC lacks, breaking the bpy import. CPU Cycles (KUBRIC_USE_GPU=0) needs no host GL, so the
# container's own libraries load cleanly. A GPU-Cycles pass later needs a GLIBC-matched image, not --nv.
# PYTHONPATH=$BASE/pylibs supplies imageio-ffmpeg (bundled static ffmpeg) for mp4 encoding.
apptainer exec \
  --bind "$BASE:$BASE" \
  --env PYTHONPATH="$BASE/pylibs" \
  "$SIF" \
  python3 "$CODE/kubric_physics/physics_gen.py" \
    --out "$OUT" --seeds "$SEEDS" --violations "$VIOL" --limit "$LIMIT" $SHEETS_FLAG

echo "EXIT=$?"

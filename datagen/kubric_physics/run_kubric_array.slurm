#!/bin/bash
#SBATCH --job-name=kubric-gen
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --account=p2026_0016_neu
#
# MAX-FIDELITY paired physics-video generation, parallelized across a SLURM array.
#
# WHY CPU + ARRAY (not GPU): the kubruntu image ships Blender 2.93, whose Cycles exposes
# only ('NONE','CUDA','OPENCL') compute types (no OptiX) and cannot enumerate the Blackwell
# (sm_120) GPU under CUDA. Even with the driver libcuda/optix libs bound cleanly (no --nv,
# so no GLIBC_2.34 break), the 2021 Blender falls back to CPU. GPU Cycles here needs a NEW
# container with Blender 4.x+ (recognizes sm_120 / OptiX). Until then, THROUGHPUT comes from
# fanning the plan across array tasks on the 128-core EPYC cpu nodes, where CPU Cycles is fast.
#
# ONE PAIR PER TASK: Kubric's bpy + PyBullet are process-global singletons that corrupt after
# ~2 scenes in one process, so each array task renders EXACTLY ONE (seed,violation) pair via
# --index $SLURM_ARRAY_TASK_ID. Set --array=0-(P-1) where P = len(seeds) * len(violations).
# For seeds 0-63 x 3 violations = 192 pairs -> --array=0-191%32 (32 concurrent).
#
# Usage: sbatch --array=0-(P-1)%32 run_kubric_array.slurm <OUT_DIR> <SEEDS> [VIOL] [RES] [SPP] [SHEETS]
#   e.g. sbatch --array=0-17 run_kubric_array.slurm /work/neu/p2026_0016_neu/kubric/dataset_v2 0-5
#        (6 seeds x 3 violations = 18 pairs). Merge after with merge_manifests.py.

set -euo pipefail
BASE=/work/neu/p2026_0016_neu/kubric
SIF=$BASE/kubruntu.sif
CODE=$BASE/datagen
OUT=${1:-$BASE/dataset_v2}
SEEDS=${2:-0-19}
VIOL=${3:-solidity,permanence,immutability}
RES=${4:-512}
SAMPLES=${5:-128}
SHEETS=${6:-1}
SHEETS_FLAG=""
[ "$SHEETS" = "1" ] && SHEETS_FLAG="--sheets"

IDX=${SLURM_ARRAY_TASK_ID:-0}

mkdir -p "$BASE/logs" "$OUT"
cd "$BASE"
export DISPLAY=""

echo "host=$(hostname) index=$IDX out=$OUT seeds=$SEEDS res=$RES spp=$SAMPLES"
# CPU Cycles: no --nv, no host GL. PYTHONPATH supplies imageio-ffmpeg for mp4 encode.
apptainer exec \
  --bind "/work/neu/p2026_0016_neu:/work/neu/p2026_0016_neu" \
  --env PYTHONPATH="$BASE/pylibs" \
  "$SIF" \
  python3 "$CODE/kubric_physics/physics_gen.py" \
    --out "$OUT" --seeds "$SEEDS" --violations "$VIOL" \
    --resolution "$RES" --samples "$SAMPLES" $SHEETS_FLAG \
    --index "$IDX"

echo "EXIT=$? index=$IDX done"

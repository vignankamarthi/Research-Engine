#!/bin/bash
#SBATCH --job-name=kubric-gpu-gen
#SBATCH --partition=rtx-batch
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --account=p2026_0016_neu
#
# GPU paired physics-video generation on the Blackwell RTX PRO 6000 via the modern
# Blender image (blender_gpu.sif, bpy 4.5 LTS). Replaces the CPU-only kubruntu path.
#
# Blender 4.5 Cycles enumerates sm_120 and renders on OptiX (confirmed 2026-08-02).
# Tuned throughput: ~2.5-4s/pair amortized (128spp, NO OptiX denoiser, 1k HDRI). The
# denoiser was the real bottleneck (~0.27s/frame fixed); off + 128spp matches the
# original CPU max-fidelity and is far faster. GPU needs --nv.
#
# BLOCK MODE (amortize the ~5s process startup): each array task renders a CONTIGUOUS
# BLOCK of BLOCK pairs in ONE warm process (bpy + pybullet reset between pairs, OptiX
# pipeline + world stay warm). Set --array=0-(ceil(P/BLOCK)-1) where P = len(seeds) *
# len(violations). A block of ~12-24 pairs is stable (validated 24/24, no corruption).
#
# Usage: sbatch --array=0-A run_kubric_gpu_array.slurm <OUT> <SEEDS> [VIOL] [RES] [SPP] [SHEETS] [BLOCK]
#   e.g. seeds 0-63 x 3 = 192 pairs, BLOCK=12 -> 16 tasks -> --array=0-15
#        sbatch --array=0-15 run_kubric_gpu_array.slurm \
#          /work/neu/p2026_0016_neu/kubric/dataset_gpu 0-63 solidity,permanence,immutability 512 128 1 12
# Merge after with merge_manifests.py (handles the manifest_block_*.json shards).

set -euo pipefail
BASE=/work/neu/p2026_0016_neu/kubric
SIF=$BASE/build/blender_gpu.sif
GEN=$BASE/build/physics_gen_bpy.py
OUT=${1:-$BASE/dataset_gpu}
SEEDS=${2:-0-19}
VIOL=${3:-solidity,permanence,immutability}
RES=${4:-512}
SAMPLES=${5:-128}
SHEETS=${6:-1}
BLOCK=${7:-12}
SHEETS_FLAG=""
[ "$SHEETS" = "1" ] && SHEETS_FLAG="--sheets"

TASK=${SLURM_ARRAY_TASK_ID:-0}
IDX_START=$((TASK * BLOCK))
IDX_END=$((IDX_START + BLOCK - 1))
mkdir -p "$BASE/logs" "$OUT"
cd "$BASE"

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) task=$TASK block=[$IDX_START..$IDX_END] out=$OUT res=$RES spp=$SAMPLES"
# --nv exposes the Blackwell driver/OptiX to Cycles. --device gpu refuses a silent CPU fallback.
# 1k HDRI (default) + no denoiser (default) are the tuned fast path; add --denoise to restore it.
apptainer exec --nv \
  --bind "/work/neu/p2026_0016_neu:/work/neu/p2026_0016_neu" \
  "$SIF" \
  python3.11 "$GEN" \
    --out "$OUT" --seeds "$SEEDS" --violations "$VIOL" \
    --resolution "$RES" --samples "$SAMPLES" $SHEETS_FLAG \
    --device gpu --index "$IDX_START" --index-end "$IDX_END"

echo "EXIT=$? task=$TASK block=[$IDX_START..$IDX_END] done"

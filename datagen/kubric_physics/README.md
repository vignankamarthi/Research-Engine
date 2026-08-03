# Kubric paired physics-video generator (max-fidelity)

Renders PAIRED possible-vs-impossible clips for the Research-Engine intuitive-physics
eval (Qwen2.5-VL-7B). One seeded scene yields two pixel-aligned clips that differ only
in a single injected violation. Un-memorizable by construction: regenerate with a new
seed and the assets, colours, positions, and camera-relative motion all change.

## Fidelity (upgraded 2026-08-01)

- REAL OBJECTS: Google Scanned Objects (GSO) meshes via `kb.AssetSource.from_manifest`
  (`gs://kubric-public/assets/GSO`), varied per seed from a vetted compact-object pool.
  Solidity + immutability use GSO meshes; immutability overrides the material to a solid
  colour so the colour jump is a clean identity cue. Permanence keeps a glossy PBR SPHERE
  ON PURPOSE (its violation needs PROVABLE full occlusion, which a tumbling arbitrary mesh
  cannot guarantee; a sphere of known radius can).
- PHOTOREAL LIGHTING + BACKGROUND: Poly Haven HDRI (`.../HDRI_haven`) as both image-based
  lighting (`_set_ambient_light_hdri`) and a real photographed backdrop (`_set_background_hdri`),
  one crisp key light for a defined contact shadow, and a PBR PrincipledBSDF floor.
- RESOLUTION + SAMPLES: 512x512, 128 Cycles samples per pixel (both CLI-overridable).
- Assets download once into a persistent cache (`--asset-cache`, default
  `/work/neu/p2026_0016_neu/kubric/asset_cache`) and are reused across runs.

## Engine = Kubric (Blender Cycles + PyBullet)

Physics (PyBullet) and rendering (Blender) are decoupled, so we compute one valid
rollout, then override exactly one attribute for the impossible twin and re-render with
identical assets, camera, lights, seed, and timing. Chosen over TDW/Genesis/etc. in
`../../../sim-engine-comparison.md` (headless-native, Apache-2.0, physics-video precedent).

## Violations

| type | possible | impossible (single injected event) | divergence |
|------|----------|------------------------------------|------------|
| `solidity` | object falls, rests on the floor | object free-falls THROUGH the floor | first floor contact |
| `permanence` | object rolls behind an occluder, re-emerges | object removed while fully hidden, never re-emerges | when it should re-emerge |
| `immutability` | object keeps its colour in flight | object's colour jumps mid-trajectory | the jump frame |

## Pixel-alignment guarantee

The impossible twin copies the possible per-frame keyframes verbatim up to the
divergence frame, then overrides ONE attribute (position for solidity/permanence,
material colour for immutability) from that frame on. Nothing else in the scene is
touched, so the pair is byte-identical before the event and differs only in the
manipulated attribute after. `divergence_frame` is recorded per pair in the manifest.

## Files

- `physics_gen.py` -- the generator (scene, rollout, violation injection, mp4 + manifest).
- `probe_api.py` -- one-shot introspection of the pinned kubric API in the container.
- `run_kubric_array.slurm` -- sbatch ARRAY runner (CPU nodes, one pair per task).
- `merge_manifests.py` -- merge per-pair manifests into `manifest.json`.
- `run_kubric.slurm` -- legacy single-process runner (kept; NOT for multi-pair, see below).

## GPU path (built + validated 2026-08-02)

The GPU blocker is RESOLVED. The `kubruntu` image ships **Blender 2.93** (2021), whose Cycles
has no OptiX and cannot enumerate the Blackwell (sm_120) GPU, so it renders on CPU. A NEW image
`blender_gpu.sif` (recipe `blender_gpu.def`, **bpy 4.5 LTS**) enumerates the RTX PRO 6000 via
OptiX and renders on the GPU:

- per-frame 512px/128spp: **GPU 0.27s vs CPU 3.0s (~11x)**. `render_device` is recorded per pair.
- **Tuned throughput: ~5s/pair amortized** (validated 24/24 in one process). Three wins, in order
  of impact:
  - The **OptiX denoiser was the real bottleneck**, about 0.27s/frame of fixed overhead (48spp with
    denoise is 0.33s/frame, 128spp with no denoise is 0.10s/frame). It is **OFF by default**. 128spp
    no-denoise matches the original CPU max-fidelity (which also had no denoiser) and stays clean at 512px.
  - The impossible twin renders **only its post-divergence frames** (about half) and splices the reused
    possible prefix onto them.
  - A **block of pairs runs in ONE warm process** (bpy + pybullet reset between pairs, the OptiX pipeline
    and world stay warm), so the ~5s process startup amortizes across the block.
  - Once the denoiser is off, samples and HDRI size barely matter (per-frame is dominated by scene sync
    plus the OptiX acceleration-structure refit, not sampling). The HDRI still downscales to 1k by
    default via `--hdri-long-edge`.
- GPU needs **`--nv`** (the explicit driver-lib bind does NOT enumerate on this image; the old
  GLIBC break that forced 2.93 off `--nv` is gone with the modern glibc base).
- Kubric pins Blender 2.93's bpy, so the GPU path drops the Kubric wrapper and rebuilds the SAME
  pipeline directly on **bpy 4.5 + pybullet** (`physics_gen_bpy.py`): PyBullet loads the GSO
  `object.urdf` for collision, bpy imports `visual_geometry.obj` + HDRI world for rendering. The
  violation logic, paired-invariant splice (prefix_absdiff = 0), manifest, and contact sheets are
  identical to `physics_gen.py`.

Files: `blender_gpu.def` (image recipe), `physics_gen_bpy.py` (bpy/pybullet generator),
`run_kubric_gpu_array.slurm` (GPU array runner), `populate_assets.sh` (fills the GSO/HDRI cache
from the public GCS tarballs over HTTPS). Build once on a cpu node
(`apptainer build --fakeroot blender_gpu.sif blender_gpu.def`). The CPU array
(`run_kubric_array.slurm` + `kubruntu.sif`) still works and is kept as a fallback.

Key validity note: an early version silently rendered on CPU because `read_factory_settings`
(the per-pair scene wipe) reset the Cycles GPU preferences after they were set. Fixed by
re-asserting the OptiX device inside `build_scene` after the wipe, with a hard check that refuses
a silent CPU fallback when `--device gpu` is requested.

## One pair per process (hard constraint)

Kubric's `bpy` + PyBullet are **process-global singletons** that corrupt after ~2 scenes in one
process (`pybullet.error: Not connected to physics server` on the 3rd pair). So multi-pair
generation runs **one pair per process** via `--index`, one SLURM array task per pair.

## Run (on AICR)

```bash
# 6 pairs (seeds 0-1 x 3 violations = 6): array 0-5, 512px, 128 spp, with contact sheets
sbatch --array=0-5 datagen/kubric_physics/run_kubric_array.slurm \
       /work/neu/p2026_0016_neu/kubric/dataset_v2 0-1 solidity,permanence,immutability 512 128 1
# after it finishes, merge the per-pair manifests:
apptainer exec --bind /work/neu/p2026_0016_neu:/work/neu/p2026_0016_neu kubruntu.sif \
  python3 datagen/kubric_physics/merge_manifests.py /work/neu/p2026_0016_neu/kubric/dataset_v2
# full run: P = len(seeds)*3 pairs -> --array=0-(P-1)%32  (e.g. seeds 0-63 -> 0-191%32)
```

Render cost: ~3 min per pair at 512px/128spp on a 32-core EPYC `cpu` node.
Blender renders headless (background mode, no display); Cycles runs on CPU inside the container.

## Manifest

`manifest.json` is a flat list, two records per pair:

```json
{"clip_path": "clips/solidity_seed0000_possible.mp4", "label": 1,
 "violation_type": "solidity", "seed": 0, "pair_id": "solidity_seed0000",
 "role": "possible", "num_frames": 16, "resolution": [512, 512],
 "divergence_frame": 12, "prefix_absdiff": 0.0, "suffix_absdiff": 1.43,
 "object": "UGG_Cambridge_Womens_Black_7", "hdri": "anniversary_lounge",
 "material": "gso_native", "samples": 128, "obj_color": [...]}
```

`label`: 1 = possible, 0 = impossible. `prefix_absdiff` is 0.0 by construction (the impossible
twin reuses the possible pixels before divergence). `material` is `gso_native` (solidity),
`gso_solid_override` (immutability), or `pbr_sphere` (permanence).

## How this maps onto the Research-Engine

Self-generated task, no external incumbent, so the claim-type classifier routes to
EFFECT (`RESEARCH-LOOP-SPEC.md`). Each violation type is a separately-carved, disjoint,
power-sized confirmation BOX of paired clips scored once. The paired possible/impossible
design is the cleanest effect stimulus (identical-except-the-event). The untrained-init
FLOOR runs paired on the SAME clips as the mandatory geometry-artifact catcher, and the
reported effect is the trained-minus-untrained residual. The MIE is anchored on the
human-vs-model gap: humans must score ~100% (the perceptibility validity gate), so the
MIE is a HIGH percentile of accepted violation-of-expectation deltas, well above the
untrained floor, signed by Vignan. IntPhys 2 stays the real-data generalization anchor.
Every violation type passes the human-perceptibility gate before any finding rests on it.

"""Paired possible/impossible physics-video generator (Kubric: Blender Cycles + PyBullet).

MAX-FIDELITY build. Real Google Scanned Objects (GSO) meshes, Poly Haven HDRI
environment lighting + a real photographed background, a PBR floor, 512x512 (or higher),
and enough Cycles samples to kill the noise. The paired possible/impossible design and the
pixel-aligned splice are preserved exactly.

For each (seed, violation_type) we build ONE seeded scene, compute a valid PyBullet
rollout, render the POSSIBLE clip, then inject a SINGLE violation into the SAME scene
(same camera, HDRI, object, seed, timing) and render the pixel-aligned IMPOSSIBLE twin.
The pair shares a byte-identical prefix and diverges only at the injected event.

Violations implemented:
  - solidity     : object free-falls THROUGH the solid floor instead of resting on it.
  - permanence   : object is removed while fully occluded, so it never re-emerges.
  - immutability : object's colour jumps mid-trajectory (identity change).

Object choice per violation (fidelity vs. invariant trade-off, stated plainly):
  - solidity     : a real GSO mesh (native scanned PBR texture), varied per seed.
  - immutability : a real GSO mesh with a solid PrincipledBSDF override so the colour
                   jump is a clean, unambiguous identity cue (silhouette stays a real object).
  - permanence   : a glossy PBR SPHERE (not a GSO mesh) ON PURPOSE. The violation semantics
                   demand PROVABLE full occlusion behind the occluder (the object must be
                   completely hidden when it is removed, else a human sees it pop out rather
                   than fail-to-re-emerge). An arbitrary tumbling scanned mesh cannot guarantee
                   that; a sphere of known radius can. It is still HDRI-lit and photoreal, not
                   the old flat-shaded toy.

Pixel-alignment guarantee (by construction, not by luck). The impossible twin REUSES the
possible clip's rendered frames verbatim up to the divergence frame `d`, then renders only
frames d..N-1 after overriding exactly one attribute (object position for solidity/permanence,
object colour for immutability). `prefix_absdiff` is therefore 0 by construction.

Run INSIDE the kubruntu Apptainer container (CPU Cycles, headless, no display):
  apptainer exec --bind <base>:<base> \
    --env PYTHONPATH=<base>/pylibs kubruntu.sif \
    python3 physics_gen.py --out <dir> --seeds 0-19 --sheets
"""

import argparse
import json
import os
import pathlib
import shutil
import tempfile

import numpy as np

import kubric as kb
from kubric.simulator import PyBullet
from kubric.renderer import Blender


# ----------------------------------------------------------------------------
# constants (deterministic; nothing here reads a global RNG)
# ----------------------------------------------------------------------------
RESOLUTION = (512, 512)         # >= 512 for real VLM engagement (override via --resolution)
NUM_FRAMES = 16
FRAME_START = 1
FRAME_RATE = 12
STEP_RATE = 240
GRAVITY = (0.0, 0.0, -9.81)
SAMPLES_PER_PIXEL = 128         # CPU Cycles; high enough to kill noise at 512 (override via --samples)
FLOOR_TOP_Z = 0.0               # world z of the top face of the floor slab

VIOLATIONS = ("solidity", "permanence", "immutability")
OCC_X = 0.0                     # occluder centre x (permanence)
OCC_HALF_W = 0.6                # occluder half-width in x (wide enough to hide the sphere)

# --- asset sources (Google Scanned Objects + Poly Haven HDRI, public GCS buckets) ---
GSO_MANIFEST = "gs://kubric-public/assets/GSO/GSO.json"
HDRI_MANIFEST = "gs://kubric-public/assets/HDRI_haven/HDRI_haven.json"
ASSET_CACHE = "/work/neu/p2026_0016_neu/kubric/asset_cache"   # persistent, reused across runs

# Vetted compact, physics-friendly GSO objects (bounds ~cubic, moderate face count,
# collision URDF present). Selected once from the GSO manifest metadata; varied per seed.
GSO_OBJECTS = [
    "Nickelodeon_Teenage_Mutant_Ninja_Turtles_Leonardo",
    "Digital_Camo_Double_Decker_Lunch_Bag",
    "Mad_Gab_Refresh_Card_Game",
    "Creatine_Monohydrate",
    "ACE_Coffee_Mug_Kristen_16_oz_cup",
    "BIA_Cordon_Bleu_White_Porcelain_Utensil_Holder_900028",
    "UGG_Cambridge_Womens_Black_7",
    "Sootheze_Cold_Therapy_Elephant",
    "30_CONSTRUCTION_SET",
    "Playmates_Industrial_CoSplinter_Teenage_Mutant_Ninja_Turtle_Action_Figure",
    "Olive_Kids_Birdie_Munch_n_Lunch",
    "Don_Franciscos_Gourmet_Coffee_Medium_Decaf_100_Colombian_12_oz_340_g",
    "Aroma_Stainless_Steel_Milk_Frother_2_Cup",
    "BlackBlack_Nintendo_3DSXL",
    "3D_Dollhouse_Swing",
    "Olive_Kids_Butterfly_Garden_Munch_n_Lunch_Bag",
    "IsoRich_Soy",
    "Central_Garden_Flower_Pot_Goo_425",
    "BALANCING_CACTUS",
    "Organic_Whey_Protein_Unflavored",
    "Circo_Fish_Toothbrush_Holder_14995988",
    "Cole_Hardware_Antislip_Surfacing_Material_White",
]

# Indoor / studio HDRIs: real photographed backgrounds + image-based lighting, varied per seed.
HDRI_ENVS = [
    "abandoned_games_room_01",
    "abandoned_workshop",
    "aerodynamics_workshop",
    "aft_lounge",
    "aircraft_workshop_01",
    "anniversary_lounge",
    "art_studio",
    "artist_workshop",
    "ballroom",
    "cayley_interior",
    "christmas_photo_studio_01",
    "abandoned_hall_01",
]

# module-level asset-source singletons (built once, reused across all pairs)
_GSO_SRC = None
_HDRI_SRC = None
_GSO_VALID = None
_HDRI_VALID = None


def _rng(seed):
    return np.random.RandomState(seed)


def _stable_source(manifest, cache_subdir):
    """Build an AssetSource whose local cache is a STABLE dir (assets download once, reuse)."""
    src = kb.AssetSource.from_manifest(manifest, scratch_dir=None)
    d = pathlib.Path(ASSET_CACHE) / cache_subdir
    d.mkdir(parents=True, exist_ok=True)
    src.local_dir = d                       # override mkdtemp() so the cache persists
    return src


def _ensure_sources():
    """Load GSO + HDRI sources once; validate our curated ids against the manifests.

    FAIL LOUD (ANTIPATTERNS 9/14): if the manifests cannot be read or too few assets
    survive validation, raise. No silent fallback to primitives.
    """
    global _GSO_SRC, _HDRI_SRC, _GSO_VALID, _HDRI_VALID
    if _GSO_SRC is not None:
        return
    os.makedirs(ASSET_CACHE, exist_ok=True)
    _GSO_SRC = _stable_source(GSO_MANIFEST, "GSO")
    _HDRI_SRC = _stable_source(HDRI_MANIFEST, "HDRI")
    gso_keys = set(_GSO_SRC._assets.keys())
    hdri_keys = set(_HDRI_SRC._assets.keys())
    _GSO_VALID = [a for a in GSO_OBJECTS if a in gso_keys]
    _HDRI_VALID = [h for h in HDRI_ENVS if h in hdri_keys]
    if len(_GSO_VALID) < 4:
        raise RuntimeError(f"too few valid GSO objects ({len(_GSO_VALID)}); manifest load broken")
    if len(_HDRI_VALID) < 3:
        raise RuntimeError(f"too few valid HDRIs ({len(_HDRI_VALID)}); manifest load broken")


# ----------------------------------------------------------------------------
# scene construction
# ----------------------------------------------------------------------------
def _rand_color(rng):
    # bright, well-separated hues so a colour jump is unmistakable to a human
    h = float(rng.uniform(0.0, 1.0))
    s = float(rng.uniform(0.85, 1.0))
    v = float(rng.uniform(0.85, 1.0))
    return kb.Color.from_hsv(h, s, v)


def _place_gso(source, asset_id, target_size, override_color=None):
    """Create a GSO object, scale so its largest dimension == target_size, and return
    (obj, bottom_offset) where bottom_offset is the origin->lowest-point z distance."""
    obj = source.create(asset_id=asset_id)
    bb = np.asarray(obj.aabbox)
    dims = bb[1] - bb[0]
    s = float(target_size) / float(max(dims))
    obj.scale = (s, s, s)
    bb = np.asarray(obj.aabbox)               # recompute after scaling
    bottom_offset = -float(bb[0][2])          # add to a desired floor z to seat the object
    obj.friction = 0.5
    obj.restitution = 0.0
    if override_color is not None:
        obj.material = kb.PrincipledBSDFMaterial(color=override_color,
                                                 roughness=0.4, metallic=0.0)
    return obj, bottom_offset


def _sphere(name, color, position, radius, roughness=0.35):
    mat = kb.PrincipledBSDFMaterial(color=color, roughness=roughness, metallic=0.0)
    return kb.Sphere(name=name, scale=radius, position=position, material=mat,
                     friction=0.3, restitution=0.0)


def build_scene(seed, violation_type, scratch, resolution, samples):
    """Return (scene, simulator, renderer, obj, meta) fully populated, pre-run."""
    _ensure_sources()
    rng = _rng(seed)
    scene = kb.Scene(resolution=resolution)
    scene.frame_start = FRAME_START
    scene.frame_end = NUM_FRAMES
    scene.frame_rate = FRAME_RATE
    scene.step_rate = STEP_RATE
    scene.gravity = GRAVITY

    # renderer + simulator (renderer must exist before HDRI setup)
    renderer = Blender(scene, scratch_dir=scratch, samples_per_pixel=samples,
                       background_transparency=False)
    simulator = PyBullet(scene, scratch_dir=scratch)

    # --- HDRI: real image-based lighting + real photographed background ---
    hdri_id = _HDRI_VALID[int(rng.randint(len(_HDRI_VALID)))]
    bg = _HDRI_SRC.create(asset_id=hdri_id)
    renderer._set_ambient_light_hdri(bg.filename)      # environment lighting
    renderer._set_background_hdri(bg.filename)          # visible real backdrop

    # --- PBR floor (HDRI-lit; large so it reads as real ground meeting the backdrop) ---
    floor = kb.Cube(
        name="floor", scale=(12.0, 12.0, 0.5),
        position=(0.0, 0.0, FLOOR_TOP_Z - 0.5), static=True,
        material=kb.PrincipledBSDFMaterial(color=kb.Color(0.32, 0.30, 0.28),
                                           roughness=0.7, metallic=0.0),
        friction=0.8, restitution=0.0,
    )
    # one crisp key light for a defined contact shadow (HDRI alone gives only soft shadow)
    sun = kb.DirectionalLight(name="sun",
                              position=(float(rng.uniform(-2, 0)),
                                        float(rng.uniform(-3, -1)), 6.0),
                              look_at=(0, 0, 0), intensity=2.2)
    cam = kb.PerspectiveCamera(name="camera", position=(0.0, -7.5, 2.4),
                               look_at=(0.0, 0.0, 1.0))
    for a in (floor, sun, cam):
        scene += a
    scene.camera = cam

    obj_color = _rand_color(rng)
    meta = {"hdri": hdri_id,
            "obj_color": [round(float(c), 4) for c in obj_color.rgb],
            "resolution": list(resolution), "samples": samples}

    if violation_type == "solidity":
        asset_id = _GSO_VALID[int(rng.randint(len(_GSO_VALID)))]
        size = float(rng.uniform(0.9, 1.15))
        obj, boff = _place_gso(_GSO_SRC, asset_id, size)   # native GSO texture
        x0 = float(rng.uniform(-0.4, 0.4))
        y0 = float(rng.uniform(-0.3, 0.3))
        z0 = float(rng.uniform(2.7, 3.0)) + boff           # seat bottom ~2.7-3.0m up
        obj.position = (x0, y0, z0)
        obj.velocity = (0.0, 0.0, 0.0)
        meta.update({"object": asset_id, "target_size": round(size, 4), "material": "gso_native"})

    elif violation_type == "permanence":
        # glossy PBR sphere (occlusion-geometry safety; see module docstring)
        radius = float(rng.uniform(0.6, 0.75))
        z0 = FLOOR_TOP_Z + radius
        x0 = float(rng.uniform(-3.3, -3.0))
        y0 = float(rng.uniform(-0.2, 0.2))
        vx = float(rng.uniform(4.3, 4.7))
        obj = _sphere("obj", obj_color, (x0, y0, z0), radius)
        obj.friction = 0.1                                  # keeps rolling, emerges early
        obj.velocity = (vx, 0.0, 0.0)
        occluder = kb.Cube(
            name="occluder", scale=(OCC_HALF_W, 0.4, 1.3),
            position=(OCC_X, y0 - 2.5, FLOOR_TOP_Z + 1.3), static=True,
            material=kb.PrincipledBSDFMaterial(color=_rand_color(rng),
                                               roughness=0.5, metallic=0.0))
        scene += occluder
        meta.update({"object": "sphere", "radius": round(radius, 4),
                     "material": "pbr_sphere", "occluder": True})

    else:  # immutability
        asset_id = _GSO_VALID[int(rng.randint(len(_GSO_VALID)))]
        size = float(rng.uniform(0.9, 1.1))
        # solid-colour override so the colour jump is a clean identity cue
        obj, boff = _place_gso(_GSO_SRC, asset_id, size, override_color=obj_color)
        x0 = float(rng.uniform(-2.4, -2.0))
        y0 = float(rng.uniform(-0.2, 0.2))
        z0 = float(rng.uniform(1.7, 2.1)) + boff
        vx = float(rng.uniform(3.0, 3.4))
        vz = float(rng.uniform(2.6, 3.0))
        obj.position = (x0, y0, z0)
        obj.velocity = (vx, 0.0, vz)
        meta.update({"object": asset_id, "target_size": round(size, 4),
                     "material": "gso_solid_override"})

    scene += obj
    return scene, simulator, renderer, obj, meta


# ----------------------------------------------------------------------------
# trajectory helpers
# ----------------------------------------------------------------------------
def _contact_index(pos):
    """0-based frame of first floor contact for a falling object, from the velocity profile.

    Robust to arbitrary GSO rest heights (the object may settle in any pose): contact is the
    first frame where the clearly-downward motion has effectively stopped. Falls back to the
    last frame if it never lands within the clip.
    """
    dz = np.diff(pos[:, 2])
    falling = False
    for i in range(1, len(pos)):
        if dz[i - 1] < -0.03:
            falling = True
        elif falling and dz[i - 1] > -0.010:
            return i
    return len(pos) - 1


def _freefall_continuation(pos, contact_idx):
    """pos up to contact, then ballistic free fall continuing THROUGH the floor."""
    out = pos.copy()
    c = max(contact_idx, 2)
    v = pos[c - 1] - pos[c - 2]
    dt = 1.0 / FRAME_RATE
    a = np.array([0.0, 0.0, GRAVITY[2]]) * dt * dt
    base = pos[c - 1].copy()
    for k, f in enumerate(range(c, len(pos)), start=1):
        out[f] = base + v * k + 0.5 * a * (k * k)
    return out, c


def _occlusion_index(pos):
    """0-based frame where the moving object first reaches the occluder centre."""
    for i in range(len(pos)):
        if pos[i, 0] >= OCC_X:
            return i
    return len(pos) - 1


def _first_divergence(a, b, thresh=0.3):
    """1-based frame of the first frame where the two clips visibly differ (measured)."""
    for i in range(len(a)):
        if np.abs(a[i].astype(np.int16) - b[i].astype(np.int16)).mean() > thresh:
            return i + 1
    return len(a)


# ----------------------------------------------------------------------------
# keyframes + render + encode
# ----------------------------------------------------------------------------
def _write_positions(obj, positions):
    for i in range(len(positions)):
        obj.position = tuple(positions[i])
        obj.keyframe_insert("position", FRAME_START + i)


def _render(renderer, frames=None):
    """Return (T,H,W,3) uint8 RGB. frames = absolute frame numbers, or None for all."""
    out = renderer.render(frames=frames, return_layers=("rgba",))
    rgba = np.asarray(out["rgba"])
    return rgba[..., :3].astype(np.uint8)


def _encode_mp4(rgb, path):
    import imageio.v2 as imageio
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimwrite(path, list(rgb), fps=FRAME_RATE, quality=9, macro_block_size=1)


def _contact_sheet(rgb, path):
    """Horizontal strip of all frames (for the human-perceptibility check)."""
    import imageio.v2 as imageio
    strip = np.concatenate(list(rgb), axis=1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.imwrite(path, strip)


# ----------------------------------------------------------------------------
# one paired sample
# ----------------------------------------------------------------------------
def generate_pair(seed, violation_type, out_dir, resolution, samples, sheets=False):
    scratch = tempfile.mkdtemp(prefix="kubric_")
    try:
        scene, simulator, renderer, obj, meta = build_scene(
            seed, violation_type, scratch, resolution, samples)
        animation, _ = simulator.run(frame_start=FRAME_START, frame_end=NUM_FRAMES)
        pos = np.asarray(animation[obj]["position"], dtype=np.float64)
        n = len(pos)

        pair_id = f"{violation_type}_seed{seed:04d}"
        clips = os.path.join(out_dir, "clips")
        possible_path = os.path.join(clips, f"{pair_id}_possible.mp4")
        impossible_path = os.path.join(clips, f"{pair_id}_impossible.mp4")

        # ---- POSSIBLE: render the valid rollout in full ----
        rgb_pos = _render(renderer)                 # (n,H,W,3)
        _encode_mp4(rgb_pos, possible_path)

        # ---- inject ONE violation, compute the divergence frame d (0-based) ----
        if violation_type == "solidity":
            contact = _contact_index(pos)
            new_pos, d = _freefall_continuation(pos, contact)
            _write_positions(obj, new_pos)

        elif violation_type == "permanence":
            occ = _occlusion_index(pos)             # fully hidden here: teleport is invisible
            new_pos = pos.copy()
            for f in range(occ, n):
                new_pos[f] = np.array([1000.0, 1000.0, 1000.0])
            _write_positions(obj, new_pos)
            d = occ

        else:  # immutability: colour swap at mid-clip (two-pass, positions unchanged)
            d = n // 2
            base_c = kb.Color(*meta["obj_color"])
            new_hue = (base_c.hsv[0] + 0.5) % 1.0
            obj.material.color = kb.Color.from_hsv(new_hue, 1.0, 1.0)
            meta["new_color"] = [round(float(c), 4) for c in obj.material.color.rgb]

        # ---- IMPOSSIBLE: full re-render, then splice the reused possible prefix on ----
        # (render(frames=subset) length is version-dependent, so re-render all n and
        #  overwrite [0:d] with the possible pixels -> provably identical prefix.)
        rgb_full = _render(renderer)
        rgb_imp = rgb_full.copy()
        if d > 0:
            rgb_imp[:d] = rgb_pos[:d]
        _encode_mp4(rgb_imp, impossible_path)

        # divergence frame = first frame that visibly differs (measured, not estimated)
        divergence_frame = _first_divergence(rgb_pos, rgb_imp)

        # exact prefix identity is guaranteed by reuse; record the measured suffix delta
        prefix_absdiff = float(np.abs(
            rgb_pos[:d].astype(np.int16) - rgb_imp[:d].astype(np.int16)
        ).mean()) if d > 0 else 0.0
        suffix_absdiff = float(np.abs(
            rgb_pos[d:].astype(np.int16) - rgb_imp[d:].astype(np.int16)).mean())

        if sheets:
            sd = os.path.join(out_dir, "contact_sheets")
            _contact_sheet(rgb_pos, os.path.join(sd, f"{pair_id}_possible.png"))
            _contact_sheet(rgb_imp, os.path.join(sd, f"{pair_id}_impossible.png"))

        common = {"violation_type": violation_type, "seed": seed, "pair_id": pair_id,
                  "num_frames": n, "resolution": list(resolution),
                  "divergence_frame": int(divergence_frame),
                  "prefix_absdiff": round(prefix_absdiff, 4),
                  "suffix_absdiff": round(suffix_absdiff, 4), **meta}
        records = [
            {"clip_path": os.path.relpath(possible_path, out_dir), "label": 1,
             "role": "possible", **common},
            {"clip_path": os.path.relpath(impossible_path, out_dir), "label": 0,
             "role": "impossible", **common},
        ]
        return records
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def _parse_seeds(spec):
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in spec.split(",")]


def main():
    global ASSET_CACHE
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0-19")
    ap.add_argument("--violations", default=",".join(VIOLATIONS))
    ap.add_argument("--limit", type=int, default=0, help="cap total pairs (0 = all)")
    ap.add_argument("--sheets", action="store_true", help="also save contact-sheet PNGs")
    ap.add_argument("--resolution", type=int, default=RESOLUTION[0],
                    help="square render size (default 512)")
    ap.add_argument("--samples", type=int, default=SAMPLES_PER_PIXEL,
                    help="Cycles samples per pixel (default 128)")
    ap.add_argument("--asset-cache", default=ASSET_CACHE,
                    help="persistent GSO/HDRI download cache")
    ap.add_argument("--index", type=int, default=-1,
                    help="render ONLY plan[index] (one pair per process). Kubric's bpy/PyBullet "
                         "are process-global singletons, so a SLURM array runs one task per pair.")
    ap.add_argument("--manifest-name", default="",
                    help="override manifest filename (default manifest.json, or "
                         "manifest_pair_<index>.json when --index is set)")
    args = ap.parse_args()

    ASSET_CACHE = args.asset_cache
    resolution = (args.resolution, args.resolution)

    seeds = _parse_seeds(args.seeds)
    violations = args.violations.split(",")
    os.makedirs(args.out, exist_ok=True)

    full_plan = [(s, v) for s in seeds for v in violations]   # round-robin friendly
    if args.limit:
        full_plan = full_plan[: args.limit]

    # One pair per process (Kubric bpy/PyBullet singletons). --index picks exactly one.
    if args.index >= 0:
        if args.index >= len(full_plan):
            print(f"index {args.index} >= plan size {len(full_plan)}; nothing to do", flush=True)
            return
        plan = [full_plan[args.index]]
        mname = args.manifest_name or f"manifest_pair_{args.index}.json"
    else:
        plan = full_plan
        mname = args.manifest_name or "manifest.json"

    if args.index < 0 and len(plan) > 1:
        print(f"WARNING: rendering {len(plan)} pairs in ONE process. Kubric's bpy/PyBullet are "
              f"process-global singletons and corrupt after ~2 scenes. Use the SLURM array "
              f"(one --index per pair) for reliable multi-pair generation.", flush=True)

    manifest = []
    mpath = os.path.join(args.out, mname)
    for i, (seed, v) in enumerate(plan):
        print(f"[{i + 1}/{len(plan)}] seed={seed} violation={v} "
              f"res={resolution[0]} spp={args.samples}", flush=True)
        recs = generate_pair(seed, v, args.out, resolution, args.samples, sheets=args.sheets)
        manifest.extend(recs)
        with open(mpath, "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"    div_frame={recs[0]['divergence_frame']} "
              f"prefix_absdiff={recs[0]['prefix_absdiff']} "
              f"suffix_absdiff={recs[0]['suffix_absdiff']} obj={recs[0].get('object')}",
              flush=True)

    print(f"DONE: {len(manifest)} clips ({len(manifest) // 2} pairs) -> {args.out}",
          flush=True)


if __name__ == "__main__":
    main()

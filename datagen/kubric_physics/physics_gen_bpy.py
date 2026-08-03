"""Paired possible/impossible physics-video generator -- bpy 4.x + PyBullet DIRECT.

GPU port of physics_gen.py. The Kubric high-level wrapper pins Blender 2.93's bpy
API and cannot drive a modern Blender, so this module drops Kubric and rebuilds the
SAME pipeline directly on:
  - bpy 4.5 (Blender as a python module) for Cycles rendering ON the Blackwell GPU
  - pybullet (DIRECT) for the rigid-body rollout, loading the GSO URDF verbatim

Everything that defines the DATA is preserved exactly from physics_gen.py:
  - the three violations (solidity / permanence / immutability) and their injection
  - the paired construction: render the POSSIBLE rollout, then reuse its pre-divergence
    frames verbatim for the IMPOSSIBLE twin, so prefix_absdiff == 0 by construction
  - GSO real meshes + Poly Haven HDRI (image-based lighting + real backdrop), PBR floor
  - the manifest schema and the contact-sheet output
  - one pair per process (bpy + pybullet are process-global singletons); --index picks one

Physics/render are decoupled exactly as before: compute one valid rollout, keyframe it
(position + orientation), render the possible clip, override ONE attribute for the twin
(position for solidity/permanence, material colour for immutability), re-render, splice.

Run inside the modern image with --nv (confirmed: --nv enumerates the Blackwell GPU
via OptiX; the explicit driver-lib bind does NOT on this image):
  apptainer exec --nv \
    --bind /work/neu/p2026_0016_neu:/work/neu/p2026_0016_neu \
    blender_gpu.sif python3.11 physics_gen_bpy.py --out <dir> --seeds 0-1 --sheets --device gpu
"""

import argparse
import colorsys
import json
import os
import shutil
import tempfile

import numpy as np

import bpy
import mathutils
import pybullet as pb

# ----------------------------------------------------------------------------
# constants (identical to physics_gen.py)
# ----------------------------------------------------------------------------
RESOLUTION = (512, 512)
NUM_FRAMES = 16
FRAME_START = 1
FRAME_RATE = 12
STEP_RATE = 240
GRAVITY = (0.0, 0.0, -9.81)
SAMPLES_PER_PIXEL = 128
FLOOR_TOP_Z = 0.0

VIOLATIONS = ("solidity", "permanence", "immutability",
              "continuity", "gravity", "momentum", "collision", "inertia")
OCC_X = 0.0
OCC_HALF_W = 0.6

ASSET_CACHE = "/work/neu/p2026_0016_neu/kubric/asset_cache"

# Same curated GSO + HDRI pools as physics_gen.py.
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

_GSO_VALID = None
_HDRI_VALID = None
_GPU_BACKEND = None       # set once by enable_gpu(): "OPTIX" / "CUDA" / None (CPU)
_HDRI_LONG_EDGE = 1024    # downscale the 4k HDRI to this long edge (env sampling was the
                          # single largest render cost); 0 = keep native 4k. Set from CLI.
_DENOISE = False          # OptiX denoiser. Measured ~0.27s/frame of FIXED overhead (the real
                          # bottleneck: 48spp+denoise 0.33s/frame vs 128spp no-denoise 0.10s/frame).
                          # OFF by default -> matches the original CPU max-fidelity (also no denoiser)
                          # and is far faster. Set --denoise to turn it back on. Set from CLI.


# ----------------------------------------------------------------------------
# asset discovery (read the on-disk cache directly; no kubric AssetSource)
# ----------------------------------------------------------------------------
def _gso_dir(asset_id):
    return os.path.join(ASSET_CACHE, "GSO", asset_id)


def _hdri_file(hdri_id):
    return os.path.join(ASSET_CACHE, "HDRI", hdri_id, "environment_4k.hdr")


def _ensure_sources():
    """Validate curated ids against what is actually unpacked in the cache. FAIL LOUD."""
    global _GSO_VALID, _HDRI_VALID
    if _GSO_VALID is not None:
        return
    _GSO_VALID = [a for a in GSO_OBJECTS
                  if os.path.isfile(os.path.join(_gso_dir(a), "visual_geometry.obj"))
                  and os.path.isfile(os.path.join(_gso_dir(a), "object.urdf"))]
    _HDRI_VALID = [h for h in HDRI_ENVS if os.path.isfile(_hdri_file(h))]
    if len(_GSO_VALID) < 4:
        raise RuntimeError(
            f"too few unpacked GSO objects ({len(_GSO_VALID)}) under {ASSET_CACHE}/GSO; "
            f"populate the cache (run the CPU kubruntu path once, or unpack the .tar.gz)")
    if len(_HDRI_VALID) < 3:
        raise RuntimeError(
            f"too few unpacked HDRIs ({len(_HDRI_VALID)}) under {ASSET_CACHE}/HDRI")


def _gso_bounds(asset_id):
    """(dims xyz, bounds) from the asset data.json (native, pre-scale)."""
    with open(os.path.join(_gso_dir(asset_id), "data.json")) as fh:
        d = json.load(fh)
    bounds = np.asarray(d["kwargs"]["bounds"], dtype=np.float64)   # [[min],[max]]
    return bounds[1] - bounds[0], bounds


# ----------------------------------------------------------------------------
# GPU enablement (log which device is actually used; never silent CPU)
# ----------------------------------------------------------------------------
def enable_gpu(device_mode):
    """device_mode in {gpu, cpu, auto}. Returns backend str used ('OPTIX'/'CUDA'/'CPU')."""
    global _GPU_BACKEND
    if device_mode == "cpu":
        _GPU_BACKEND = "CPU"
        print("device: CPU (forced)", flush=True)
        return "CPU"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for kind in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = kind
            prefs.get_devices()
        except Exception as e:  # noqa: BLE001
            print(f"device: {kind} unavailable ({e})", flush=True)
            continue
        gpu_devs = [d for d in prefs.devices if d.type == kind]
        if gpu_devs:
            for d in prefs.devices:
                d.use = d.type == kind
            _GPU_BACKEND = kind
            names = [d.name for d in gpu_devs]
            print(f"device: GPU via {kind} -> {names}", flush=True)
            return kind
        print(f"device: {kind} set but no {kind} device enumerated", flush=True)
    if device_mode == "gpu":
        raise RuntimeError(
            "device=gpu requested but Cycles enumerated NO OptiX/CUDA GPU. "
            "Refusing silent CPU fallback (ANTIPATTERNS 9/14). Check driver-lib bind / image.")
    _GPU_BACKEND = "CPU"
    print("device: no GPU enumerated -> CPU (auto)", flush=True)
    return "CPU"


def _apply_device(scene):
    """Re-assert the Cycles device on THIS scene. Must run AFTER build_scene's
    read_factory_settings wipe, which resets the cycles addon preferences (that wipe
    was silently disabling the GPU and falling back to CPU). Re-select + hard-check."""
    scene.render.engine = "CYCLES"
    if _GPU_BACKEND in ("OPTIX", "CUDA"):
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = _GPU_BACKEND
        prefs.get_devices()
        n_on = 0
        for d in prefs.devices:
            d.use = d.type == _GPU_BACKEND
            n_on += int(d.use)
        if n_on == 0:
            raise RuntimeError(
                f"{_GPU_BACKEND} selected but 0 devices enabled after scene wipe "
                f"(ANTIPATTERNS 9/14: refusing silent CPU fallback)")
        scene.cycles.device = "GPU"
    else:
        scene.cycles.device = "CPU"


# ----------------------------------------------------------------------------
# small bpy helpers
# ----------------------------------------------------------------------------
def _wipe():
    """Empty the default scene (factory startup ships a cube+camera+light)."""
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _principled(name, rgb, roughness, metallic=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (rgb[0], rgb[1], rgb[2], 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    return mat


def _set_world_hdri(hdr_path):
    """One Environment Texture node = image-based lighting AND the visible backdrop."""
    world = bpy.data.worlds.new("world")
    bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    img = bpy.data.images.load(hdr_path)
    # downscale the env map (the 4k importance map + per-sample lookups were the biggest
    # render cost). Lighting + a real backdrop are preserved; only fine backdrop detail drops.
    if _HDRI_LONG_EDGE and img.size[0] > _HDRI_LONG_EDGE:
        h = max(1, round(img.size[1] * _HDRI_LONG_EDGE / img.size[0]))
        img.scale(_HDRI_LONG_EDGE, h)
    env.image = img
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])


def _add_cube(name, half_extents, location, material):
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=location)  # unit cube spans -1..1
    o = bpy.context.object
    o.name = name
    o.scale = half_extents                                        # kubric-style half-extents
    o.data.materials.append(material)
    return o


def _add_sphere(name, radius, location, material):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=location, segments=48, ring_count=24)
    o = bpy.context.object
    o.name = name
    bpy.ops.object.shade_smooth()
    o.data.materials.append(material)
    return o


def _point(obj, target):
    d = mathutils.Vector(target) - obj.location
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = d.to_track_quat("-Z", "Y")


def _import_gso(asset_id, scale, solid_color=None):
    """Import the GSO visual mesh verbatim (identity axes, matching the pybullet collision
    load), scale it, and return the bpy object. solid_color overrides the material."""
    obj_path = os.path.join(_gso_dir(asset_id), "visual_geometry.obj")
    before = set(bpy.data.objects)
    # identity axes (up=Z, forward=Y) so bpy vertices == the raw obj == pybullet collision frame
    bpy.ops.wm.obj_import(filepath=obj_path, forward_axis="Y", up_axis="Z")
    new = [o for o in bpy.data.objects if o not in before]
    if not new:
        raise RuntimeError(f"OBJ import produced no object for {asset_id}")
    # join multi-part imports into one object
    for o in bpy.data.objects:
        o.select_set(o in new)
    bpy.context.view_layer.objects.active = new[0]
    if len(new) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    obj.name = "obj"
    obj.location = (0.0, 0.0, 0.0)
    obj.scale = (scale, scale, scale)
    if solid_color is not None:
        obj.data.materials.clear()
        obj.data.materials.append(_principled("obj_solid", solid_color, roughness=0.4))
    return obj


# ----------------------------------------------------------------------------
# colour helpers (replace kb.Color)
# ----------------------------------------------------------------------------
def _rand_color(rng):
    h = float(rng.uniform(0.0, 1.0))
    s = float(rng.uniform(0.85, 1.0))
    v = float(rng.uniform(0.85, 1.0))
    return colorsys.hsv_to_rgb(h, s, v)


def _rng(seed):
    return np.random.RandomState(seed)


# ----------------------------------------------------------------------------
# scene construction (returns a context bundle used by generate_pair)
# ----------------------------------------------------------------------------
class Scene:
    pass


def build_scene(seed, violation_type, resolution, samples):
    _ensure_sources()
    _pb_reset()          # fresh physics world for this pair (enables many pairs per process)
    rng = _rng(seed)
    _wipe()
    scene = bpy.context.scene
    _apply_device(scene)
    scene.cycles.samples = int(samples)
    scene.render.resolution_x = int(resolution[0])
    scene.render.resolution_y = int(resolution[1])
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    # keep geometry/BVH/kernels resident on the device across frames: this is the
    # difference between ~3-4s/frame (rebuild every frame) and a fraction of that.
    scene.render.use_persistent_data = True
    # OptiX denoiser is ~0.27s/frame of fixed overhead (the real bottleneck); off by default.
    scene.cycles.use_denoising = _DENOISE
    bpy.context.view_layer.cycles.use_denoising = _DENOISE
    scene.frame_start = FRAME_START
    scene.frame_end = NUM_FRAMES
    scene.render.fps = FRAME_RATE

    # HDRI world (lighting + backdrop)
    hdri_id = _HDRI_VALID[int(rng.randint(len(_HDRI_VALID)))]
    _set_world_hdri(_hdri_file(hdri_id))

    # PBR floor (kubric half-extents (12,12,0.5), top face at z=0)
    floor_mat = _principled("floor", (0.32, 0.30, 0.28), roughness=0.7)
    _add_cube("floor", (12.0, 12.0, 0.5), (0.0, 0.0, FLOOR_TOP_Z - 0.5), floor_mat)

    # crisp key light for a defined contact shadow
    sun_data = bpy.data.lights.new("sun", type="SUN")
    sun_data.energy = 3.0
    sun = bpy.data.objects.new("sun", sun_data)
    bpy.context.collection.objects.link(sun)
    sun.location = (float(rng.uniform(-2, 0)), float(rng.uniform(-3, -1)), 6.0)
    _point(sun, (0, 0, 0))

    # camera (kubric PerspectiveCamera defaults: 50mm lens, 36mm sensor)
    cam_data = bpy.data.cameras.new("camera")
    cam_data.lens = 50.0
    cam_data.sensor_width = 36.0
    cam = bpy.data.objects.new("camera", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = (0.0, -7.5, 2.4)
    _point(cam, (0.0, 0.0, 1.0))
    scene.camera = cam

    obj_color = _rand_color(rng)
    meta = {"hdri": hdri_id,
            "obj_color": [round(float(c), 4) for c in obj_color],
            "resolution": list(resolution), "samples": int(samples)}

    ctx = Scene()
    ctx.scene = scene
    ctx.violation_type = violation_type
    ctx.meta = meta

    if violation_type == "solidity":
        asset_id = _GSO_VALID[int(rng.randint(len(_GSO_VALID)))]
        size = float(rng.uniform(0.9, 1.15))
        dims, bounds = _gso_bounds(asset_id)
        s = size / float(dims.max())
        boff = -float(bounds[0][2]) * s
        obj = _import_gso(asset_id, s)
        x0 = float(rng.uniform(-0.4, 0.4))
        y0 = float(rng.uniform(-0.3, 0.3))
        z0 = float(rng.uniform(2.7, 3.0)) + boff
        ctx.body = _load_gso_body(asset_id, s, (x0, y0, z0), vel=(0, 0, 0),
                                  friction=0.5, restitution=0.0)
        ctx.obj = obj
        meta.update({"object": asset_id, "target_size": round(size, 4), "material": "gso_native"})

    elif violation_type == "permanence":
        radius = float(rng.uniform(0.6, 0.75))
        z0 = FLOOR_TOP_Z + radius
        x0 = float(rng.uniform(-3.3, -3.0))
        y0 = float(rng.uniform(-0.2, 0.2))
        vx = float(rng.uniform(4.3, 4.7))
        sph_mat = _principled("obj", obj_color, roughness=0.35)
        obj = _add_sphere("obj", radius, (x0, y0, z0), sph_mat)
        ctx.body = _sphere_body(radius, (x0, y0, z0), vel=(vx, 0, 0),
                                friction=0.1, restitution=0.0)
        # visual occluder (static box; positioned toward the camera in -y, purely visual)
        occ_mat = _principled("occluder", _rand_color(rng), roughness=0.5)
        _add_cube("occluder", (OCC_HALF_W, 0.4, 1.3),
                  (OCC_X, y0 - 2.5, FLOOR_TOP_Z + 1.3), occ_mat)
        ctx.obj = obj
        meta.update({"object": "sphere", "radius": round(radius, 4),
                     "material": "pbr_sphere", "occluder": True})

    elif violation_type in VIOLATION_BUILDERS:
        # extended violations (continuity / gravity / momentum / collision / inertia):
        # each builder sets ctx.body + ctx.obj (+ any static second body) and meta.
        VIOLATION_BUILDERS[violation_type](ctx, rng, obj_color, meta)

    else:  # immutability
        asset_id = _GSO_VALID[int(rng.randint(len(_GSO_VALID)))]
        size = float(rng.uniform(0.9, 1.1))
        dims, bounds = _gso_bounds(asset_id)
        s = size / float(dims.max())
        boff = -float(bounds[0][2]) * s
        obj = _import_gso(asset_id, s, solid_color=obj_color)
        x0 = float(rng.uniform(-2.4, -2.0))
        y0 = float(rng.uniform(-0.2, 0.2))
        z0 = float(rng.uniform(1.7, 2.1)) + boff
        vx = float(rng.uniform(3.0, 3.4))
        vz = float(rng.uniform(2.6, 3.0))
        ctx.body = _load_gso_body(asset_id, s, (x0, y0, z0), vel=(vx, 0, vz),
                                  friction=0.5, restitution=0.0)
        ctx.obj = obj
        meta.update({"object": asset_id, "target_size": round(size, 4),
                     "material": "gso_solid_override"})

    return ctx


# ----------------------------------------------------------------------------
# PyBullet rollout (replaces kubric.simulator.PyBullet)
# ----------------------------------------------------------------------------
def _pb_reset():
    """Fresh physics world for THIS pair (reset, do not accumulate) so many pairs can
    share one process: connect once, then resetSimulation per pair and rebuild the
    floor. Without the reset, bodies from prior pairs would linger and corrupt the
    rollout (the reason Kubric was one-pair-per-process)."""
    if not pb.getConnectionInfo().get("isConnected"):
        pb.connect(pb.DIRECT)
    pb.resetSimulation()
    pb.setGravity(*GRAVITY)
    pb.setTimeStep(1.0 / STEP_RATE)
    # static floor collision matching the visual floor (top at z=0)
    fcol = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[12.0, 12.0, 0.5])
    floor = pb.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=fcol,
                               basePosition=[0.0, 0.0, FLOOR_TOP_Z - 0.5])
    pb.changeDynamics(floor, -1, lateralFriction=0.8, restitution=0.0)


def _load_gso_body(asset_id, scale, position, vel, friction, restitution):
    urdf = os.path.join(_gso_dir(asset_id), "object.urdf")
    pb.setAdditionalSearchPath(_gso_dir(asset_id))
    body = pb.loadURDF(urdf, basePosition=list(position), globalScaling=float(scale))
    pb.resetBaseVelocity(body, linearVelocity=list(vel))
    pb.changeDynamics(body, -1, lateralFriction=friction, restitution=restitution)
    return body


def _sphere_body(radius, position, vel, friction, restitution):
    col = pb.createCollisionShape(pb.GEOM_SPHERE, radius=float(radius))
    body = pb.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=col,
                              basePosition=list(position))
    pb.resetBaseVelocity(body, linearVelocity=list(vel))
    pb.changeDynamics(body, -1, lateralFriction=friction, restitution=restitution)
    return body


def run_physics(body):
    """Return (positions (N,3), quats_wxyz (N,4)) sampled at each of NUM_FRAMES frames."""
    steps_per_frame = STEP_RATE // FRAME_RATE
    pos = np.zeros((NUM_FRAMES, 3), dtype=np.float64)
    quat = np.zeros((NUM_FRAMES, 4), dtype=np.float64)   # Blender wxyz order

    def _sample(i):
        p, o = pb.getBasePositionAndOrientation(body)    # o = xyzw
        pos[i] = p
        quat[i] = (o[3], o[0], o[1], o[2])               # -> wxyz

    _sample(0)
    for f in range(1, NUM_FRAMES):
        for _ in range(steps_per_frame):
            pb.stepSimulation()
        _sample(f)
    return pos, quat


# ----------------------------------------------------------------------------
# trajectory helpers (identical to physics_gen.py)
# ----------------------------------------------------------------------------
def _contact_index(pos):
    dz = np.diff(pos[:, 2])
    falling = False
    for i in range(1, len(pos)):
        if dz[i - 1] < -0.03:
            falling = True
        elif falling and dz[i - 1] > -0.010:
            return i
    return len(pos) - 1


def _freefall_continuation(pos, contact_idx):
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
    for i in range(len(pos)):
        if pos[i, 0] >= OCC_X:
            return i
    return len(pos) - 1


def _first_divergence(a, b, thresh=0.3):
    for i in range(len(a)):
        if np.abs(a[i].astype(np.int16) - b[i].astype(np.int16)).mean() > thresh:
            return i + 1
    return len(a)


# ----------------------------------------------------------------------------
# EXTENDED VIOLATION LAYER (additive; violation-logic only, no render coupling)
# ----------------------------------------------------------------------------
# Five more intuitive-physics violations, registered exactly like solidity /
# permanence / immutability. Two registries:
#   VIOLATION_BUILDERS[type](ctx, rng, obj_color, meta)  -- physics/scene setup
#       (sets ctx.body = pybullet body, ctx.obj = keyframed bpy object, and may
#        add a static second body; consulted by build_scene)
#   VIOLATION_INJECTORS[type](pos, meta) -> (new_pos, d)  -- the violation itself
#       (consulted by generate_pair; d = 0-based divergence frame)
#
# PAIRED INVARIANT (identical to the three originals): every injector builds
# new_pos = pos.copy() and OVERWRITES ONLY frames f >= d. So new_pos[:d] == pos[:d]
# byte-for-byte, the impossible twin reuses the possible prefix, and
# prefix_absdiff == 0 by construction. The divergence is injected AT/AFTER frame d,
# and 0 < d < NUM_FRAMES for all five so generate_pair's splice path is taken.
# Injectors touch POSITION only; the physics orientation keyframes are kept, exactly
# as solidity/permanence do (via _rekey_positions).

# extended-violation tunables (kept local to this layer; do not touch render params)
TELEPORT_GAP = 2.4              # continuity: instantaneous x-jump (world units).
                               # A normal per-frame step here is ~0.3-0.4 units, so a
                               # 2.4 jump in one frame reads unambiguously as a teleport.
GRAVITY_ASCENT_PER_FRAME = 0.13  # gravity: upward drift per frame after divergence.
ENERGY_GAIN = 1.30             # momentum: rebound apex as a multiple of the drop
                               # height (>1 => the ball returns HIGHER than released).
COLLISION_TARGET_X = 0.0       # collision: static target-box centre, in the sphere path.
COLLISION_TARGET_HALF = (0.4, 0.7, 0.7)   # target-box half-extents (x, y, z).
COLLISION_DECEL_FRAC = 0.5     # collision: forward-speed collapse fraction => contact.


def _decel_index(pos, frac=COLLISION_DECEL_FRAC):
    """First frame at which forward (x) speed collapses below `frac` of the initial
    forward speed -- i.e. the frame the moving body actually hits the target and
    decelerates. Mirrors _contact_index / _occlusion_index for the collision case."""
    dx = np.diff(pos[:, 0])
    v0 = dx[0] if len(dx) else 0.0
    if abs(v0) < 1e-6:
        return max(2, len(pos) // 2)
    for i in range(1, len(pos)):
        if dx[i - 1] < frac * v0:
            return max(2, i)
    return len(pos) - 1


def _inject_continuity(pos, meta=None):
    """CONTINUITY: the object teleports -- a discontinuous jump in position across a
    gap -- instead of a smooth trajectory.

    Human-perceptibility: the object is sliding steadily across the scene, then in a
    single frame vanishes and reappears well ahead, skipping a stretch of ground it is
    never seen to cross. The jump (~2.4 units) is many times a normal per-frame step,
    so the discontinuity is obvious to any viewer (clears the ~100% human gate).

    Invariant: new_pos[:d] == pos[:d]; from d the whole tail is rigidly offset by the
    gap, so the motion stays smooth on each side of a single hard jump at d."""
    n = len(pos)
    d = max(2, n // 2)
    new = pos.copy()
    new[d:] = pos[d:] + np.array([TELEPORT_GAP, 0.0, 0.0])
    if meta is not None:
        meta["teleport_gap"] = round(float(TELEPORT_GAP), 3)
    return new, d


def _inject_gravity(pos, meta=None):
    """GRAVITY / SUPPORT: the object stops falling and instead drifts UPWARD in mid-air
    with nothing under it, instead of falling and resting on the floor.

    Human-perceptibility: an object dropped from a height falls as expected, then part
    way down reverses and floats steadily upward through empty space, unsupported. A
    falling thing reversing into a rise with no contact is a flat violation of gravity
    and support that everyone sees immediately.

    Invariant: divergence d is a mid-FALL frame (contact // 2, still descending), so
    new_pos[:d] == pos[:d]; from d the object holds x,y and gains z each frame."""
    n = len(pos)
    contact = _contact_index(pos)
    d = max(2, contact // 2)
    new = pos.copy()
    anchor = pos[d - 1].copy()
    for k, f in enumerate(range(d, n), start=1):
        new[f] = anchor + np.array([0.0, 0.0, GRAVITY_ASCENT_PER_FRAME * k])
    if meta is not None:
        meta["ascent_per_frame"] = round(float(GRAVITY_ASCENT_PER_FRAME), 3)
    return new, d


def _inject_momentum(pos, meta=None):
    """MOMENTUM / ENERGY: the object bounces back HIGHER than it was dropped from, so
    kinetic energy is created from nothing (a real bounce loses energy and returns
    lower).

    Human-perceptibility: a ball drops, strikes the floor, and rebounds ABOVE its
    release height. A bounce that overshoots the drop point is the textbook "energy
    from nowhere" cue and is obvious side by side with a normal (lower) bounce.

    Invariant: divergence d is the bounce/contact frame; new_pos[:d] == pos[:d]. From d
    the object follows a projectile arc whose apex is ENERGY_GAIN x the original drop
    height (a real restitution bounce, which the possible clip shows, peaks lower)."""
    n = len(pos)
    contact = _contact_index(pos)
    d = min(max(2, contact), n - 1)
    new = pos.copy()
    z_floor = float(pos[d, 2])
    h0 = float(pos[0, 2])
    x_c, y_c = float(pos[d, 0]), float(pos[d, 1])
    h_peak = z_floor + ENERGY_GAIN * (h0 - z_floor)      # apex ABOVE the release height
    g = -GRAVITY[2]                                       # +9.81
    dt = 1.0 / FRAME_RATE
    v_up = float(np.sqrt(max(2.0 * g * (h_peak - z_floor), 0.0)))
    for k, f in enumerate(range(d, n), start=0):
        t = k * dt
        z = z_floor + v_up * t - 0.5 * g * t * t
        new[f] = np.array([x_c, y_c, max(z, z_floor)])
    if meta is not None:
        meta["rebound_peak_z"] = round(float(h_peak), 3)
        meta["drop_height_z"] = round(h0, 3)
    return new, d


def _inject_collision(pos, meta=None):
    """COLLISION: the object passes straight THROUGH a second body (the static target
    box), failing to react to a collision that should deflect or stop it. This is a
    second-body pass-through, distinct from solidity (which passes through the floor).

    Human-perceptibility: a moving object reaches a solid block squarely in its path and
    slides clean through it instead of stopping or bouncing off. Interpenetration of two
    solids is a clear, obvious impossibility.

    Invariant: divergence d is the frame the object reaches the box and (in the possible
    rollout) decelerates; new_pos[:d] == pos[:d]. From d the object continues on its
    pre-contact straight-line velocity, carrying it through the box's volume."""
    n = len(pos)
    d = _decel_index(pos)
    new = pos.copy()
    v = pos[d - 1] - pos[d - 2]                           # pre-contact per-frame velocity
    base = pos[d - 1].copy()
    for k, f in enumerate(range(d, n), start=1):
        new[f] = base + v * k
    if meta is not None:
        meta["passthrough_from_frame"] = int(d)
    return new, d


def _inject_inertia(pos, meta=None):
    """INERTIA: the object changes direction on its own -- a right-angle swerve with no
    force acting on it -- instead of continuing in a straight line.

    Human-perceptibility: an object sliding straight across the floor abruptly turns a
    sharp corner in mid-motion, with nothing touching it and no ramp or wall to explain
    the turn. An unforced change of direction violates inertia (Newton's first law) and
    is immediately visible.

    Invariant: divergence d is mid-slide; new_pos[:d] == pos[:d]. From d the pre-turn
    forward speed is redirected 90 degrees (into +y) at the same magnitude, so the path
    breaks sharply sideways while staying on the floor plane (z held)."""
    n = len(pos)
    d = max(2, n // 2)
    new = pos.copy()
    v = pos[d - 1] - pos[d - 2]
    speed = float(np.linalg.norm(v[:2]))
    base = pos[d - 1].copy()
    perp = np.array([0.0, speed, 0.0])                   # redirect same speed into +y
    for k, f in enumerate(range(d, n), start=1):
        new[f] = base + perp * k
    if meta is not None:
        meta["swerve_speed"] = round(speed, 4)
    return new, d


VIOLATION_INJECTORS = {
    "continuity": _inject_continuity,
    "gravity": _inject_gravity,
    "momentum": _inject_momentum,
    "collision": _inject_collision,
    "inertia": _inject_inertia,
}


# ---- scene / physics setup for the five extended violations --------------------
# Each builder sets ctx.body (the pybullet body run_physics samples), ctx.obj (the
# single bpy object generate_pair keyframes), and updates meta. The collision builder
# also adds a STATIC second body (mass 0), which run_physics never samples and the
# render loop never keyframes: it just sits in the scene as the thing to pass through.
def _build_continuity(ctx, rng, obj_color, meta):
    radius = float(rng.uniform(0.6, 0.75))
    z0 = FLOOR_TOP_Z + radius
    x0 = float(rng.uniform(-3.3, -3.0))
    y0 = float(rng.uniform(-0.2, 0.2))
    vx = float(rng.uniform(3.8, 4.2))
    mat = _principled("obj", obj_color, roughness=0.35)
    ctx.obj = _add_sphere("obj", radius, (x0, y0, z0), mat)
    ctx.body = _sphere_body(radius, (x0, y0, z0), vel=(vx, 0, 0),
                            friction=0.05, restitution=0.0)
    meta.update({"object": "sphere", "radius": round(radius, 4), "material": "pbr_sphere"})


def _build_gravity(ctx, rng, obj_color, meta):
    radius = float(rng.uniform(0.55, 0.7))
    x0 = float(rng.uniform(-0.4, 0.4))
    y0 = float(rng.uniform(-0.3, 0.3))
    z0 = float(rng.uniform(2.7, 3.0))
    mat = _principled("obj", obj_color, roughness=0.35)
    ctx.obj = _add_sphere("obj", radius, (x0, y0, z0), mat)
    ctx.body = _sphere_body(radius, (x0, y0, z0), vel=(0, 0, 0),
                            friction=0.5, restitution=0.0)
    meta.update({"object": "sphere", "radius": round(radius, 4), "material": "pbr_sphere"})


def _build_momentum(ctx, rng, obj_color, meta):
    radius = float(rng.uniform(0.5, 0.65))
    x0 = float(rng.uniform(-0.2, 0.2))
    y0 = float(rng.uniform(-0.2, 0.2))
    z0 = float(rng.uniform(2.7, 3.0))
    mat = _principled("obj", obj_color, roughness=0.35)
    ctx.obj = _add_sphere("obj", radius, (x0, y0, z0), mat)
    # restitution > 0 so the POSSIBLE clip shows a real (lower) bounce to contrast against
    ctx.body = _sphere_body(radius, (x0, y0, z0), vel=(0, 0, 0),
                            friction=0.2, restitution=0.6)
    meta.update({"object": "sphere", "radius": round(radius, 4),
                 "material": "pbr_sphere", "restitution": 0.6})


def _build_collision(ctx, rng, obj_color, meta):
    radius = float(rng.uniform(0.45, 0.55))
    z0 = FLOOR_TOP_Z + radius
    x0 = float(rng.uniform(-3.3, -3.0))
    y0 = float(rng.uniform(-0.15, 0.15))
    vx = float(rng.uniform(3.8, 4.2))
    mat = _principled("obj", obj_color, roughness=0.35)
    ctx.obj = _add_sphere("obj", radius, (x0, y0, z0), mat)
    ctx.body = _sphere_body(radius, (x0, y0, z0), vel=(vx, 0, 0),
                            friction=0.05, restitution=0.0)
    # static second body (the thing to pass through): bpy visual + pybullet collision.
    hx, hy, hz = COLLISION_TARGET_HALF
    box_pos = (COLLISION_TARGET_X, y0, FLOOR_TOP_Z + hz)
    tmat = _principled("target", _rand_color(rng), roughness=0.5)
    ctx.target = _add_cube("target", COLLISION_TARGET_HALF, box_pos, tmat)
    tcol = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[hx, hy, hz])
    tbody = pb.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=tcol,
                               basePosition=list(box_pos))
    pb.changeDynamics(tbody, -1, lateralFriction=0.8, restitution=0.0)
    meta.update({"object": "sphere", "radius": round(radius, 4),
                 "material": "pbr_sphere", "target_body": True})


def _build_inertia(ctx, rng, obj_color, meta):
    radius = float(rng.uniform(0.6, 0.75))
    z0 = FLOOR_TOP_Z + radius
    x0 = float(rng.uniform(-3.0, -2.6))
    y0 = float(rng.uniform(-0.1, 0.1))
    vx = float(rng.uniform(3.5, 3.9))
    mat = _principled("obj", obj_color, roughness=0.35)
    ctx.obj = _add_sphere("obj", radius, (x0, y0, z0), mat)
    ctx.body = _sphere_body(radius, (x0, y0, z0), vel=(vx, 0, 0),
                            friction=0.02, restitution=0.0)
    meta.update({"object": "sphere", "radius": round(radius, 4), "material": "pbr_sphere"})


VIOLATION_BUILDERS = {
    "continuity": _build_continuity,
    "gravity": _build_gravity,
    "momentum": _build_momentum,
    "collision": _build_collision,
    "inertia": _build_inertia,
}


# ----------------------------------------------------------------------------
# keyframing + render + encode
# ----------------------------------------------------------------------------
def _keyframe_motion(obj, positions, quats):
    """Keyframe position AND orientation for every frame (the possible animation)."""
    obj.rotation_mode = "QUATERNION"
    for i in range(len(positions)):
        obj.location = tuple(positions[i])
        obj.rotation_quaternion = tuple(quats[i])
        obj.keyframe_insert("location", frame=FRAME_START + i)
        obj.keyframe_insert("rotation_quaternion", frame=FRAME_START + i)


def _rekey_positions(obj, positions):
    """Overwrite ONLY the location keyframes (orientation from physics is kept)."""
    if obj.animation_data and obj.animation_data.action:
        for fc in list(obj.animation_data.action.fcurves):
            if fc.data_path == "location":
                obj.animation_data.action.fcurves.remove(fc)
    for i in range(len(positions)):
        obj.location = tuple(positions[i])
        obj.keyframe_insert("location", frame=FRAME_START + i)


def _render(ctx, scratch, tag, start=FRAME_START, end=NUM_FRAMES):
    """Render frames [start..end] as ONE animation pass (device data persists across
    frames), read the PNGs back, return (end-start+1,H,W,3) uint8 RGB. tag keeps the
    possible and impossible passes in separate dirs. A frame subrange lets the
    impossible twin render only its post-divergence frames."""
    scene = ctx.scene
    scene.frame_start = start
    scene.frame_end = end
    outdir = os.path.join(scratch, tag)
    os.makedirs(outdir, exist_ok=True)
    scene.render.filepath = os.path.join(outdir, "f_")   # Blender appends 4-digit frame no.
    bpy.ops.render.render(animation=True)
    frames = [_read_png(os.path.join(outdir, f"f_{f:04d}.png"))
              for f in range(start, end + 1)]
    return np.asarray(frames, dtype=np.uint8)


def _read_png(path):
    import imageio.v2 as imageio
    a = np.asarray(imageio.imread(path))
    return a[..., :3].astype(np.uint8)


def _encode_mp4(rgb, path):
    import imageio.v2 as imageio
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimwrite(path, list(rgb), fps=FRAME_RATE, quality=9, macro_block_size=1)


def _contact_sheet(rgb, path):
    import imageio.v2 as imageio
    strip = np.concatenate(list(rgb), axis=1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.imwrite(path, strip)


# ----------------------------------------------------------------------------
# one paired sample (SAME flow/splice/measurement as physics_gen.py)
# ----------------------------------------------------------------------------
def generate_pair(seed, violation_type, out_dir, resolution, samples, sheets=False):
    scratch = tempfile.mkdtemp(prefix="bpy_")
    try:
        ctx = build_scene(seed, violation_type, resolution, samples)
        obj, meta = ctx.obj, ctx.meta

        pos, quat = run_physics(ctx.body)
        n = len(pos)
        _keyframe_motion(obj, pos, quat)

        pair_id = f"{violation_type}_seed{seed:04d}"
        clips = os.path.join(out_dir, "clips")
        possible_path = os.path.join(clips, f"{pair_id}_possible.mp4")
        impossible_path = os.path.join(clips, f"{pair_id}_impossible.mp4")

        # ---- POSSIBLE: render the valid rollout in full ----
        rgb_pos = _render(ctx, scratch, "pos")
        _encode_mp4(rgb_pos, possible_path)

        # ---- inject ONE violation, compute the divergence frame d (0-based) ----
        if violation_type == "solidity":
            contact = _contact_index(pos)
            new_pos, d = _freefall_continuation(pos, contact)
            _rekey_positions(obj, new_pos)

        elif violation_type == "permanence":
            occ = _occlusion_index(pos)
            new_pos = pos.copy()
            for f in range(occ, n):
                new_pos[f] = np.array([1000.0, 1000.0, 1000.0])
            _rekey_positions(obj, new_pos)
            d = occ

        elif violation_type in VIOLATION_INJECTORS:
            # extended violations: compute the impossible trajectory (positions only) and
            # the 0-based divergence frame d. new_pos[:d] == pos[:d] by construction, so
            # the impossible twin reuses the possible prefix (prefix_absdiff == 0) and the
            # existing splice path below renders only frames d..N-1.
            new_pos, d = VIOLATION_INJECTORS[violation_type](pos, meta)
            _rekey_positions(obj, new_pos)

        else:  # immutability: colour swap at mid-clip (positions unchanged)
            d = n // 2
            base_h = colorsys.rgb_to_hsv(*meta["obj_color"])[0]
            new_rgb = colorsys.hsv_to_rgb((base_h + 0.5) % 1.0, 1.0, 1.0)
            bsdf = obj.data.materials[0].node_tree.nodes.get("Principled BSDF")
            bsdf.inputs["Base Color"].default_value = (new_rgb[0], new_rgb[1], new_rgb[2], 1.0)
            meta["new_color"] = [round(float(c), 4) for c in new_rgb]

        # ---- IMPOSSIBLE: only the post-divergence frames actually differ, so render
        #      ONLY frames d..N-1 and splice them onto the reused possible prefix
        #      (prefix stays byte-identical -> prefix_absdiff = 0 by construction). ----
        if 0 < d < n:
            suffix = _render(ctx, scratch, "imp", start=FRAME_START + d, end=NUM_FRAMES)
            rgb_imp = np.concatenate([rgb_pos[:d], suffix], axis=0)
        else:  # d == 0 or d >= n: nothing to reuse, render the whole clip
            rgb_imp = _render(ctx, scratch, "imp")
        _encode_mp4(rgb_imp, impossible_path)

        divergence_frame = _first_divergence(rgb_pos, rgb_imp)

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
                  "suffix_absdiff": round(suffix_absdiff, 4),
                  "render_device": _GPU_BACKEND, **meta}
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
# CLI (identical surface to physics_gen.py, plus --device)
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
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sheets", action="store_true")
    ap.add_argument("--resolution", type=int, default=RESOLUTION[0])
    ap.add_argument("--samples", type=int, default=SAMPLES_PER_PIXEL)
    ap.add_argument("--asset-cache", default=ASSET_CACHE)
    ap.add_argument("--device", choices=("gpu", "cpu", "auto"), default="auto",
                    help="gpu = require an OptiX/CUDA device (no silent CPU fallback)")
    ap.add_argument("--hdri-long-edge", type=int, default=1024,
                    help="downscale the HDRI env map to this long edge (0 = native 4k).")
    ap.add_argument("--denoise", action="store_true",
                    help="enable the OptiX denoiser (~0.27s/frame fixed overhead; OFF by default "
                         "since 128spp no-denoise matches the original fidelity and is 3x faster)")
    ap.add_argument("--index", type=int, default=-1,
                    help="render plan[index] (one pair). With --index-end, a whole block.")
    ap.add_argument("--index-end", type=int, default=-1,
                    help="with --index, render the CONTIGUOUS block plan[index..index-end] in "
                         "ONE process. bpy + pybullet are reset between pairs, so the ~4-5s "
                         "process startup (bpy import + OptiX pipeline warm-up) is paid ONCE "
                         "and amortized across the block.")
    ap.add_argument("--manifest-name", default="")
    args = ap.parse_args()

    global _HDRI_LONG_EDGE, _DENOISE
    ASSET_CACHE = args.asset_cache
    _HDRI_LONG_EDGE = args.hdri_long_edge
    _DENOISE = args.denoise
    resolution = (args.resolution, args.resolution)

    enable_gpu(args.device)

    seeds = _parse_seeds(args.seeds)
    violations = args.violations.split(",")
    os.makedirs(args.out, exist_ok=True)

    full_plan = [(s, v) for s in seeds for v in violations]
    if args.limit:
        full_plan = full_plan[: args.limit]

    if args.index >= 0:
        if args.index >= len(full_plan):
            print(f"index {args.index} >= plan size {len(full_plan)}; nothing to do", flush=True)
            return
        end = args.index_end if args.index_end >= args.index else args.index
        end = min(end, len(full_plan) - 1)
        plan = full_plan[args.index:end + 1]
        if args.index == end:
            mname = args.manifest_name or f"manifest_pair_{args.index}.json"
        else:
            mname = args.manifest_name or f"manifest_block_{args.index}_{end}.json"
    else:
        plan = full_plan
        mname = args.manifest_name or "manifest.json"

    manifest = []
    mpath = os.path.join(args.out, mname)
    for i, (seed, v) in enumerate(plan):
        print(f"[{i + 1}/{len(plan)}] seed={seed} violation={v} "
              f"res={resolution[0]} spp={args.samples} dev={_GPU_BACKEND}", flush=True)
        recs = generate_pair(seed, v, args.out, resolution, args.samples, sheets=args.sheets)
        manifest.extend(recs)
        with open(mpath, "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"    div_frame={recs[0]['divergence_frame']} "
              f"prefix_absdiff={recs[0]['prefix_absdiff']} "
              f"suffix_absdiff={recs[0]['suffix_absdiff']} obj={recs[0].get('object')}", flush=True)

    print(f"DONE: {len(manifest)} clips ({len(manifest) // 2} pairs) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

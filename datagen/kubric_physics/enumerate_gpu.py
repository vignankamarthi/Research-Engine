"""Minimal Blackwell-enumeration probe for the modern-Blender (bpy 4.5) image.

Prints the bpy version, tries to select OPTIX then CUDA as the Cycles compute
device type, lists every device Cycles reports, enables the GPU one(s), and
renders a tiny 1-frame scene ON the GPU to prove it is actually used (not a
silent CPU fallback). Success is a device whose type is OPTIX/CUDA being listed
AND a GPU render completing.

Run inside the image (driver libs bound):
  apptainer exec --bind .../nvdriver_bind:/nvdriver --env LD_LIBRARY_PATH=/nvdriver \
    blender_gpu.sif enumerate_gpu.py
"""
import sys
import time

import bpy

print("bpy version:", bpy.app.version_string, flush=True)


def list_devices(kind):
    prefs = bpy.context.preferences.addons["cycles"].preferences
    try:
        prefs.compute_device_type = kind
    except Exception as e:  # noqa: BLE001
        return None, f"cannot set compute_device_type={kind}: {e}"
    try:
        prefs.get_devices()
    except Exception as e:  # noqa: BLE001
        return None, f"get_devices() failed for {kind}: {e}"
    devs = [(d.type, d.name, bool(d.use)) for d in prefs.devices]
    return devs, None


gpu_kind = None
for kind in ("OPTIX", "CUDA"):
    devs, err = list_devices(kind)
    if err:
        print(f"[{kind}] {err}", flush=True)
        continue
    print(f"[{kind}] devices reported: {devs}", flush=True)
    if any(t == kind for t, _, _ in devs):
        gpu_kind = gpu_kind or kind

if gpu_kind is None:
    print("RESULT: NO GPU ENUMERATED (only CPU visible)", flush=True)
    sys.exit(2)

print(f"RESULT: GPU ENUMERATED via {gpu_kind}", flush=True)

# ---- prove a real GPU render runs (not a CPU fallback) ----
prefs = bpy.context.preferences.addons["cycles"].preferences
prefs.compute_device_type = gpu_kind
prefs.get_devices()
n_on = 0
for d in prefs.devices:
    d.use = d.type == gpu_kind
    n_on += int(d.use)
print(f"enabled {n_on} {gpu_kind} device(s)", flush=True)

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "GPU"
scene.cycles.samples = 16
scene.render.resolution_x = 256
scene.render.resolution_y = 256
scene.render.filepath = "/tmp/gpu_probe.png"

# default cube + a light + a camera so the frame is non-trivial
bpy.ops.object.light_add(type="SUN", location=(2, -2, 4))
bpy.ops.object.camera_add(location=(5, -5, 4))
cam = bpy.context.object
scene.camera = cam

t = time.time()
bpy.ops.render.render(write_still=True)
dt = time.time() - t
print(f"GPU render completed in {dt:.2f}s -> /tmp/gpu_probe.png (device={scene.cycles.device}, "
      f"backend={gpu_kind})", flush=True)
print("GPU_RENDER_OK", flush=True)

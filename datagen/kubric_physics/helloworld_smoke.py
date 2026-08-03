"""Minimal headless Kubric render smoke (stands in for examples/helloworld.py, which
the kubruntu image does not bundle). Builds a tiny scene, renders ONE frame in Blender
background mode with NO display, and writes an RGBA PNG. Proves headless render works.

  apptainer exec --bind <base>:<base> kubruntu.sif python3 helloworld_smoke.py <out_png>
"""
import sys
import numpy as np
import kubric as kb
from kubric.renderer.blender import Blender

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/helloworld.png"

scene = kb.Scene(resolution=(256, 256))
scene.frame_start = 1
scene.frame_end = 1
renderer = Blender(scene, samples_per_pixel=16)
scene += kb.Cube(name="floor", scale=(10, 10, 0.1), position=(0, 0, -0.1), static=True,
                 material=kb.PrincipledBSDFMaterial(color=kb.Color(0.3, 0.3, 0.35)))
scene += kb.Sphere(name="ball", scale=1.0, position=(0, 0, 1.0),
                   material=kb.PrincipledBSDFMaterial(color=kb.Color(0.9, 0.2, 0.2)))
scene += kb.DirectionalLight(name="sun", position=(-1, -0.5, 3),
                             look_at=(0, 0, 0), intensity=2.5)
cam = kb.PerspectiveCamera(name="camera", position=(4, -4, 3), look_at=(0, 0, 1))
scene += cam
scene.camera = cam

frame = renderer.render_still()
rgba = np.asarray(frame["rgba"])
print("RGBA shape:", rgba.shape, "dtype:", rgba.dtype,
      "min/max:", int(rgba.min()), int(rgba.max()), flush=True)
kb.write_png(rgba, out)
print("WROTE", out, flush=True)
print("HELLOWORLD_OK", flush=True)

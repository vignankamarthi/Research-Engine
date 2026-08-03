"""Introspect the pinned kubric API inside the container before running the generator.

Prints what we depend on: version, Color helpers, render() signature, keyframe_insert,
simulator.run() return shape, and whether imageio/ffmpeg can encode mp4. Non-fatal:
each check is wrapped so we see the full picture in one pass.
"""
import inspect
import sys


def line(k, v):
    print(f"{k:32s} {v}", flush=True)


def check(name, fn):
    try:
        line(name, fn())
    except Exception as e:  # noqa: BLE001
        line(name, f"ERR {type(e).__name__}: {e}")


import kubric as kb  # noqa: E402

line("python", sys.version.split()[0])
line("kubric", getattr(kb, "__version__", "unknown"))
check("Color.from_hsv", lambda: kb.Color.from_hsv(0.1, 1, 1))
check("Color(*rgb).hsv", lambda: kb.Color(0.2, 0.4, 0.6).hsv)
check("Color(*rgb).rgb", lambda: kb.Color(0.2, 0.4, 0.6).rgb)
check("has Sphere", lambda: hasattr(kb, "Sphere"))
check("has Cube", lambda: hasattr(kb, "Cube"))
check("PrincipledBSDFMaterial", lambda: kb.PrincipledBSDFMaterial(color=kb.Color(1, 0, 0)))

from kubric.renderer import Blender  # noqa: E402
from kubric.simulator import PyBullet  # noqa: E402

check("Blender.render sig", lambda: str(inspect.signature(Blender.render)))
check("PyBullet.run sig", lambda: str(inspect.signature(PyBullet.run)))
check("Asset.keyframe_insert", lambda: str(inspect.signature(kb.core.Asset.keyframe_insert)))

# encoder availability
check("imageio", lambda: __import__("imageio").__version__)
try:
    import imageio_ffmpeg
    line("imageio_ffmpeg", imageio_ffmpeg.__version__)
except Exception as e:  # noqa: BLE001
    line("imageio_ffmpeg", f"ERR {e}")
import shutil  # noqa: E402
line("ffmpeg_bin", shutil.which("ffmpeg") or "MISSING")
print("PROBE_DONE", flush=True)

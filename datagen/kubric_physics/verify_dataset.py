"""Data-validity gate for a generated paired dataset.

Decodes every MP4 in the manifest and asserts the properties a downstream eval relies on:
  1. both clips of a pair are readable and have the expected frame count + resolution
  2. each pair has exactly one label=1 (possible) and one label=0 (impossible)
  3. the decoded prefix (frames before divergence) is visually identical (mp4 is lossy, so
     a small tolerance), i.e. the pair diverges ONLY at/after the injected event
  4. the suffix (frames from divergence on) actually differs (the violation is visible)

Run inside the container so imageio can decode mp4:
  apptainer exec --bind <base>:<base> --env PYTHONPATH=<base>/pylibs kubruntu.sif \
    python3 verify_dataset.py <dataset_dir>
"""
import json
import os
import sys

import numpy as np
import imageio.v2 as imageio

PREFIX_TOL = 2.5   # mean abs pixel diff allowed in the "identical" prefix (mp4 is lossy)
SUFFIX_MIN = 0.5   # mean abs pixel diff required somewhere in the suffix (violation visible)


def read_mp4(path):
    r = imageio.get_reader(path, "ffmpeg")
    frames = [f[..., :3] for f in r]
    r.close()
    return np.asarray(frames)


def main():
    root = sys.argv[1]
    manifest = json.load(open(os.path.join(root, "manifest.json")))

    # group records into pairs
    pairs = {}
    for rec in manifest:
        pairs.setdefault(rec["pair_id"], []).append(rec)

    n_ok = 0
    per_viol = {}
    failures = []
    for pair_id, recs in sorted(pairs.items()):
        try:
            assert len(recs) == 2, "pair does not have 2 records"
            labels = sorted(r["label"] for r in recs)
            assert labels == [0, 1], f"labels not balanced: {labels}"
            pos = next(r for r in recs if r["label"] == 1)
            imp = next(r for r in recs if r["label"] == 0)
            v = pos["violation_type"]
            nfr = pos["num_frames"]
            d = pos["divergence_frame"]                 # 1-based

            a = read_mp4(os.path.join(root, pos["clip_path"]))
            b = read_mp4(os.path.join(root, imp["clip_path"]))
            assert a.shape[0] == nfr and b.shape[0] == nfr, f"frame count {a.shape[0]}/{b.shape[0]} != {nfr}"
            assert list(a.shape[1:3]) == pos["resolution"], f"resolution {a.shape[1:3]}"

            per_frame = np.abs(a.astype(np.int16) - b.astype(np.int16)).reshape(nfr, -1).mean(1)
            prefix = per_frame[: d - 1]                 # frames strictly before divergence
            suffix = per_frame[d - 1:]
            prefix_max = float(prefix.max()) if len(prefix) else 0.0
            suffix_max = float(suffix.max())
            assert prefix_max <= PREFIX_TOL, f"prefix differs (max {prefix_max:.3f})"
            assert suffix_max >= SUFFIX_MIN, f"suffix identical (max {suffix_max:.3f})"

            per_viol.setdefault(v, 0)
            per_viol[v] += 1
            n_ok += 1
            print(f"OK  {pair_id:28s} div={d:2d} prefix_max={prefix_max:.3f} "
                  f"suffix_max={suffix_max:.3f}", flush=True)
        except Exception as e:  # noqa: BLE001
            failures.append((pair_id, str(e)))
            print(f"FAIL {pair_id}: {e}", flush=True)

    print("\n=== SUMMARY ===")
    print(f"pairs ok: {n_ok}/{len(pairs)}  by violation: {per_viol}")
    if failures:
        print(f"FAILURES: {failures}")
        sys.exit(1)
    print("ALL PAIRS VALID")


if __name__ == "__main__":
    main()

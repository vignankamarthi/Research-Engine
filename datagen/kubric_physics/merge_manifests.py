"""Merge per-pair / per-block manifests into a single manifest.json.

Handles both the one-pair-per-task shards (manifest_pair_*.json) and the
multi-pair-per-task block shards (manifest_block_*.json).

Run after a SLURM-array generation completes:
  python3 merge_manifests.py <DATASET_DIR>
"""
import glob
import json
import os
import sys


def main():
    root = sys.argv[1]
    shards = sorted(glob.glob(os.path.join(root, "manifest_pair_*.json"))
                    + glob.glob(os.path.join(root, "manifest_block_*.json")))
    if not shards:
        print(f"no per-pair/per-block manifests in {root}", flush=True)
        sys.exit(1)
    merged = []
    for s in shards:
        recs = json.load(open(s))
        merged.extend(recs)
        print(f"  {os.path.basename(s)}: {len(recs)} records", flush=True)
    # stable order by pair_id then role
    merged.sort(key=lambda r: (r["pair_id"], r["role"]))
    out = os.path.join(root, "manifest.json")
    with open(out, "w") as fh:
        json.dump(merged, fh, indent=2)
    print(f"MERGED {len(merged)} records ({len(merged)//2} pairs) -> {out}", flush=True)


if __name__ == "__main__":
    main()

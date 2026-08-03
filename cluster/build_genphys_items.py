"""Build the driver items.json from a generated-clip pool's manifest shards. Each item is
{key, answer (1 possible / 0 impossible), rel (ABSOLUTE clip path), condition (violation type)},
the exact shape intphys2_qwen.score_items + the genphys service path_for expect.

Usage:  python build_genphys_items.py <pool_dir> [out.json]
"""
import json
import sys
from pathlib import Path

pool = Path(sys.argv[1])
out = Path(sys.argv[2]) if len(sys.argv) > 2 else pool / "items.json"
entries, seen = [], set()
for mf in sorted(pool.glob("manifest*.json")):
    data = json.loads(mf.read_text())
    for e in data:
        cp = e["clip_path"]
        if cp in seen:
            continue
        seen.add(cp)
        entries.append({
            "key": f'{e["pair_id"]}_{e["role"]}',
            "answer": int(e["label"]),
            "rel": str((pool / cp).resolve()),   # absolute, so the service is pool-layout-agnostic
            "condition": e["violation_type"],
        })
out.write_text(json.dumps(entries))
n_pos = sum(1 for e in entries if e["answer"] == 1)
print(f"{len(entries)} items ({n_pos} possible / {len(entries) - n_pos} impossible) -> {out}")

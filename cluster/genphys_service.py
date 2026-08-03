"""Persistent Qwen scoring service for the engine's GENERATED paired physics clips. Identical scoring
to intphys2_service (binary possible/impossible via the continuation-lean scorer in intphys2_qwen),
but resolves each item to an ABSOLUTE generated-clip path, so it is decoupled from any pool layout.

Request JSON:  {"items": [{"key","answer","rel"}, ...], "ablate_keep": null|float, "untrained_seed": null|int}
Response JSON: {"scores": [..]}   # per-item P(correct), continuous [0,1], chance 0.5

The untrained-init arm is a genuinely weights-RANDOMIZED Qwen (the constructor, not pretrained), the
honest FLOOR. Stop by creating <rpc>/STOP. Run on a GPU node via sbatch."""
import json
import sys
import time
from pathlib import Path

BASE = Path("/work/neu/p2026_0016_neu/generated_physics")
sys.path.insert(0, str(BASE))
RPC = BASE / "rpc"
_UNTRAINED = {}  # seed -> a weights-randomized model, at most 2 cached


def path_for(item):
    # generated items carry `rel` = the ABSOLUTE clip path (build_genphys_items writes absolute paths).
    return item["rel"]


def _untrained_model(seed, ssv2_qwen):
    """A weights-RANDOMIZED Qwen (the constructor) at `seed`, cached (<=2) to bound GPU memory."""
    import torch
    from transformers import AutoConfig
    if seed in _UNTRAINED:
        return _UNTRAINED[seed]
    if len(_UNTRAINED) >= 2:
        _, old = _UNTRAINED.popitem()
        del old
        torch.cuda.empty_cache()
    torch.manual_seed(int(seed))
    cfg = AutoConfig.from_pretrained(ssv2_qwen.MODEL_ID)
    m = ssv2_qwen.Qwen2_5_VLForConditionalGeneration(cfg).to("cuda", dtype=torch.bfloat16).eval()
    _UNTRAINED[seed] = m
    return m


def main():
    import intphys2_qwen
    import ssv2_qwen
    RPC.mkdir(parents=True, exist_ok=True)
    print("loading Qwen ...", flush=True)
    model = ssv2_qwen.Qwen2_5_VLForConditionalGeneration.from_pretrained(
        ssv2_qwen.MODEL_ID, torch_dtype="auto", device_map="auto")
    processor = ssv2_qwen.AutoProcessor.from_pretrained(ssv2_qwen.MODEL_ID)
    print("SERVICE READY", flush=True)
    (RPC / "READY").write_text("ok")

    while not (RPC / "STOP").exists():
        reqs = sorted(RPC.glob("req_*.json"))
        if not reqs:
            time.sleep(1)
            continue
        for req in reqs:
            try:
                spec = json.loads(req.read_text())
                seed = spec.get("untrained_seed")
                m = model if seed is None else _untrained_model(seed, ssv2_qwen)
                scores = intphys2_qwen.score_items(
                    spec["items"], m, processor, path_for, ablate_keep=spec.get("ablate_keep"))
                out = RPC / req.name.replace("req_", "res_")
                tmp = RPC / (".tmp_" + req.name.replace("req_", "res_"))
                tmp.write_text(json.dumps({"scores": [float(s) for s in scores]}))
                tmp.replace(out)  # atomic: the Mac never reads a half-written response
                tag = "trained" if seed is None else f"untrained[{seed}]"
                print(f"served {req.name} ({tag}): {len(scores)} items, acc={scores.mean():.3f}", flush=True)
            except Exception as e:
                (RPC / req.name.replace("req_", "res_")).write_text(json.dumps({"error": str(e)}))
                print(f"error on {req.name}: {e}", flush=True)
            finally:
                req.unlink(missing_ok=True)
    print("STOP seen, shutting down", flush=True)


if __name__ == "__main__":
    main()

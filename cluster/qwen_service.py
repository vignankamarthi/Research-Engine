"""Persistent Qwen scoring service on the cluster. Loads Qwen2.5-VL-7B ONCE, then serves scoring
requests over the shared scratch filesystem (a simple file-RPC): it polls an RPC dir for `req_*.json`
files, scores the clips (optionally with a temporal-frequency ablation), writes `res_<id>.json`, and
removes the request. This lets the real-agent campaign run on Vignan's Mac (where `claude -p` works)
while the model scoring stays on the cluster GPU, the two-tier split made physical.

A request JSON: {"clips": [[clip_id, template], ...], "ablate_keep": null|float}
A response JSON: {"scores": [0.0, 1.0, ...]}
Stop by creating `<rpc>/STOP`. Run on a GPU node via sbatch."""
import json
import sys
import time
from pathlib import Path

SSV2 = Path("/work/neu/p2026_0016_neu/ssv2")
sys.path.insert(0, str(SSV2))
RPC = SSV2 / "rpc"


def main():
    import ssv2_qwen
    RPC.mkdir(exist_ok=True)
    classes = ssv2_qwen.load_classes()
    print(f"loading Qwen ({len(classes)} classes) ...", flush=True)
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
                pairs = [tuple(c) for c in spec["clips"]]
                scores = ssv2_qwen.score_clips(
                    pairs, classes, model, processor, ablate_keep=spec.get("ablate_keep"))
                out = RPC / req.name.replace("req_", "res_")
                out.write_text(json.dumps({"scores": [float(s) for s in scores]}))
                print(f"served {req.name}: {len(scores)} clips, acc={scores.mean():.3f}", flush=True)
            except Exception as e:  # a bad request must not kill the service
                err = RPC / req.name.replace("req_", "res_")
                err.write_text(json.dumps({"error": str(e)}))
                print(f"error on {req.name}: {e}", flush=True)
            finally:
                req.unlink(missing_ok=True)
    print("STOP seen, shutting down", flush=True)


if __name__ == "__main__":
    main()

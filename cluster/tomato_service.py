"""Persistent Qwen MCQ scoring service for TOMATO. Loads Qwen2.5-VL-7B once, then serves scoring
requests over the shared scratch file-RPC (same pattern as qwen_service.py, but MCQ instead of
classification). Resolves each item's video as videos/<demonstration_type>/<key>.mp4.

Request JSON: {"items": [{"key","demonstration_type","question","options","answer"}, ...],
               "ablate_keep": null|float}
Response JSON: {"scores": [0.0, 1.0, ...]}
Stop by creating <rpc>/STOP. Run on a GPU node via sbatch."""
import json
import sys
import time
from pathlib import Path

TOMATO = Path("/scratch/kamarthi_v_neu/tomato")
sys.path.insert(0, str(TOMATO))
RPC = TOMATO / "rpc"
VIDEOS = TOMATO / "videos"


def path_for(item):
    return str(VIDEOS / item["demonstration_type"] / f"{item['key']}.mp4")


def main():
    import ssv2_qwen
    import tomato_qwen
    RPC.mkdir(exist_ok=True)
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
                scores = tomato_qwen.score_items(
                    spec["items"], model, processor, path_for, ablate_keep=spec.get("ablate_keep"))
                out = RPC / req.name.replace("req_", "res_")
                out.write_text(json.dumps({"scores": [float(s) for s in scores]}))
                print(f"served {req.name}: {len(scores)} items, acc={scores.mean():.3f}", flush=True)
            except Exception as e:
                (RPC / req.name.replace("req_", "res_")).write_text(json.dumps({"error": str(e)}))
                print(f"error on {req.name}: {e}", flush=True)
            finally:
                req.unlink(missing_ok=True)
    print("STOP seen, shutting down", flush=True)


if __name__ == "__main__":
    main()

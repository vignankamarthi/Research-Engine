"""SSv2 recognition scorer on the cluster. Runs Qwen2.5-VL-7B on real SSv2 val clips, classifies the
action against the 174 labels, and returns per-item correctness (a real, model-derived, accuracy-unit
per-item score the gauntlet gates on). An optional temporal-frequency ablation (spectral mask along
time) is applied to the decoded frames before inference, so the same function serves the mechanism
experiment (full vs ablated). Probe mode (`python ssv2_qwen.py <n>`) scores n val clips and reports
accuracy, validating the inference path end to end."""
import json
import sys
from pathlib import Path

import decord
import numpy as np
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

ROOT = Path("/scratch/kamarthi_v_neu/ssv2")
VIDEOS = ROOT / "videos" / "20bn-something-something-v2"
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
NFRAMES = 8


def load_classes():
    labels = json.load(open(ROOT / "labels" / "labels.json"))
    return sorted(labels, key=lambda k: int(labels[k]))  # canonical class order


def debracket(t):
    return t.replace("[", "").replace("]", "")


def load_val(n, extracted_only=True):
    val = json.load(open(ROOT / "labels" / "validation.json"))
    out = []
    for e in val:
        cid, true = e["id"], debracket(e["template"])
        if not extracted_only or (VIDEOS / f"{cid}.webm").exists():
            out.append((cid, true))
        if len(out) >= n:
            break
    return out


def spectral_mask_time(frames, keep_fraction=0.5):
    """Zero high temporal-frequency bands along the time axis (the temporal_frequency ablation)."""
    x = frames.astype(np.float64)
    t = x.shape[0]
    if t < 2:
        return frames
    spec = np.fft.rfft(x, axis=0)
    n_keep = max(1, int(round(keep_fraction * spec.shape[0])))
    spec[n_keep:] = 0.0
    out = np.fft.irfft(spec, n=t, axis=0)
    return np.clip(out, 0, 255).astype(np.uint8)


def _decode_av(path, nframes):
    """pyav fallback (handles VP9 webm reliably where decord's threaded decoder chokes)."""
    import av
    container = av.open(path)
    frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    container.close()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    idx = np.linspace(0, len(frames) - 1, nframes).astype(int)
    return np.stack([frames[i] for i in idx])


def decode_frames(clip_path, nframes=NFRAMES):
    # decord's THREADED decoder errors on these VP9 webms ("Thread worker: Error sending packet"),
    # so force single-threaded, and fall back to pyav if decord still fails.
    try:
        vr = decord.VideoReader(str(clip_path), num_threads=1)
        idx = np.linspace(0, len(vr) - 1, nframes).astype(int)
        return vr.get_batch(idx).asnumpy()  # (T, H, W, C) uint8
    except Exception:
        return _decode_av(str(clip_path), nframes)


def classify(model, processor, frames, classes):
    from PIL import Image
    images = [Image.fromarray(f) for f in frames]
    label_block = "\n".join(f"- {c}" for c in classes)
    prompt = ("This is a short video of a hand manipulating objects. Classify the action using "
              f"EXACTLY ONE label from this list:\n{label_block}\n\nReply with ONLY the exact label.")
    messages = [{"role": "user", "content": [
        {"type": "video", "video": images},
        {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=40, do_sample=False)
    gen = out[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(gen, skip_special_tokens=True)[0].strip()


def match(reply, classes):
    r = reply.lower().strip().strip(".")
    for c in classes:
        if c.lower() == r:
            return c
    for c in classes:
        if c.lower() in r or r in c.lower():
            return c
    return reply  # unmatched -> counts as wrong


def score_clips(clip_ids_and_truths, classes, model, processor, ablate_keep=None, ablation_fn=None):
    """Per-item correctness over the given (clip_id, true_label) pairs. `ablate_keep` (e.g. 0.25)
    applies the temporal-frequency ablation; `ablation_fn(frames)->frames` applies an arbitrary
    constructed ablation (the campaign passes the red/blue-constructed ablation's `.apply`)."""
    scores = []
    for cid, true in clip_ids_and_truths:
        frames = decode_frames(VIDEOS / f"{cid}.webm")
        if ablation_fn is not None:
            frames = np.clip(np.asarray(ablation_fn(frames)), 0, 255).astype(np.uint8)
        elif ablate_keep is not None:
            frames = spectral_mask_time(frames, ablate_keep)
        pred = match(classify(model, processor, frames, classes), classes)
        ok = 1.0 if pred == true else 0.0
        scores.append(ok)
        print(f"{cid}: pred={pred[:36]!r} true={true[:36]!r} {'OK' if ok else 'x'}", flush=True)
    return np.array(scores, dtype=float)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    ablate = float(sys.argv[2]) if len(sys.argv) > 2 else None
    classes = load_classes()
    clips = load_val(n)
    print(f"{len(classes)} classes, scoring {len(clips)} clips, ablate_keep={ablate}", flush=True)
    print(f"cuda={torch.cuda.is_available()} device={torch.cuda.get_device_name(0)}", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    scores = score_clips(clips, classes, model, processor, ablate_keep=ablate)
    print(f"\nACCURACY {scores.mean():.3f} over {len(scores)} clips", flush=True)
    np.save(ROOT / f"probe_scores{'_ablated' if ablate else ''}.npy", scores)


if __name__ == "__main__":
    main()

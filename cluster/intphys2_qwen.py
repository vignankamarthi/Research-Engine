"""IntPhys 2 scorer on the cluster. IntPhys 2 is a BINARY intuitive-physics task (watch a synthetic
Unreal clip, judge whether what happens is physically possible or impossible), so this is the binary
variant of the TOMATO MCQ scorer: it prompts Qwen2.5-VL with the video, parses possible-vs-impossible,
and returns per-item correctness. Reuses the frame decoding + model plumbing from ssv2_qwen, and the
temporal-frequency (spectral-mask) ablation still applies to the decoded frames, so it serves the
mechanism experiment too. Chance is 0.5 (the Main split is balanced 506 possible / 506 impossible).

The video-key -> file-path mapping is injected (`path_for(item)`), resolved against the extracted
Main/Videos layout. Ground truth comes from the `type` column of Main/metadata.csv (X_Possible /
X_Impossible), loaded by `load_items`."""
import csv

import numpy as np

import ssv2_qwen  # decode_frames, spectral_mask_time, model classes, MODEL_ID


def _physics_prompt() -> str:
    return ("Watch this video of a physical scene. Decide whether what happens is physically POSSIBLE "
            "in the real world, or physically IMPOSSIBLE (it violates physics, for example an object "
            "vanishes, passes through a solid wall, or changes identity while hidden).\n\n"
            "Answer with exactly one word: 'possible' or 'impossible'.")


def _parse_possible(reply: str) -> int:
    """1 = possible, 0 = impossible, -1 = unparseable. 'impossible' is checked FIRST because it
    contains the substring 'possible', so a naive possible-first check would misread every impossible."""
    r = reply.strip().lower()
    if "impossible" in r:
        return 0
    if "possible" in r:
        return 1
    return -1


def classify_physics(model, processor, frames):
    from PIL import Image
    images = [Image.fromarray(f) for f in frames]
    messages = [{"role": "user", "content": [
        {"type": "video", "video": images},
        {"type": "text", "text": _physics_prompt()}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    from qwen_vl_utils import process_vision_info
    image_inputs, video_inputs = process_vision_info(messages)
    import torch
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    gen = out[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(gen, skip_special_tokens=True)[0].strip()


def load_items(metadata_csv: str, videos_dir: str = "Videos") -> list:
    """Build scorer items from a Main/Debug metadata.csv. Each item: {key, answer, condition, rel}.
    `answer` is 1 for a Possible clip, 0 for an Impossible one (read from the `type` column). `rel` is
    the clip path relative to the split dir, so `path_for` can join it to the extracted location."""
    items = []
    for row in csv.DictReader(open(metadata_csv)):
        possible = "possible" in row["type"].strip().lower()
        items.append({"key": row["name"], "answer": 1 if possible else 0,
                      "condition": row["condition"], "rel": row["file_name"]})
    return items


def score_items(items, model, processor, path_for, ablate_keep=None, ablation_fn=None,
                guess_on_fail=False):
    """Per-item correctness over IntPhys 2 items. Each item: {key, answer (1 possible / 0 impossible)}.
    `path_for(item) -> video file path`. Ablation applies to the decoded frames (mechanism pass).

    `guess_on_fail=True` is for the untrained FLOOR arm: a weights-randomized model that cannot produce
    a valid possible/impossible word GUESSES at chance (a per-item deterministic coin flip) instead of
    scoring a fail-closed wrong, giving a fair ~0.5 CHANCE baseline rather than an identically-zero
    vector that both trivializes the FLOOR residual and trips the FLOOR's stub-rejection guard. The
    trained arm keeps the fail-closed wrong (an unparseable answer is a real failure of the model)."""
    import random
    scores = []
    for it in items:
        frames = ssv2_qwen.decode_frames(path_for(it))
        if ablation_fn is not None:
            frames = np.clip(np.asarray(ablation_fn(frames)), 0, 255).astype(np.uint8)
        elif ablate_keep is not None:
            frames = ssv2_qwen.spectral_mask_time(frames, ablate_keep)
        reply = classify_physics(model, processor, frames)
        pred = _parse_possible(reply)
        if pred == -1 and guess_on_fail:
            pred = random.Random(str(it["key"])).randrange(2)
        ok = 1.0 if pred == int(it["answer"]) else 0.0
        scores.append(ok)
        print(f"{it['key']}: pred={pred} true={it['answer']} {'OK' if ok else 'x'}", flush=True)
    return np.array(scores, dtype=float)

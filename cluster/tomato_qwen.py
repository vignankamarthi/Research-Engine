"""TOMATO recognition scorer on the cluster. TOMATO is multiple-choice QA (question + options ->
pick one), so this is the MCQ variant of the SSv2 scorer: it prompts Qwen2.5-VL with the video + the
question + the numbered options, parses the chosen index, and returns per-item correctness. Reuses
the frame decoding + model plumbing from ssv2_qwen. The temporal-frequency (spectral-mask) ablation
still applies to the decoded frames, so this serves the mechanism experiment too.

The video-key -> file-path mapping is injected (`path_for(key)`), finalized once the video zip is
extracted and its layout is known."""
import re

import numpy as np

import ssv2_qwen  # decode_frames, spectral_mask_time, model classes, MODEL_ID


def _mcq_prompt(question, options):
    lines = "\n".join(f"{i}. {o}" for i, o in enumerate(options))
    return (f"Watch the video and answer this multiple-choice question.\n\nQuestion: {question}\n\n"
            f"Options:\n{lines}\n\nReply with ONLY the number of the single best option.")


def _parse_choice(reply, n_options):
    # first integer in the reply that is a valid option index; fail-closed to -1 (wrong)
    for tok in re.findall(r"-?\d+", reply):
        v = int(tok)
        if 0 <= v < n_options:
            return v
    return -1


def classify_mcq(model, processor, frames, question, options):
    from PIL import Image
    images = [Image.fromarray(f) for f in frames]
    messages = [{"role": "user", "content": [
        {"type": "video", "video": images},
        {"type": "text", "text": _mcq_prompt(question, options)}]}]
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


def score_items(items, model, processor, path_for, ablate_keep=None, ablation_fn=None,
                guess_on_fail=False):
    """Per-item correctness over TOMATO items. Each item: {question, options, answer, key}.
    `path_for(item) -> video file path`. Ablation applies to the decoded frames (mechanism pass).

    `guess_on_fail=True` is for the untrained FLOOR arm: a weights-randomized model that cannot
    produce a valid option index GUESSES at chance (a per-item deterministic random choice) instead
    of scoring a fail-closed wrong. This gives a fair CHANCE-level baseline (~1/n_options) rather than
    an identically-zero vector that both trivializes the FLOOR residual and trips the FLOOR's
    stub-rejection guard. The trained arm keeps the fail-closed wrong (an unparseable answer is a real
    failure of the model that claims the capability)."""
    import random
    scores = []
    for it in items:
        frames = ssv2_qwen.decode_frames(path_for(it))
        if ablation_fn is not None:
            frames = np.clip(np.asarray(ablation_fn(frames)), 0, 255).astype(np.uint8)
        elif ablate_keep is not None:
            frames = ssv2_qwen.spectral_mask_time(frames, ablate_keep)
        reply = classify_mcq(model, processor, frames, it["question"], list(it["options"]))
        pred = _parse_choice(reply, len(it["options"]))
        if pred == -1 and guess_on_fail:
            pred = random.Random(str(it["key"])).randrange(len(it["options"]))
        ok = 1.0 if pred == int(it["answer"]) else 0.0
        scores.append(ok)
        print(f"{it['key']}: pred={pred} true={it['answer']} {'OK' if ok else 'x'}", flush=True)
    return np.array(scores, dtype=float)

"""SSv2 recognition scorer on the cluster. Runs Qwen2.5-VL-7B on real SSv2 val clips, classifies the
action against the 174 labels, and returns a per-item LOGIT-LEAN score (the model's softmax
probability on the correct label, a continuous [0,1] accuracy-unit the gauntlet gates on) rather than
a greedy-decoded exact-match {0,1}. An optional temporal-frequency ablation (spectral mask along time)
is applied to the decoded frames before inference, so the same function serves the mechanism
experiment (full vs ablated). This module also holds the SHARED model plumbing the tomato / intphys2
scorers reuse: frame decode, the answer-position forward pass (`answer_logits`), the option-token
resolver (`first_token_ids`), and the message builder. Probe mode (`python ssv2_qwen.py <n>`) scores n
val clips and reports the mean P(correct), validating the inference path end to end."""
import json
import sys
from pathlib import Path

import decord
import numpy as np
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

import scoring_math
from scoring_math import spectral_mask_time  # re-exported: tomato/intphys2 call ssv2_qwen.spectral_mask_time

ROOT = Path("/work/neu/p2026_0016_neu/ssv2")
VIDEOS = ROOT / "videos" / "20bn-something-something-v2"
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
# 16 frames (was 8) so a localized physics/motion event is actually sampled and the temporal-frequency
# ablation has enough bins (rfft over 16 -> 9 bins) to remove a real band without collapsing to DC.
# Tradeoff: vision tokens and forward-pass memory/latency grow roughly linearly with the frame count;
# 16 is the safe default on a single H200, 32 is feasible if a run needs finer temporal resolution.
NFRAMES = 16


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


def build_video_messages(frames, prompt):
    """The Qwen chat message for a video + text prompt (shared by all three scorers)."""
    from PIL import Image
    images = [Image.fromarray(f) for f in frames]
    return [{"role": "user", "content": [
        {"type": "video", "video": images},
        {"type": "text", "text": prompt}]}]


def answer_logits(model, processor, messages):
    """ONE forward pass over the chat prompt, NO generation. Returns the next-token logit row at the
    answer position (a 1D numpy float array over the vocab). With add_generation_prompt=True the last
    input position predicts the FIRST answer token, so this row is what the logit-lean scorer reads
    the candidate-answer-token logits from. Full-seq logits are materialized here; fine on an H200 for
    one clip at a time (the batch is a single item, so position -1 is the true last token, no padding)."""
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs)
    return out.logits[0, -1, :].float().detach().cpu().numpy()


def continuation_logprobs(model, processor, messages, options):
    """Length-normalized CONTINUATION log-likelihood of each option string as the assistant's answer.

    The first-token read (`answer_logits` + `first_token_ids`) is invalid when the model's emitted
    surface form does not match the bare option word: Qwen answers "Possible" (capital), and
    "impossible" tokenizes to the generic prefix "im", so a first-token softmax reads the wrong logits
    (confirmed on-cluster, job 255434). This scores the WHOLE option instead. For each option it
    teacher-forces prompt + option through the model and averages the per-token log-probs of the
    option's own tokens, so the value is the model's mean log-likelihood of writing that exact word.

    Every option is scored the same way, so any constant surface-form cost cancels when the caller
    softmaxes these log-likelihoods into P(option). Length-normalization (dividing by the option's
    token count) keeps a longer option (e.g. "Impossible" as 2+ tokens) from being penalized against a
    shorter one. An untrained-init model's near-flat vocabulary geometry gives both options nearly the
    same mean log-prob, so P stays ~uniform: the honest FLOOR is preserved, same as the first-token
    path it replaces. One forward pass per option (two for the binary possible/impossible task)."""
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    # Prompt-only length, tokenized WITH the vision inputs so it matches the full forward's layout.
    prompt_inputs = processor(text=[prompt_text], images=image_inputs, videos=video_inputs,
                              padding=True, return_tensors="pt")
    base_len = int(prompt_inputs.input_ids.shape[1])
    out = []
    for opt in options:
        # Let the PROCESSOR build every tensor (input_ids, attention_mask, vision grid) consistently:
        # appending tokens by hand desyncs Qwen2.5-VL's get_rope_index (3D rope from the video grid).
        full = processor(text=[prompt_text + str(opt)], images=image_inputs, videos=video_inputs,
                         padding=True, return_tensors="pt").to(model.device)
        n = int(full.input_ids.shape[1]) - base_len   # option token count, in context
        if n < 1:
            raise ValueError(f"option {opt!r} added no tokens after the prompt")
        with torch.no_grad():
            logits = model(**full).logits[0]  # (L_full, vocab)
        # The logit at position base_len-1+j predicts the option token at base_len+j.
        total = 0.0
        for j in range(n):
            tid = int(full.input_ids[0, base_len + j])
            logrow = torch.log_softmax(logits[base_len - 1 + j].float(), dim=-1)
            total += float(logrow[tid])
        out.append(total / n)   # length-normalized
    return np.array(out, dtype=float)


def first_token_ids(processor, labels, require_unique=False):
    """The first sub-token id of each option label, used as that option's single answer token for the
    logit-lean softmax. `require_unique=True` (binary possible/impossible, MCQ indices) fails LOUD on a
    first-token collision, since there a collision would silently mis-score rather than being a real
    tie. For a free-label set (SSv2's 174 classes) collisions are expected and left in (option_probs
    splits the shared token's mass uniformly), so require_unique stays False."""
    tok = processor.tokenizer
    ids = []
    for lb in labels:
        enc = tok.encode(str(lb), add_special_tokens=False)
        if not enc:
            raise ValueError(f"empty tokenization for option label {lb!r}")
        ids.append(int(enc[0]))
    if require_unique and len(set(ids)) != len(ids):
        raise ValueError("answer options collide on their first token, logit-lean scoring would be "
                         f"ambiguous: {list(zip(labels, ids))}")
    return ids


def _ssv2_prompt(classes):
    label_block = "\n".join(f"- {c}" for c in classes)
    return ("This is a short video of a hand manipulating objects. Classify the action using "
            f"EXACTLY ONE label from this list:\n{label_block}\n\nReply with ONLY the exact label.")


def score_clips(clip_ids_and_truths, classes, model, processor, ablate_keep=None, ablation_fn=None):
    """Per-item LOGIT-LEAN score over the given (clip_id, true_label) pairs: P(correct label) from a
    single forward pass (softmax over the 174 labels' first-token logits), a continuous [0,1] value,
    for BOTH the trained and the untrained-init arm (the untrained model's near-uniform lean is the
    honest FLOOR). `ablate_keep` (e.g. 0.25) applies the temporal-frequency ablation; `ablation_fn(
    frames)->frames` applies an arbitrary constructed ablation (the campaign passes the red/blue-
    constructed ablation's `.apply`).

    NOTE (cluster-smoke item): the 174-way score uses each label's FIRST token. Labels that share a
    first token split that token's mass uniformly (option_probs handles it). This is the standard
    first-token approximation to full-label likelihood; the probe's mean P(correct) validates it."""
    class_ids = first_token_ids(processor, classes)  # free-label set: duplicate first tokens allowed
    index_of = {c: i for i, c in enumerate(classes)}
    prompt = _ssv2_prompt(classes)
    scores = []
    for cid, true in clip_ids_and_truths:
        if true not in index_of:
            raise ValueError(f"true label {true!r} not in the class list")
        frames = decode_frames(VIDEOS / f"{cid}.webm")
        if ablation_fn is not None:
            frames = np.clip(np.asarray(ablation_fn(frames)), 0, 255).astype(np.uint8)
        elif ablate_keep is not None:
            frames = spectral_mask_time(frames, ablate_keep)
        messages = build_video_messages(frames, prompt)
        probs = scoring_math.option_probs(answer_logits(model, processor, messages), class_ids)
        p = float(probs[index_of[true]])
        scores.append(p)
        print(f"{cid}: P(correct)={p:.3f} true={true[:36]!r}", flush=True)
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
    print(f"\nMEAN P(correct) {scores.mean():.3f} over {len(scores)} clips", flush=True)
    np.save(ROOT / f"probe_scores{'_ablated' if ablate else ''}.npy", scores)


if __name__ == "__main__":
    main()

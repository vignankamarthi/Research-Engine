"""IntPhys 2 scorer on the cluster, ALSO the scorer for the generated paired physics clips (same
binary possible/impossible question). IntPhys 2 is a BINARY intuitive-physics task (watch a clip,
judge whether what happens is physically possible or impossible). It scores each item by CONTINUATION
LEAN: the length-normalized log-likelihood of the whole word "Possible" vs the whole word
"Impossible" as the assistant's answer, softmaxed to P(correct answer), a continuous [0,1] score for
BOTH the trained and the untrained-init arm. Reuses the frame decoding + model plumbing from
ssv2_qwen, and the temporal-frequency (spectral-mask) ablation still applies to the decoded frames, so
it serves the mechanism experiment too. Chance is 0.5 (the Main split is balanced 506 possible / 506
impossible), and the untrained model's near-uniform lean IS that 0.5 FLOOR, no coin flip.

WHY CONTINUATION, NOT A SINGLE FIRST-TOKEN LOGIT (fixed 2026-08-02, on-cluster job 255434): the model
emits "Possible" (capital) as its actual answer token, not the bare lowercase "possible", and
"impossible" tokenizes to the generic prefix "im", so a first-token softmax read the wrong logits and
produced plausible-but-meaningless scores. Scoring the whole option word (`continuation_logprobs`)
matches the model's real surface form and is robust to capitalization + multi-token options.

The video-key -> file-path mapping is injected (`path_for(item)`), resolved against the extracted
Main/Videos layout. Ground truth comes from the `type` column of Main/metadata.csv (X_Possible /
X_Impossible), loaded by `load_items`."""
import csv

import numpy as np

import scoring_math
import ssv2_qwen  # decode_frames, spectral_mask_time, answer_logits, first_token_ids, build_video_messages

# The two answer options, in a fixed order. continuation_probs returns [P(possible), P(impossible)].
# CAPITALIZED to match the model's actual emitted surface form (it answers "Possible", not "possible");
# both options are scored the same way, so the shared surface-form cost cancels in the softmax.
_OPTIONS = ("Possible", "Impossible")


def _physics_prompt() -> str:
    return ("Watch this video of a physical scene. Decide whether what happens is physically POSSIBLE "
            "in the real world, or physically IMPOSSIBLE (it violates physics, for example an object "
            "vanishes, passes through a solid wall, or changes identity while hidden).\n\n"
            "Answer with exactly one word, either possible or impossible.")


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


def score_items(items, model, processor, path_for, ablate_keep=None, ablation_fn=None):
    """Per-item CONTINUATION-LEAN score over IntPhys 2 / generated physics items. Each item: {key,
    answer (1 possible / 0 impossible)}. `path_for(item) -> video file path`. Ablation applies to the
    decoded frames (mechanism pass). For each item: the length-normalized log-likelihood of "Possible"
    vs "Impossible" as the answer (two forward passes), softmaxed to P, score = P(correct answer).

    This is used for BOTH arms. The untrained-init (weights-randomized) FLOOR model is scored the
    same way; its near-uniform option log-likelihoods give a real ~0.5 chance FLOOR, so there is no
    `guess_on_fail` coin and no greedy-word yes-bias. The trained-minus-untrained residual the
    SEPARATION gate reads is a graded lean on the [0,1] scale."""
    prompt = _physics_prompt()
    scores = []
    for it in items:
        frames = ssv2_qwen.decode_frames(path_for(it))
        if ablation_fn is not None:
            frames = np.clip(np.asarray(ablation_fn(frames)), 0, 255).astype(np.uint8)
        elif ablate_keep is not None:
            frames = ssv2_qwen.spectral_mask_time(frames, ablate_keep)
        messages = ssv2_qwen.build_video_messages(frames, prompt)
        logprobs = ssv2_qwen.continuation_logprobs(model, processor, messages, _OPTIONS)
        probs = scoring_math.continuation_probs(logprobs)
        # probs = [P(possible), P(impossible)]; answer 1 -> P(possible), answer 0 -> P(impossible)
        p_correct = float(probs[0] if int(it["answer"]) == 1 else probs[1])
        scores.append(p_correct)
        print(f"{it['key']}: P(correct)={p_correct:.3f} true={it['answer']}", flush=True)
    return np.array(scores, dtype=float)

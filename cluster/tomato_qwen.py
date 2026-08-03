"""TOMATO recognition scorer on the cluster. TOMATO is multiple-choice QA (question + options ->
pick one), so this is the MCQ variant of the SSv2 scorer. It scores each item by LOGIT LEAN: one
forward pass, then softmax over the option-index answer-token logits, returning P(correct index) as a
continuous [0,1] score for BOTH the trained and the untrained-init arm. Reuses the frame decoding +
model plumbing from ssv2_qwen. The temporal-frequency (spectral-mask) ablation still applies to the
decoded frames, so this serves the mechanism experiment too.

The video-key -> file-path mapping is injected (`path_for(key)`), finalized once the video zip is
extracted and its layout is known."""
import numpy as np

import scoring_math
import ssv2_qwen  # decode_frames, spectral_mask_time, answer_logits, first_token_ids, build_video_messages


def _mcq_prompt(question, options):
    lines = "\n".join(f"{i}. {o}" for i, o in enumerate(options))
    return (f"Watch the video and answer this multiple-choice question.\n\nQuestion: {question}\n\n"
            f"Options:\n{lines}\n\nReply with ONLY the number of the single best option.")


def score_items(items, model, processor, path_for, ablate_keep=None, ablation_fn=None):
    """Per-item LOGIT-LEAN score over TOMATO items. Each item: {question, options, answer, key}.
    `path_for(item) -> video file path`. Ablation applies to the decoded frames (mechanism pass).
    For each item: one forward pass, softmax over the option-INDEX first-token logits (the digits
    0..n-1), score = P(correct index) in [0,1].

    Used for BOTH arms. The untrained-init (weights-randomized) FLOOR model is scored the same way;
    its near-uniform lean gives a real ~1/n_options chance FLOOR, so there is no `guess_on_fail`
    coin and no greedy-word bias. `first_token_ids(require_unique=True)` fails LOUD if the option
    indices are not single distinct tokens (n_options > 10 would collide, e.g. index 10's first
    token is the same as index 1's), rather than silently mis-scoring."""
    scores = []
    for it in items:
        options = list(it["options"])
        opt_ids = ssv2_qwen.first_token_ids(
            processor, [str(i) for i in range(len(options))], require_unique=True)
        frames = ssv2_qwen.decode_frames(path_for(it))
        if ablation_fn is not None:
            frames = np.clip(np.asarray(ablation_fn(frames)), 0, 255).astype(np.uint8)
        elif ablate_keep is not None:
            frames = ssv2_qwen.spectral_mask_time(frames, ablate_keep)
        messages = ssv2_qwen.build_video_messages(frames, _mcq_prompt(it["question"], options))
        probs = scoring_math.option_probs(ssv2_qwen.answer_logits(model, processor, messages), opt_ids)
        p_correct = float(probs[int(it["answer"])])
        scores.append(p_correct)
        print(f"{it['key']}: P(correct)={p_correct:.3f} true={it['answer']}", flush=True)
    return np.array(scores, dtype=float)

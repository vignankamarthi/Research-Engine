"""Step-71 acceptance gate: the LEADING-SPACE tokenization check for the logit-lean scorer.

The scorers read P(answer) from the answer-position logit of each option's FIRST token, where the
option token id comes from `first_token_ids(processor, ["possible", "impossible"])`, i.e. the
tokenization of the bare word "possible" with NO leading space. Qwen's BPE may instead emit
" possible" (leading space) as the actual first token of the assistant turn, which is a DIFFERENT
id. If so, the scorer reads the wrong logit and mis-scores every item.

This proves it empirically on the real model + real (sim) clips: for a handful of clips it compares
(a) the bare-word option ids the scorer uses, (b) the leading-space variant ids, and (c) the token
the model ACTUALLY emits first under greedy generate. Verdict:
  - greedy first token in the BARE-word set   -> the current scorer read is VALID.
  - greedy first token in the LEADING-SPACE set -> switch first_token_ids to the leading-space encoding.
  - neither (capitalization, a determiner, punctuation) -> the one-word constraint is not binding,
    continuation scoring is needed.

Usage on a GPU node (same env as the scorers):
    python cluster/leading_space_smoke.py <clip_dir> [n]
`clip_dir` holds the paired sim .mp4 clips (e.g. out_newtypes_smoke/clips).
"""

import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

import ssv2_qwen
import scoring_math
from intphys2_qwen import _physics_prompt

_OPTIONS = ("possible", "impossible")


def _first_id(tok, s):
    enc = tok.encode(s, add_special_tokens=False)
    return int(enc[0]) if enc else None


def main():
    clip_dir = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    clips = sorted(clip_dir.glob("*.mp4"))[:n]
    if not clips:
        raise SystemExit(f"no .mp4 clips under {clip_dir}")

    print(f"cuda={torch.cuda.is_available()} device={torch.cuda.get_device_name(0)}", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        ssv2_qwen.MODEL_ID, torch_dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(ssv2_qwen.MODEL_ID)
    tok = processor.tokenizer

    bare_ids = ssv2_qwen.first_token_ids(processor, _OPTIONS, require_unique=True)
    space_ids = [_first_id(tok, " " + w) for w in _OPTIONS]
    print(f"bare  ids {dict(zip(_OPTIONS, bare_ids))} -> {[tok.decode([i]) for i in bare_ids]!r}", flush=True)
    print(f"space ids {dict(zip(_OPTIONS, space_ids))} -> {[tok.decode([i]) for i in space_ids]!r}", flush=True)
    bare_set, space_set = set(bare_ids), set(space_ids)

    prompt = _physics_prompt()
    n_bare = n_space = n_other = 0
    for clip in clips:
        frames = ssv2_qwen.decode_frames(str(clip))
        messages = ssv2_qwen.build_video_messages(frames, prompt)

        # (c) what the model actually emits first, greedy.
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = ssv2_qwen.process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=3, do_sample=False)
        emitted = gen[0, inputs.input_ids.shape[1]:]
        first_id = int(emitted[0])
        first_str = tok.decode([first_id])

        # (a)/(b) the logit read the scorer uses, and the argmax over the full vocab.
        logits = ssv2_qwen.answer_logits(model, processor, messages)
        argmax_id = int(np.argmax(logits))
        probs = scoring_math.option_probs(logits, bare_ids)

        if first_id in bare_set:
            bucket, n_bare = "BARE", n_bare + 1
        elif first_id in space_set:
            bucket, n_space = "SPACE", n_space + 1
        else:
            bucket, n_other = "OTHER", n_other + 1
        match = "argmax==greedy" if argmax_id == first_id else f"argmax={argmax_id}({tok.decode([argmax_id])!r})!=greedy"
        print(f"{clip.name:42s} greedy_first={first_id}({first_str!r}) [{bucket}] {match} "
              f"P(poss)={probs[0]:.3f} P(imp)={probs[1]:.3f} full={tok.decode(emitted)!r}", flush=True)

    print(f"\nSUMMARY over {len(clips)} clips: BARE={n_bare} SPACE={n_space} OTHER={n_other}", flush=True)
    if n_bare == len(clips):
        print("VERDICT: logit-lean read VALID -- the scorer's bare-word first_token_ids match greedy.", flush=True)
    elif n_space == len(clips):
        print("VERDICT: SWITCH first_token_ids to leading-space encoding (\" possible\"/\" impossible\").", flush=True)
    else:
        print("VERDICT: MIXED/OTHER -- the one-word constraint is not binding; continuation scoring needed.", flush=True)


if __name__ == "__main__":
    main()

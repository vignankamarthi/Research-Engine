"""Verify the CONTINUATION scorer reads the real model faithfully (the step-71 acceptance gate, take 2).

The first-token read failed job 255434 (model emits "Possible", not the bare "possible" token). This
checks the replacement: for each clip it computes the new continuation P(possible) (the exact path
`intphys2_qwen.score_items` now uses) AND what the model actually emits under greedy generation, then
checks they AGREE, P(possible) > 0.5 iff the model greedily answers a possible-word. Agreement on
every clip proves the scorer faithfully reflects the model's own belief (it is not about whether the
model is good at physics, only whether we read it correctly).

Usage on a GPU node (scorer env):  python cluster/continuation_verify.py <clip_dir> [n]
"""

import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

import scoring_math
import ssv2_qwen
from intphys2_qwen import _OPTIONS, _physics_prompt


def main():
    clip_dir = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    clips = sorted(clip_dir.glob("*.mp4"))[:n]
    if not clips:
        raise SystemExit(f"no .mp4 clips under {clip_dir}")

    print(f"cuda={torch.cuda.is_available()} device={torch.cuda.get_device_name(0)}", flush=True)
    print(f"options={_OPTIONS}", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        ssv2_qwen.MODEL_ID, torch_dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(ssv2_qwen.MODEL_ID)
    prompt = _physics_prompt()

    n_agree = 0
    p_poss_clips, p_imp_clips = [], []
    for clip in clips:
        frames = ssv2_qwen.decode_frames(str(clip))
        messages = ssv2_qwen.build_video_messages(frames, prompt)

        # New continuation scorer path (identical to score_items).
        logprobs = ssv2_qwen.continuation_logprobs(model, processor, messages, _OPTIONS)
        probs = scoring_math.continuation_probs(logprobs)
        p_possible = float(probs[0])

        # What the model actually says, greedy.
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = ssv2_qwen.process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=3, do_sample=False)
        emitted = processor.tokenizer.decode(gen[0, inputs.input_ids.shape[1]:]).strip().lower()
        greedy_possible = emitted.startswith("possible") or (
            "possible" in emitted and "impossible" not in emitted)

        agree = (p_possible > 0.5) == greedy_possible
        n_agree += int(agree)
        (p_poss_clips if "_possible" in clip.name else p_imp_clips).append(p_possible)
        print(f"{clip.name:42s} P(possible)={p_possible:.3f} logprobs={np.round(logprobs,2).tolist()} "
              f"greedy={emitted!r} agree={'YES' if agree else 'NO'}", flush=True)

    print(f"\nAGREEMENT with greedy: {n_agree}/{len(clips)} clips", flush=True)
    if p_poss_clips and p_imp_clips:
        print(f"mean P(possible): on POSSIBLE clips={np.mean(p_poss_clips):.3f}  "
              f"on IMPOSSIBLE clips={np.mean(p_imp_clips):.3f} "
              f"(separation={np.mean(p_poss_clips) - np.mean(p_imp_clips):+.3f}; "
              f"model skill is separate from read-faithfulness)", flush=True)
    if n_agree == len(clips):
        print("VERDICT: continuation scorer FAITHFUL -- P(possible) agrees with the model's own answer "
              "on every clip. The read is correct; wire it into the marathon.", flush=True)
    else:
        print("VERDICT: DISAGREEMENT remains -- inspect the mismatched clips before trusting the scorer.",
              flush=True)


if __name__ == "__main__":
    main()

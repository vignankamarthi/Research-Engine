"""Real-GPU box-scoring smoke for AICR (Blackwell). Proves the confirmatory scoring
path works on a real GPU: a real torch model forward pass over N synthetic box items,
with a planted input signal, yields per-item effect scores. Runs on the cluster only
(needs torch on a GPU); the local Mac suite never imports this."""
import json
import sys

import numpy as np
import torch
import torch.nn as nn


def main(out_path: str, n: int = 400, seed: int = 0) -> None:
    if not torch.cuda.is_available():
        print(json.dumps({"error": "cuda_unavailable"}))
        sys.exit(2)
    dev = torch.device("cuda")
    torch.manual_seed(seed)
    d_in = 64
    model = nn.Sequential(nn.Linear(d_in, 256), nn.GELU(), nn.Linear(256, 1)).to(dev).eval()

    direction = torch.randn(d_in, device=dev)
    direction = direction / direction.norm()
    g = torch.Generator(device=dev).manual_seed(seed + 1)
    base = torch.randn(n, d_in, device=dev, generator=g)

    with torch.no_grad():
        control = model(base).squeeze(-1)
        treatment = model(base + 0.6 * direction).squeeze(-1)  # planted input signal
    effect = (treatment - control).float().cpu().numpy().astype(np.float64)

    np.save(out_path, effect)
    print(json.dumps({
        "device": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "n": int(n),
        "mean_effect": float(effect.mean()),
        "std": float(effect.std()),
        "out": out_path,
    }))


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/work/neu/p2026_0016_neu/re_smoke_effect.npy"
    main(out)

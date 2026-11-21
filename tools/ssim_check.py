#!/usr/bin/env python3
"""Compare two images with a windowed structural similarity (SSIM) score."""
import argparse
import sys

import numpy as np
from PIL import Image


def ssim_channel(ref: np.ndarray, cand: np.ndarray, data_range: float = 255.0) -> float:
    """Return the mean of uniform-window SSIM scores for one channel."""
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    scores = []
    block = 64
    for h in range(0, ref.shape[0] - block + 1, block):
        for w in range(0, ref.shape[1] - block + 1, block):
            r = ref[h:h + block, w:w + block].astype(np.float64)
            c = cand[h:h + block, w:w + block].astype(np.float64)
            mu_r, mu_c = r.mean(), c.mean()
            var_r, var_c = r.var(), c.var()
            cov = ((r - mu_r) * (c - mu_c)).mean()
            scores.append(((2 * mu_r * mu_c + C1) * (2 * cov + C2)) /
                          ((mu_r * mu_r + mu_c * mu_c + C1) * (var_r + var_c + C2)))
    return float(np.mean(scores))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()

    ref = np.asarray(Image.open(args.reference).convert("RGB"))
    cand = np.asarray(Image.open(args.candidate).convert("RGB"))
    if ref.shape != cand.shape:
        print(f"FAIL: image dimensions differ {ref.shape} vs {cand.shape}")
        return 2

    per_channel = [ssim_channel(ref[..., ch], cand[..., ch]) for ch in range(ref.shape[2])]
    ssim = float(np.mean(per_channel))
    verdict = "pass" if ssim >= args.threshold else "fail"
    print(f'{{"metric":"ssim","value":{ssim:.6f},"threshold":{args.threshold},'
          f'"per_channel":[{", ".join(f"{s:.4f}" for s in per_channel)}],"verdict":"{verdict}"}}')
    return 0 if verdict == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with Image.open(args.reference) as image:
        reference = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    with Image.open(args.candidate) as image:
        candidate = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    if reference.shape != candidate.shape:
        raise ValueError(f"Image shape mismatch: {reference.shape} != {candidate.shape}")
    error = candidate - reference
    mse = float(np.mean(error.astype(np.float64) ** 2))
    report = {
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "shape": list(reference.shape),
        "mae": float(np.mean(np.abs(error))),
        "rmse": mse**0.5,
        "psnr_db": None if mse == 0 else 10 * np.log10(1 / mse),
        "pixel_identical": bool(mse == 0),
        "pixel_correlation": float(np.corrcoef(reference.ravel(), candidate.ravel())[0, 1]),
        "scope": "Similarity between two implementations, not an absolute image-quality score",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

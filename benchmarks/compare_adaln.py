import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from onyx.reference.dit import curve_coordinates, original_precompute, original_time_embedding, precompute
from onyx.reference.weights import Weights


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/reference-tests/original-adaln-comparison.json")
    args = parser.parse_args()
    sigmas = json.loads((ROOT / "artifacts/models/h3-w4/pdd_config.json").read_text())["sigmas"]
    original = Weights(ROOT / "artifacts/models/h3-w4/transformer.onxpack")
    community = Weights(ROOT / "artifacts/models/community-int8/diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors")
    temb = original_time_embedding(original, sigmas)
    a = original_precompute(original, "blocks.0", temb)
    af = original_precompute(original, "final_layer", temb, final=True)
    coordinates = curve_coordinates(community, sigmas[:-1])
    b = precompute(community, "blocks.0", coordinates, 3, 6)
    bf = precompute(community, "final_layer", coordinates, 1, 2)
    mx.eval(a, af, b, bf)
    report = {"parts": []}
    for index, name in enumerate(("shift_msa", "scale_msa", "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp")):
        x = np.asarray(a[:, :, index].astype(mx.float32))
        y = np.asarray(b[:, :, index].astype(mx.float32))
        report["parts"].append({
            "name": name,
            "correlation": float(np.corrcoef(x.ravel(), y.ravel())[0, 1]),
            "original_std": float(x.std()),
            "community_std": float(y.std()),
            "rmse": float(np.sqrt(np.mean((x - y) ** 2))),
            "original_mean": float(x.mean()),
            "community_mean": float(y.mean()),
        })
    x = np.asarray(af.astype(mx.float32))
    y = np.asarray(bf.astype(mx.float32))
    report["final"] = {
        "correlation": float(np.corrcoef(x.ravel(), y.ravel())[0, 1]),
        "original_std": float(x.std()),
        "community_std": float(y.std()),
        "rmse": float(np.sqrt(np.mean((x - y) ** 2))),
    }
    original.close()
    community.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

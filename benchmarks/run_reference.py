import hashlib
import json
import subprocess
import sys
from pathlib import Path

from onyx.artifacts import canonical_json


ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "external/h3-image-mlx/src/h3_image_mlx/weights.json").read_text())
    manifest_digest = hashlib.sha256(canonical_json(manifest["files"])).hexdigest()
    output_root = ROOT / "artifacts/benchmarks/onyx-reference-h3"
    command = [
        sys.executable, "-m", "onyx.replay",
        "--case", str(ROOT / "benchmarks/requests/teapot-1024.json"),
        "--output-root", str(output_root),
        "--backend-version", "onyx-reference-mlx-w8a16",
        "--plan-id", "reference.w8a16.resident.tiled-vae",
        "--numeric-profile", "w8a16.bf16-fp16",
        "--quantization-profile", "comfy.int8-convrot.w8a16-reference",
        "--resolved-source-revision", "b8ebf14be96f02e5c58960620a101de71b0bd9f3",
        "--manifest-digest", manifest_digest,
        "--rng-algorithm", "numpy.pcg64.standard_normal",
        "--rng-version", "2.5.4",
        "--cache-state", "unspecified",
        "--", sys.executable, "-m", "onyx.reference.generate",
        "--models", str(ROOT / "artifacts/models/community-int8"),
        "--output", "{output}", "--prompt", "{prompt}",
        "--width", "{width}", "--height", "{height}", "--seed", "{seed}",
        "--dit-mode", "resident",
    ]
    result = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    print(result.stdout, end="")
    (output_root / "latest.json").write_text(json.dumps({"run": json.loads(result.stdout)}, indent=2) + "\n")


if __name__ == "__main__":
    main()

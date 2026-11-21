import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    package = json.loads((ROOT / "artifacts/models/h3-w4/package-report.json").read_text())
    output_root = ROOT / "artifacts/benchmarks/onyx-w4-reference-h3"
    command = [
        sys.executable, "-m", "onyx.replay",
        "--case", str(ROOT / "benchmarks/requests/teapot-1024.json"),
        "--output-root", str(output_root),
        "--backend-version", "onyx-reference-mlx-w4",
        "--plan-id", "reference.w4bf16.stream.original-pdd",
        "--numeric-profile", "w4.bf16.fp32-sensitive",
        "--quantization-profile", "onyx-w4s64-linear-v1",
        "--resolved-source-revision", package["source_revision"],
        "--manifest-digest", package["semantic_digest"],
        "--rng-algorithm", "numpy.pcg64.standard_normal",
        "--rng-version", "2.5.4",
        "--cache-state", "unspecified",
        "--", sys.executable, "-m", "onyx.reference.generate",
        "--models", str(ROOT / "artifacts/models/h3-w4"),
        "--output", "{output}", "--prompt", "{prompt}",
        "--width", "{width}", "--height", "{height}", "--seed", "{seed}",
        "--dit-mode", "stream", "--original-w4",
    ]
    result = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    print(result.stdout, end="")
    (output_root / "latest.json").write_text(json.dumps({"run": json.loads(result.stdout)}, indent=2) + "\n")


if __name__ == "__main__":
    main()

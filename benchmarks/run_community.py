import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

from onyx.artifacts import canonical_json


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "benchmarks/requests/teapot-1024.json"
OUT = ROOT / "artifacts/benchmarks/community-h3"
COMMUNITY = ROOT / "external/h3-image-mlx"
MODELS = ROOT / "artifacts/models/community-int8"


def run(expected=None):
    command = [
        sys.executable, "-m", "onyx.replay",
        "--case", str(CASE), "--output-root", str(OUT),
        "--backend-version", "eccdf321cf009e7c636253f7a3a477f076b8314c",
        "--plan-id", "community.w8a8.resident",
        "--numeric-profile", "community.w8a8.fp16-bf16",
        "--quantization-profile", "comfy.int8-convrot.v1",
        "--resolved-source-revision", "b8ebf14be96f02e5c58960620a101de71b0bd9f3",
        "--manifest-digest", MANIFEST_DIGEST,
        "--rng-algorithm", "numpy.pcg64.standard_normal",
        "--rng-version", NUMPY_VERSION,
        "--cache-state", "unspecified",
    ]
    if expected:
        command += ["--expect-output-sha256", expected]
    command += [
        "--", str(COMMUNITY / ".venv/bin/h3-image-mlx"), "generate",
        "--models", str(MODELS), "--output", "{output}",
        "--prompt", "{prompt}", "--width", "{width}", "--height", "{height}",
        "--seed", "{seed}", "--backend", "w8a8", "--dit-mode", "resident",
        "--metrics", "{run_dir}/community-metrics.json",
    ]
    result = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True, env={**os.environ, "ONYX_REPLAY": "1"})
    return json.loads(result.stdout)


def main():
    subprocess.run([str(COMMUNITY / ".venv/bin/h3-image-mlx"), "verify", "--models", str(MODELS)], check=True)
    runs = []
    expected = None
    for index in range(4):
        print(f"Community generation {index + 1}/4", flush=True)
        result = run(expected)
        expected = result["output_sha256"]
        runs.append(result)
    measured = runs[1:]
    seconds = [item["seconds"] for item in measured]
    report = {
        "case_id": "h3-teapot-1024-seed42-v1",
        "implementation": "Anionex/h3-image-mlx",
        "implementation_revision": "eccdf321cf009e7c636253f7a3a477f076b8314c",
        "weights_source": "ModelScope Comfy-Org/MiniMax-H3-Image",
        "weights_revision": "b8ebf14be96f02e5c58960620a101de71b0bd9f3",
        "weights_manifest_digest": MANIFEST_DIGEST,
        "precision": "INT8 ConvRot W8A8",
        "dit_mode": "resident",
        "target_id": "mac17-14-m5-max-36gb",
        "warmups": 1,
        "measured_runs": 3,
        "process_model": "new CLI and isolated stage processes per run; OS page cache not purged",
        "seconds_median": statistics.median(seconds),
        "seconds_p95": max(seconds),
        "peak_bytes_max": max(item["peak_bytes"] for item in measured),
        "all_outputs_identical": len({item["output_sha256"] for item in runs}) == 1,
        "output_rgb_or_png_hash_scope": "PNG file SHA-256",
        "runs": runs,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "runs"}, indent=2))


if __name__ == "__main__":
    manifest = json.loads((COMMUNITY / "src/h3_image_mlx/weights.json").read_text())
    MANIFEST_DIGEST = hashlib.sha256(canonical_json(manifest["files"])).hexdigest()
    NUMPY_VERSION = subprocess.check_output([str(COMMUNITY / ".venv/bin/python"), "-c", "import numpy; print(numpy.__version__)"], text=True).strip()
    main()

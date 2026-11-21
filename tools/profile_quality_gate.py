#!/usr/bin/env python3
"""Generate a profile candidate and compare it with a reference image using SSIM.

Run with --make-reference to generate a max-quality reference on the first run.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROMPT_DEFAULT = "A close-up product photograph of one bright yellow enamel teapot on a wooden table"


def generate(profile: str, prompt: str, output: Path, seed: int) -> dict:
    started = time.monotonic()
    result = subprocess.run(
        [sys.executable, "-m", "onyx", "native-generate", "--models", str(ROOT / "artifacts/models/h3-w4"),
         "--prompt", prompt, "--output", str(output), "--profile", profile, "--seed", str(seed)],
        cwd=ROOT, capture_output=True, text=True, timeout=1800)
    if result.returncode:
        raise RuntimeError(f"native-generate ({profile}) failed: {result.stderr.strip()}")
    elapsed = time.monotonic() - started
    return {"profile": profile, "output": str(output), "wall_s": round(elapsed, 2)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default="speed-focus")
    parser.add_argument("--prompt", default=PROMPT_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--reference", type=Path, default=ROOT / "artifacts/e2e-baseline.png")
    parser.add_argument("--make-reference", action="store_true",
                        help="Generate a max-quality reference if it is missing")
    args = parser.parse_args()

    report = {"protocol": "onyx.profile-quality-gate.v1", "candidate": args.candidate, "seed": args.seed}
    if args.make_reference and not args.reference.exists():
        report["reference_generation"] = generate("max-quality", args.prompt, args.reference, args.seed)

    candidate_png = ROOT / f"artifacts/e2e-{args.candidate}.png"
    report["candidate_generation"] = generate(args.candidate, args.prompt, candidate_png, args.seed)

    ssim = subprocess.run([sys.executable, str(ROOT / "tools/ssim_check.py"),
                           str(args.reference), str(candidate_png), "--threshold", str(args.threshold)],
                          capture_output=True, text=True, cwd=ROOT)
    if ssim.returncode not in (0, 1):
        raise RuntimeError(f"ssim_check failed: {ssim.stderr.strip()}")
    report["ssim"] = json.loads(ssim.stdout.strip())
    report["verdict"] = report["ssim"]["verdict"]
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["verdict"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())

"""Measure execution profiles with an interleaved VAE-stage benchmark.

Write an onyx.profile-matrix.v1 JSON file containing latency, peak process
memory, and output hashes for tools/check_profile_gates.py.
"""

import argparse
import hashlib
import json
import statistics
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import psutil
import numpy as np
from safetensors.numpy import save_file

from onyx.profiles import DEFAULT_REGISTRY_PATH, load_registry, registry_digest, resolve_profile, stage_weights
from onyx.replay import ProcessMonitor


ROOT = Path(__file__).resolve().parents[1]
MATRIX_SCHEMA = "onyx.profile-matrix.v1"


def git_identity() -> dict:
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    return {"source_revision": revision, "source_dirty": dirty}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_latent(path: Path, width: int, height: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((24, height // 16, width // 16), dtype=np.float32)
    save_file({"latent": latent}, str(path))


def parse_size(text: str) -> tuple[int, int]:
    width, _, height = text.lower().partition("x")
    width, height = int(width), int(height)
    if width % 32 or height % 32:
        raise SystemExit(f"Size must be multiples of 32: {text}")
    return width, height


def run_stage_sampled(native_bin: Path, stage: str, arguments: list[str], output: Path, profile: str, weights: str) -> dict:
    """Run one native stage and sample peak memory across its process tree."""
    command = [str(native_bin), stage, *arguments, "--profile", profile, "--weights", weights]
    began = time.monotonic()
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
        monitor = ProcessMonitor(psutil.Process(process.pid))
        with monitor:
            stdout, stderr = process.communicate()
        elapsed = time.monotonic() - began
    if process.returncode:
        raise RuntimeError(f"Native {stage} failed for profile {profile}: {stderr.strip()}")
    stage_line = None
    for line in stdout.splitlines():
        candidate = line.strip()
        if not candidate.startswith("{"):
            continue
        try:
            details = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if details.get("stage") == stage:
            stage_line = details
            break
    if not stage_line:
        raise RuntimeError(f"Native {stage} did not echo a stage JSON line")
    if stage_line.get("profile") != profile or stage_line.get("weights") != weights:
        raise RuntimeError(f"Native {stage} echo mismatch: {stage_line}")
    if not output.is_file():
        raise RuntimeError(f"Native {stage} did not produce {output}")
    if monitor.samples == 0:
        raise RuntimeError("ProcessMonitor collected no samples")
    # Match replay.py by falling back to RSS if physical footprint is unavailable.
    peak_process_bytes = int(monitor.peak_footprint or monitor.peak_rss)
    return {
        "stage_seconds": float(stage_line["seconds"]),
        "monitor_seconds": elapsed,
        "peak_process_bytes": peak_process_bytes,
        "peak_footprint_bytes": int(monitor.peak_footprint),
        "peak_rss_bytes": int(monitor.peak_rss),
        "memory_samples": monitor.samples,
    }


def summarize(registry: dict, profile_ids: list[str], sizes: list[tuple[int, int]], runs: list[dict]) -> dict:
    summaries: dict = {}
    for profile_id in profile_ids:
        gates = registry["profiles"][profile_id]["gates"]
        for width, height in sizes:
            group = [run for run in runs if run["profile"] == profile_id and run["width"] == width and run["height"] == height]
            if not group:
                continue
            summaries[f"{profile_id}@{width}x{height}"] = {
                "samples": len(group),
                "stage_seconds_median": statistics.median(run["stage_seconds"] for run in group),
                "peak_process_bytes_max": max(run["peak_process_bytes"] for run in group),
                "output_sha256": sorted({run["output_sha256"] for run in group}),
                "min_samples_required": gates["min_samples"],
            }
    return summaries


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="onyx-profile-matrix")
    parser.add_argument("--stage", choices=("vae",), default="vae", help="Stage to measure")
    parser.add_argument("--profiles", default="max-quality,memory-min,speed-focus", help="Comma-separated profiles to measure in rotation")
    parser.add_argument("--sizes", default="1024x1024", help="Comma-separated WxH resolutions")
    parser.add_argument("--runs", type=int, default=3, help="Measured repeats per profile and resolution")
    parser.add_argument("--warmups", type=int, default=1, help="Warmup rounds per resolution, excluded from results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vae-model", type=Path, default=ROOT / "artifacts/models/h3-image-source/vae/diffusion_pytorch_model.safetensors")
    parser.add_argument("--vae-config", type=Path, default=ROOT / "artifacts/models/h3-image-source/vae/config.json")
    parser.add_argument("--native-bin", type=Path, default=ROOT / ".build/onyx-native")
    parser.add_argument("--target", type=Path, default=ROOT / "config/targets/m5-max-36gb.json")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--label", default="profile-stage-matrix")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/results/profile-stage-matrix-v1.json")
    args = parser.parse_args(argv)

    registry = load_registry(args.registry)
    profile_ids = [item.strip() for item in args.profiles.split(",") if item.strip()]
    for profile_id in profile_ids:
        resolve_profile(registry, profile_id)
    sizes = [parse_size(item) for item in args.sizes.split(",")]
    if args.runs < 1:
        parser.error("--runs must be >= 1")
    native_bin = args.native_bin.resolve()
    if not native_bin.is_file():
        raise FileNotFoundError(f"Native executable not found: {native_bin}")
    for path in (args.vae_model, args.vae_config):
        if not path.is_file():
            raise FileNotFoundError(f"Required model file missing: {path}")

    target = json.loads(args.target.read_text())
    identity = git_identity()
    workspace = ROOT / "artifacts/benchmarks/onyx-profiles" / args.label
    workspace.mkdir(parents=True, exist_ok=True)

    latent_paths: dict[tuple[int, int], Path] = {}
    for width, height in sizes:
        latent_path = workspace / f"latent-{width}x{height}-seed{args.seed}.safetensors"
        if not latent_path.is_file():
            write_latent(latent_path, width, height, args.seed)
        latent_paths[(width, height)] = latent_path

    runs: list[dict] = []
    total = len(sizes) * (args.runs + args.warmups) * len(profile_ids)
    completed = 0
    started_at = datetime.now(timezone.utc).isoformat()
    for width, height in sizes:
        # Rotate the profile order each round to reduce thermal bias.
        for iteration in range(args.runs + args.warmups):
            offset = iteration % max(1, len(profile_ids))
            order = profile_ids[offset:] + profile_ids[:offset]
            warmup = iteration < args.warmups
            for profile_id in order:
                resolved = resolve_profile(registry, profile_id)
                weights = stage_weights(profile_id, resolved["knobs"], args.stage)
                output = workspace / f"{profile_id}-{width}x{height}-s{iteration}.png"
                arguments = [
                    "--model", str(args.vae_model),
                    "--latent", str(latent_paths[(width, height)]),
                    "--config", str(args.vae_config),
                    "--output", str(output),
                ]
                completed += 1
                label = "warmup" if warmup else f"sample {iteration - args.warmups}"
                print(f"[{completed}/{total}] {profile_id} {width}x{height} {label} ({weights})", flush=True)
                result = run_stage_sampled(native_bin, args.stage, arguments, output, profile_id, weights)
                if warmup:
                    continue
                runs.append({
                    "profile": profile_id,
                    "stage": args.stage,
                    "width": width,
                    "height": height,
                    "seed": args.seed,
                    "sample_index": iteration - args.warmups,
                    "weights": weights,
                    "output_sha256": file_sha256(output),
                    "output_bytes": output.stat().st_size,
                    **result,
                })

    matrix = {
        "protocol": MATRIX_SCHEMA,
        "label": args.label,
        "run_id": str(uuid.uuid4()),
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "scope": f"{args.stage}-stage",
        "scope_note": "VAE-only measurements with unquantized FP32 source weights.",
        "registry": str(args.registry),
        "registry_sha256": registry_digest(args.registry),
        "native_binary": {"path": str(native_bin), "sha256": file_sha256(native_bin), "bytes": native_bin.stat().st_size},
        "git": identity,
        "target": target,
        "request": {"seed": args.seed, "sizes": [f"{w}x{h}" for w, h in sizes], "runs_per_profile": args.runs, "warmups_per_size": args.warmups, "order_rotation": "latin-square"},
        "interleaved": True,
        "runs": runs,
        "summaries": summarize(registry, profile_ids, sizes, runs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(matrix, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "runs": len(runs), "summaries": matrix["summaries"]}, indent=2))


if __name__ == "__main__":
    main()

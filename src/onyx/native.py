import argparse
import json
import logging
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from importlib.util import find_spec
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file
from tokenizers import Tokenizer

from onyx.artifacts import sha256_file
from onyx.profiles import ProfileError, dit_cache_bytes, load_registry, registry_digest, resolve_profile, stage_weights


ROOT = Path(__file__).resolve().parents[2]
# Regenerable tables belong in the user cache. Each native build creates distinct entries.
# Text-to-image uses about 55.5 MiB per entry, and editing adds one conditioning timestep.
DEFAULT_MODULATION_CACHE = Path.home() / "Library" / "Caches" / "onyx-infer" / "modulation"
MODULATION_CACHE_KEEP = 2


def resolve_modulation_cache(requested: Path | None, disabled: bool) -> Path | None:
    """Return the modulation cache directory, or None to compute the tables on every request."""
    if disabled:
        return None
    return requested if requested is not None else DEFAULT_MODULATION_CACHE


def prune_modulation_cache(directory: Path, keep: int = MODULATION_CACHE_KEEP) -> list[str]:
    """Delete the oldest modulation entries beyond `keep` and return their names, oldest first.

    Only the default directory is pruned. A directory that the caller chose may hold
    entries that it wants to keep.
    """
    entries = []
    for path in directory.glob("*.adaln"):
        try:
            info = path.stat()
        except FileNotFoundError:
            continue
        if stat.S_ISREG(info.st_mode):
            entries.append((info.st_mtime_ns, path))
    entries.sort(reverse=True)
    removed = []
    for _, stale in reversed(entries[keep:]):
        try:
            stale.unlink()
        except FileNotFoundError:
            continue
        removed.append(stale.name)
    return removed


def dit_memory_measurements(stderr: str) -> dict[str, int]:
    measurements = {}
    for label, key in (("peak memory footprint", "peak_footprint_bytes"),
                       ("maximum resident set size", "peak_rss_bytes")):
        match = re.search(rf"(?m)^[ \t]*(\d+)[ \t]+{label}[ \t]*$", stderr)
        if match is None or int(match[1]) == 0:
            raise RuntimeError(f"macOS time -l did not report a positive DiT {label}")
        measurements[key] = int(match[1])
    return measurements


def main(argv=None):
    parser = argparse.ArgumentParser(prog="onyx native-generate")
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--reference", type=Path, action="append", default=[],
                        help="Ordered H3 reference image. Repeat for up to 16 images.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--native-bin", type=Path, default=ROOT / ".build/onyx-native")
    parser.add_argument("--profile", default="max-quality", help="Execution profile from config/profiles/profiles.v1.json")
    parser.add_argument("--measure-dit", action="store_true",
                        help="Record the DiT latent SHA-256 and lifetime process memory from macOS time -l")
    parser.add_argument("--modulation-cache", type=Path,
                        help="Directory for reusable DiT timestep modulation tables (default: ~/Library/Caches/onyx-infer/modulation)")
    parser.add_argument("--no-modulation-cache", action="store_true",
                        help="Compute the DiT modulation tables on every request and write no cache")
    parser.add_argument("--dit-cache-mib", type=int, help="Override the DiT weight-cache budget in MiB")
    parser.add_argument("--dit-prefetch", action=argparse.BooleanOptionalAction, default=None,
                        help="Override weight prefetch for the DiT stage only")
    parser.add_argument("--dit-shared-budget", action="store_true",
                        help="Share the positive DiT cache budget between static residency and prefetch")
    parser.add_argument("--prefetch-weights", action=argparse.BooleanOptionalAction, default=None,
                        help="Prefetch weights in the text and DiT stages (default: the profile's weight_prefetch knob)")
    args = parser.parse_args(argv)
    if args.measure_dit and sys.platform != "darwin":
        parser.error("--measure-dit requires macOS time -l")
    if len(args.reference) > 16:
        parser.error("H3 accepts up to 16 reference images")
    if args.reference:
        if find_spec("mlx") is None:
            parser.error("Reference images require the onyx-infer[reference] extra")
        for path in args.reference:
            if not path.is_file():
                parser.error(f"Reference image does not exist: {path}")
    try:
        registry = load_registry()
        resolved = resolve_profile(registry, args.profile)
        text_prefetch = resolve_weight_prefetch(resolved["knobs"]["weight_prefetch"], args.prefetch_weights)
    except ProfileError as error:
        parser.error(str(error))
    profile_knobs = resolved["knobs"]
    dit_weight_cache_bytes = dit_cache_bytes(registry, args.profile)
    if args.dit_cache_mib is not None:
        if not 0 <= args.dit_cache_mib < 2**44:
            parser.error("--dit-cache-mib must be a nonnegative budget representable in 64-bit bytes")
        dit_weight_cache_bytes = args.dit_cache_mib * 2**20
    dit_prefetch = resolve_dit_prefetch(text_prefetch, args.prefetch_weights, args.dit_prefetch, dit_weight_cache_bytes)
    prefetch = {"text": text_prefetch, "dit": dit_prefetch}
    if args.dit_shared_budget and (not dit_prefetch or not dit_weight_cache_bytes):
        parser.error("--dit-shared-budget requires --dit-prefetch and a positive DiT cache budget")
    if dit_prefetch and dit_weight_cache_bytes and not args.dit_shared_budget:
        parser.error("DiT weight prefetch requires a zero DiT weight-cache budget")
    modulation_cache = resolve_modulation_cache(args.modulation_cache, args.no_modulation_cache)
    environment = os.environ.copy()
    environment["ONYX_Q4_LINEAR"] = profile_knobs["q4_kernel"]
    binary = args.native_bin.resolve()
    if not binary.is_file():
        raise FileNotFoundError(f"Native executable not found: {binary}")
    models = args.models.resolve()
    tokenizer = Tokenizer.from_file(str(models / "tokenizer/tokenizer.json"))
    ids = tokenizer.encode(args.prompt, add_special_tokens=False).ids[:4096]
    if not ids:
        raise ValueError("Empty prompt after tokenization")
    began = time.monotonic()
    timings = {}
    stage_logs = {}
    dit_measurements = {}
    modulation_cache_status = None
    pruned = []
    with tempfile.TemporaryDirectory(prefix="onyx-native-") as temporary:
        work = Path(temporary)
        reference_arguments = {}
        token_count = len(ids)
        if args.reference:
            (work / "reference-request.json").write_text(json.dumps({
                "models": str(models), "prompt_ids": ids, "seed": args.seed,
                "paths": [str(path.resolve()) for path in args.reference],
            }))
            start = time.monotonic()
            subprocess.run([sys.executable, "-m", "onyx.reference.conditioning", str(work)], check=True)
            timings["references"] = time.monotonic() - start
            token_count = json.loads((work / "tokens.json").read_text())["count"]
            reference_arguments = {
                "text": ["--vision", str(work / "vision.safetensors")],
                "dit": ["--references", str(work / "references.safetensors")],
            }
        else:
            (work / "tokens.json").write_text(json.dumps({"ids": ids, "count": len(ids)}))
        noise = np.random.default_rng(args.seed).standard_normal((24, args.height // 16, args.width // 16), dtype=np.float32)
        save_file({"latent": noise}, str(work / "noise.safetensors"))
        latent = work / "latent.safetensors"
        stages = [
            ("text", [
                "--model", str(models / "text_encoder.onxpack"),
                "--tokens", str(work / "tokens.json"), "--output", str(work / "embedding.safetensors"),
            ]),
            ("dit", [
                "--model", str(models / "transformer.onxpack"), "--embedding", str(work / "embedding.safetensors"),
                "--noise", str(work / "noise.safetensors"), "--pdd", str(models / "pdd_fused_heads.safetensors"),
                "--output", str(latent), "--width", str(args.width), "--height", str(args.height),
            ]),
            ("vae", [
                "--model", str(models / "vae/diffusion_pytorch_model.safetensors"), "--latent", str(latent),
                "--config", str(models / "vae/config.json"), "--output", str(work / "image.png"),
            ]),
        ]
        for name, arguments in stages:
            declared_weights = stage_weights(args.profile, profile_knobs, name)
            command = [str(binary), name, *arguments, *reference_arguments.get(name, []),
                       "--profile", args.profile, "--weights", declared_weights]
            declared_cache = dit_weight_cache_bytes if name == "dit" else 0
            if name == "dit":
                command.extend(["--cache-bytes", str(declared_cache)])
                if modulation_cache is not None:
                    command.extend(["--modulation-cache", str(modulation_cache.resolve())])
                if args.dit_shared_budget:
                    command.append("--shared-weight-budget")
            if name in prefetch:
                command.append("--prefetch-weights" if prefetch[name] else "--no-prefetch-weights")
            if name == "dit" and args.measure_dit:
                command = ["/usr/bin/time", "-l", *command]
            start = time.monotonic()
            result = subprocess.run(command, capture_output=True, text=True, env=environment)
            timings[name] = time.monotonic() - start
            stage_logs[name] = {"stdout": result.stdout, "stderr": result.stderr}
            if result.returncode:
                raise RuntimeError(f"Native {name} failed: {result.stderr.strip()}")
            verify_stage_echo(name, result.stdout, args.profile, declared_weights, declared_cache, prefetch.get(name, False),
                              name == "dit" and modulation_cache is not None)
            details = stage_details(name, result.stdout)
            if args.reference and name == "text" and details.get("vision_rows", 0) <= 0:
                raise RuntimeError("The native text stage did not consume vision features. Rebuild onyx-native.")
            if name == "dit":
                if details.get("reference_images", 0) != len(args.reference):
                    raise RuntimeError("The native DiT stage did not consume the requested references. Rebuild onyx-native.")
                modulation_cache_status = details.get("modulation_cache")
                if details["weight_cache"].get("shared", False) != args.dit_shared_budget:
                    raise RuntimeError("The native DiT stage did not use the requested weight budget policy")
                if args.measure_dit:
                    dit_measurements = dit_memory_measurements(result.stderr)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        partial = args.output.with_suffix(args.output.suffix + ".part")
        shutil.copyfile(work / "image.png", partial)
        os.replace(partial, args.output)
        if args.measure_dit:
            dit_measurements["latent_sha256"] = sha256_file(latent)
    if modulation_cache_status == "miss" and modulation_cache is not None and args.modulation_cache is None:
        try:
            pruned = prune_modulation_cache(modulation_cache)
        except OSError as error:
            logging.getLogger(__name__).warning("Could not prune modulation cache %s: %s", modulation_cache, error)
    print(json.dumps({
        "backend": "onyx-native-cpp-metal",
        "profile": args.profile,
        "profile_knobs": profile_knobs,
        "profile_registry_sha256": registry_digest(),
        "dit_weight_cache_bytes": dit_weight_cache_bytes,
        "weight_prefetch": prefetch,
        "modulation_cache": {"directory": str(modulation_cache) if modulation_cache is not None else None,
                             "status": modulation_cache_status, "pruned": pruned},
        "token_count": token_count,
        "reference_count": len(args.reference),
        "stage_seconds": timings,
        "stage_logs": stage_logs,
        "total_seconds": time.monotonic() - began,
        **({"dit_measurements": dit_measurements} if args.measure_dit else {}),
    }))


def resolve_dit_prefetch(knob_prefetch: bool, requested: bool | None, dit_requested: bool | None, cache_bytes: int) -> bool:
    """Return whether the DiT stage reads the next block's weights ahead of the GPU.

    A nonzero weight-cache budget replaces prefetch in the DiT stage unless the request asks
    for prefetch explicitly, which the caller rejects.
    """
    if dit_requested is not None:
        return dit_requested
    if requested is not None:
        return requested
    return knob_prefetch and cache_bytes == 0


def stage_details(stage: str, stdout: str) -> dict:
    """Return the JSON line that a native stage printed about itself."""
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            details = json.loads(line)
        except json.JSONDecodeError:
            continue
        if details.get("stage") == stage:
            return details
    return {}


def resolve_weight_prefetch(knob: str, requested: bool | None) -> bool:
    """Return whether streaming stages read the next block's weights ahead of the GPU."""
    if requested is not None:
        return requested
    if knob not in ("off", "next"):
        raise ProfileError(f"Unknown weight_prefetch setting {knob!r}")
    return knob == "next"


def verify_stage_echo(stage: str, stdout: str, profile: str, weights: str, cache_bytes: int = 0,
                      prefetch: bool = False, modulation_cache: bool = False) -> None:
    """Verify that the native stage reports the requested profile and settings."""
    details = stage_details(stage, stdout)
    if not details:
        raise RuntimeError(f"Native {stage} did not echo a stage JSON line; cannot verify profile {profile!r}")
    if details.get("profile") != profile:
        raise RuntimeError(f"Native {stage} echoed profile {details.get('profile')!r}, requested {profile!r}")
    if details.get("weights") not in (None, weights):
        raise RuntimeError(f"Native {stage} echoed weights {details.get('weights')!r}, requested {weights!r}")
    if stage in ("text", "dit") and details.get("prefetch_weights", False) != prefetch:
        raise RuntimeError(f"Native {stage} echoed prefetch {details.get('prefetch_weights')!r}, requested {prefetch}. "
                           "A native build without weight prefetch echoes nothing for it: rebuild the engine "
                           "or pass --no-prefetch-weights.")
    if stage == "dit":
        cache_echo = details.get("weight_cache") or {}
        if cache_echo.get("budget") != cache_bytes:
            raise RuntimeError(f"Native dit echoed cache budget {cache_echo.get('budget')!r}, requested {cache_bytes}")
        status = details.get("modulation_cache")
        if modulation_cache != (status != "disabled"):
            raise RuntimeError(f"Native dit echoed modulation cache {status!r}, requested {modulation_cache}")


if __name__ == "__main__":
    main()

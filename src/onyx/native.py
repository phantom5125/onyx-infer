import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file
from tokenizers import Tokenizer

from onyx.profiles import ProfileError, dit_cache_bytes, load_registry, registry_digest, resolve_profile, stage_weights


ROOT = Path(__file__).resolve().parents[2]
# Regenerable tables belong in the user cache. An entry binds one native build, so every
# rebuild of the engine leaves the previous entry behind at about 55.5 MiB.
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
    entries = sorted((path for path in directory.glob("*.adaln") if path.is_file()),
                     key=lambda path: path.stat().st_mtime, reverse=True)
    removed = []
    for stale in reversed(entries[keep:]):
        stale.unlink()
        removed.append(stale.name)
    return removed


def main(argv=None):
    parser = argparse.ArgumentParser(prog="onyx native-generate")
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--native-bin", type=Path, default=ROOT / ".build/onyx-native")
    parser.add_argument("--profile", default="max-quality", help="Execution profile from config/profiles/profiles.v1.json")
    parser.add_argument("--modulation-cache", type=Path,
                        help="Directory for reusable DiT timestep modulation tables (default: ~/Library/Caches/onyx-infer/modulation)")
    parser.add_argument("--no-modulation-cache", action="store_true",
                        help="Compute the DiT modulation tables on every request and write no cache")
    parser.add_argument("--dit-cache-mib", type=int, help="Override the DiT weight-cache budget in MiB")
    parser.add_argument("--dit-prefetch", action="store_true", help="Read the next DiT block's weights during GPU computation, whatever the profile requests")
    parser.add_argument("--prefetch-weights", action=argparse.BooleanOptionalAction, default=None,
                        help="Read the next block's weights during GPU computation (default: the profile's weight_prefetch knob)")
    args = parser.parse_args(argv)
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
    if dit_prefetch and dit_weight_cache_bytes:
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
    modulation_cache_status = None
    pruned = []
    with tempfile.TemporaryDirectory(prefix="onyx-native-") as temporary:
        work = Path(temporary)
        (work / "tokens.json").write_text(json.dumps({"ids": ids, "count": len(ids)}))
        noise = np.random.default_rng(args.seed).standard_normal((24, args.height // 16, args.width // 16), dtype=np.float32)
        save_file({"latent": noise}, str(work / "noise.safetensors"))
        stages = [
            ("text", [
                "--model", str(models / "text_encoder.onxpack"),
                "--tokens", str(work / "tokens.json"), "--output", str(work / "embedding.safetensors"),
            ]),
            ("dit", [
                "--model", str(models / "transformer.onxpack"), "--embedding", str(work / "embedding.safetensors"),
                "--noise", str(work / "noise.safetensors"), "--pdd", str(models / "pdd_fused_heads.safetensors"),
                "--output", str(work / "latent.safetensors"), "--width", str(args.width), "--height", str(args.height),
            ]),
            ("vae", [
                "--model", str(models / "vae/diffusion_pytorch_model.safetensors"), "--latent", str(work / "latent.safetensors"),
                "--config", str(models / "vae/config.json"), "--output", str(work / "image.png"),
            ]),
        ]
        for name, arguments in stages:
            declared_weights = stage_weights(args.profile, profile_knobs, name)
            command = [str(binary), name, *arguments, "--profile", args.profile, "--weights", declared_weights]
            declared_cache = dit_weight_cache_bytes if name == "dit" else 0
            if name == "dit":
                command.extend(["--cache-bytes", str(declared_cache)])
                if modulation_cache is not None:
                    command.extend(["--modulation-cache", str(modulation_cache.resolve())])
            if name in prefetch:
                command.append("--prefetch-weights" if prefetch[name] else "--no-prefetch-weights")
            start = time.monotonic()
            result = subprocess.run(command, capture_output=True, text=True, env=environment)
            timings[name] = time.monotonic() - start
            stage_logs[name] = {"stdout": result.stdout, "stderr": result.stderr}
            if result.returncode:
                raise RuntimeError(f"Native {name} failed: {result.stderr.strip()}")
            verify_stage_echo(name, result.stdout, args.profile, declared_weights, declared_cache, prefetch.get(name, False),
                              name == "dit" and modulation_cache is not None)
            if name == "dit":
                modulation_cache_status = stage_details("dit", result.stdout).get("modulation_cache")
        if modulation_cache_status == "miss" and modulation_cache is not None and \
                modulation_cache.resolve() == DEFAULT_MODULATION_CACHE.resolve():
            # The default directory holds only the newest entries of the current build.
            pruned = prune_modulation_cache(modulation_cache)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        partial = args.output.with_suffix(args.output.suffix + ".part")
        shutil.copyfile(work / "image.png", partial)
        os.replace(partial, args.output)
    print(json.dumps({
        "backend": "onyx-native-cpp-metal",
        "profile": args.profile,
        "profile_knobs": profile_knobs,
        "profile_registry_sha256": registry_digest(),
        "dit_weight_cache_bytes": dit_weight_cache_bytes,
        "weight_prefetch": prefetch,
        "modulation_cache": {"directory": str(modulation_cache) if modulation_cache is not None else None,
                             "status": modulation_cache_status, "pruned": pruned},
        "token_count": len(ids),
        "stage_seconds": timings,
        "stage_logs": stage_logs,
        "total_seconds": time.monotonic() - began,
    }))


def resolve_dit_prefetch(knob_prefetch: bool, requested: bool | None, dit_requested: bool, cache_bytes: int) -> bool:
    """Return whether the DiT stage reads the next block's weights ahead of the GPU.

    A nonzero weight-cache budget replaces prefetch in the DiT stage unless the request asks
    for prefetch explicitly, which the caller rejects.
    """
    if dit_requested or requested is True:
        return True
    if requested is False:
        return False
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

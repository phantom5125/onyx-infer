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
    parser.add_argument("--modulation-cache", type=Path, help="Directory for reusable DiT timestep modulation tables")
    parser.add_argument("--dit-cache-mib", type=int, help="Override the DiT weight-cache budget in MiB")
    parser.add_argument("--dit-prefetch", action=argparse.BooleanOptionalAction, default=None,
                        help="Read the next DiT block's projection weights during GPU computation "
                             "(default: enabled with a zero DiT weight-cache budget)")
    args = parser.parse_args(argv)
    try:
        registry = load_registry()
        resolved = resolve_profile(registry, args.profile)
    except ProfileError as error:
        parser.error(str(error))
    profile_knobs = resolved["knobs"]
    dit_weight_cache_bytes = dit_cache_bytes(registry, args.profile)
    if args.dit_cache_mib is not None:
        if not 0 <= args.dit_cache_mib < 2**44:
            parser.error("--dit-cache-mib must be a nonnegative budget representable in 64-bit bytes")
        dit_weight_cache_bytes = args.dit_cache_mib * 2**20
    if args.dit_prefetch and dit_weight_cache_bytes:
        parser.error("--dit-prefetch requires a zero DiT weight-cache budget")
    dit_prefetch = resolve_dit_prefetch(args.dit_prefetch, dit_weight_cache_bytes)
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
                if args.modulation_cache is not None:
                    command.extend(["--modulation-cache", str(args.modulation_cache.resolve())])
                command.append("--prefetch-weights" if dit_prefetch else "--no-prefetch-weights")
            start = time.monotonic()
            result = subprocess.run(command, capture_output=True, text=True, env=environment)
            timings[name] = time.monotonic() - start
            stage_logs[name] = {"stdout": result.stdout, "stderr": result.stderr}
            if result.returncode:
                raise RuntimeError(f"Native {name} failed: {result.stderr.strip()}")
            verify_stage_echo(name, result.stdout, args.profile, declared_weights, declared_cache, dit_prefetch)
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
        "dit_prefetch": dit_prefetch,
        "token_count": len(ids),
        "stage_seconds": timings,
        "stage_logs": stage_logs,
        "total_seconds": time.monotonic() - began,
    }))


def resolve_dit_prefetch(requested: bool | None, cache_bytes: int) -> bool:
    return cache_bytes == 0 if requested is None else requested


def verify_stage_echo(stage: str, stdout: str, profile: str, weights: str, cache_bytes: int = 0,
                      dit_prefetch: bool = False) -> None:
    """Verify that the native stage reports the requested profile and settings."""
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            details = json.loads(line)
        except json.JSONDecodeError:
            continue
        if details.get("stage") != stage:
            continue
        if details.get("profile") != profile:
            raise RuntimeError(f"Native {stage} echoed profile {details.get('profile')!r}, requested {profile!r}")
        if details.get("weights") not in (None, weights):
            raise RuntimeError(f"Native {stage} echoed weights {details.get('weights')!r}, requested {weights!r}")
        if stage == "dit":
            cache_echo = details.get("weight_cache") or {}
            if cache_echo.get("budget") != cache_bytes:
                raise RuntimeError(f"Native dit echoed cache budget {cache_echo.get('budget')!r}, requested {cache_bytes}")
            if details.get("prefetch_weights", False) != dit_prefetch:
                raise RuntimeError(f"Native dit echoed prefetch {details.get('prefetch_weights')!r}, requested {dit_prefetch}")
        return
    raise RuntimeError(f"Native {stage} did not echo a stage JSON line; cannot verify profile {profile!r}")


if __name__ == "__main__":
    main()

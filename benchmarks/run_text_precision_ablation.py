"""Isolate text-encoder and DiT precision on the fixed dense-label case."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
from safetensors.numpy import load_file, save_file
from tokenizers import Tokenizer

from onyx.label_prompt import build_label_prompt

def digest(path):
    with Path(path).open("rb") as artifact:
        return hashlib.file_digest(artifact, "sha256").hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-bin", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True, help="Directory containing the original sharded H3 model")
    parser.add_argument("--w4", type=Path, required=True, help="Directory containing the W4 H3 model")
    parser.add_argument("--output", type=Path, required=True, help="New directory for the stage outputs and images")
    args = parser.parse_args()
    native = args.native_bin.resolve(strict=True)
    original = args.original.resolve(strict=True)
    w4 = args.w4.resolve(strict=True)
    result_dir = args.output.resolve()
    result_dir.mkdir(parents=True, exist_ok=False)
    layout = json.loads((root / "benchmarks/requests/text-baseline-jet-engine-structured.json").read_text(encoding="utf-8"))
    prompt = build_label_prompt(None, layout)
    prior_runs = json.loads((root / "benchmarks/results/h3-structured-label-prompt-v1.json").read_text())["runs"]
    w4_controls = {row["seed"]: row["png_sha256"] for row in prior_runs if row["arm"] == "structured-square"}
    tokenizer = Tokenizer.from_file(str(w4 / "tokenizer/tokenizer.json"))
    tokens = tokenizer.encode(prompt, add_special_tokens=False).ids
    token_path = result_dir / "tokens.json"
    token_path.write_text(json.dumps({"ids": tokens}))
    mirror = result_dir / "raw-dit"
    mirror.mkdir()
    for source in (*sorted((original / "transformer").glob("model*")), original / "pdd_config.json"):
        (mirror / source.name).symlink_to(source)
    raw_text = original / "text_encoder/model.safetensors.index.json"
    raw_dit = mirror / "model.safetensors.index.json"
    env = {key: value for key, value in os.environ.items() if not key.startswith("ONYX_")}
    env["ONYX_Q4_LINEAR"] = "mpsgraph"


    def run(stage, files, log, *, weights="stream"):
        proc = subprocess.run([str(native), stage, *map(str, files), "--profile", "max-quality", "--weights", weights],
                              capture_output=True, text=True, env=env)
        log.write_text(proc.stdout + proc.stderr)
        if proc.returncode:
            raise RuntimeError(proc.stderr)
        return json.loads(proc.stdout.splitlines()[0])

    for seed in (42, 741):
        work = result_dir / str(seed)
        work.mkdir()
        noise = work / "noise.safetensors"
        save_file({"latent": np.random.default_rng(seed).standard_normal((24, 64, 64), dtype=np.float32)}, str(noise))
        for dtype, model in (("w4", w4 / "text_encoder.onxpack"), ("bf16", raw_text)):
            embedding = work / f"{dtype}-embedding.safetensors"
            text = run("text", ["--model", model, "--tokens", token_path, "--output", embedding], work / f"text-{dtype}.log")
            text["sha256"] = digest(embedding)
            (work / f"text-{dtype}.json").write_text(json.dumps(text, indent=2) + "\n")
            print(seed, "text", dtype, text["seconds"], flush=True)
        for text_dtype, dit_dtype in (("w4", "w4"), ("bf16", "w4"), ("w4", "bf16")):
            arm = f"text-{text_dtype}-dit-{dit_dtype}"
            record_file = work / f"{arm}.json"
            embedding = work / f"{text_dtype}-embedding.safetensors"
            model = w4 / "transformer.onxpack" if dit_dtype == "w4" else raw_dit
            latent = work / f"{arm}.safetensors"
            image = result_dir / f"{seed}-{arm}.png"
            dit = run("dit", ["--model", model, "--embedding", embedding, "--noise", noise,
                              "--pdd", w4 / "pdd_fused_heads.safetensors", "--width", 1024, "--height", 1024,
                              "--output", latent], work / f"{arm}-dit.log")
            values = load_file(latent)["latent"]
            assert values.shape == (24, 64, 64) and np.isfinite(values).all()
            vae = run("vae", ["--model", w4 / "vae/diffusion_pytorch_model.safetensors", "--latent", latent,
                              "--config", w4 / "vae/config.json", "--output", image], work / f"{arm}-vae.log", weights="resident")
            with Image.open(image) as pixels:
                assert pixels.size == (1024, 1024)
            result = {
                "arm": arm, "seed": seed, "embedding_sha256": digest(embedding), "noise_sha256": digest(noise),
                "latent_sha256": hashlib.sha256(values.tobytes()).hexdigest(), "png_sha256": digest(image),
                "dit_seconds": dit["seconds"], "vae_seconds": vae["seconds"],
            }
            if text_dtype == dit_dtype == "w4":
                result["matches_reference_w4_png"] = result["png_sha256"] == w4_controls[seed]
            record_file.write_text(json.dumps(result, indent=2) + "\n")
            print(seed, arm, result["dit_seconds"], result["png_sha256"][:12], flush=True)


if __name__ == "__main__":
    main()

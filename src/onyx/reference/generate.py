import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def stage(name, work):
    import mlx.core as mx

    mx.set_cache_limit(96 << 20)
    request = json.loads((work / "request.json").read_text())
    models = Path(request["models"])
    if name == "text":
        from onyx.reference.text import encode

        original = request.get("original_w4", False)
        text_weights = models / "text_encoder.onxpack" if original else models / "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
        tokenizer = models / "tokenizer/tokenizer.json" if original else models / "tokenizer.json"
        embedding, ids = encode(text_weights, tokenizer, request["prompt"], original=original)
        mx.save_safetensors(str(work / "embedding.safetensors"), {"embedding": embedding})
        (work / "text.json").write_text(json.dumps({"token_count": len(ids)}))
    elif name == "dit":
        from onyx.reference.dit import denoise

        embedding = mx.load(str(work / "embedding.safetensors"))["embedding"]
        original = request.get("original_w4", False)
        dit_weights = models / "transformer.onxpack" if original else models / "diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors"
        sigmas = json.loads((models / "pdd_config.json").read_text())["sigmas"] if original else None
        latent = denoise(
            dit_weights, embedding, request["width"], request["height"], request["seed"], request["dit_mode"] == "resident",
            original=original, pdd_path=models / "pdd_fused_heads.safetensors" if original else None, sigmas=sigmas,
        )
        mx.save_safetensors(str(work / "latent.safetensors"), {"latent": latent})
    elif name == "vae":
        import mlx.core as mx
        from onyx.reference.vae import decode

        latent = mx.load(str(work / "latent.safetensors"))["latent"]
        original = request.get("original_w4", False)
        vae_weights = models / "vae/diffusion_pytorch_model.safetensors" if original else models / "vae/minimax_h3_image_vae_fp16.safetensors"
        decode(vae_weights, latent, models / "vae/config.json", work / "image.png")
    else:
        raise ValueError(name)
    mx.synchronize()


def generate(args):
    request = {
        "models": str(Path(args.models).resolve()), "prompt": args.prompt,
        "width": args.width, "height": args.height, "seed": args.seed,
        "dit_mode": args.dit_mode, "original_w4": args.original_w4,
    }
    began = time.monotonic()
    timings = {}
    with tempfile.TemporaryDirectory(prefix="onyx-reference-") as temporary:
        work = Path(temporary)
        (work / "request.json").write_text(json.dumps(request))
        for name in ("text", "dit", "vae"):
            start = time.monotonic()
            result = subprocess.run([sys.executable, "-m", "onyx.reference.generate", "--stage", name, "--work", str(work)], capture_output=True, text=True)
            timings[name] = time.monotonic() - start
            if result.returncode:
                raise RuntimeError(f"Reference {name} failed: {result.stderr.strip()}")
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_suffix(output.suffix + ".part")
        shutil.copyfile(work / "image.png", partial)
        os.replace(partial, output)
        token_count = json.loads((work / "text.json").read_text())["token_count"]
    backend = "onyx-reference-mlx-w4" if args.original_w4 else "onyx-reference-mlx-w8a16"
    metrics = {"backend": backend, "token_count": token_count, "stage_seconds": timings, "total_seconds": time.monotonic() - began}
    print(json.dumps(metrics))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="onyx-reference")
    parser.add_argument("--stage", choices=("text", "dit", "vae"))
    parser.add_argument("--work", type=Path)
    parser.add_argument("--models")
    parser.add_argument("--prompt")
    parser.add_argument("--output")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dit-mode", choices=("resident", "stream"), default="resident")
    parser.add_argument("--original-w4", action="store_true")
    args = parser.parse_args(argv)
    if args.stage:
        if not args.work:
            parser.error("--work is required for a stage")
        stage(args.stage, args.work)
    else:
        if not args.models or not args.prompt or not args.output:
            parser.error("--models, --prompt and --output are required")
        generate(args)


if __name__ == "__main__":
    main()

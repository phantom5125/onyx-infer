import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
PREFIXES = ("transformer/", "vae/", "tokenizer/", "processor/", "text_encoder/")
EXACT = {"model_index.json", "pdd_config.json", "pdd_fused_heads.safetensors"}


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="MiniMax/MiniMax-H3-Image")
    parser.add_argument("--revision", default="df334556f585a596a710cf4f7ce3a567c0738ee5")
    parser.add_argument("--output", type=Path, default=ROOT / "config/sources/h3-image-files.json")
    args = parser.parse_args()
    query = urlencode({"Revision": args.revision, "Recursive": "true"})
    url = f"https://modelscope.cn/api/v1/models/{quote(args.repo)}/repo/files?{query}"
    with urlopen(url, timeout=60) as response:
        payload = json.load(response)
    if payload.get("Code") != 200:
        raise RuntimeError(payload)
    files = [
        {"path": item["Path"], "bytes": item["Size"], "sha256": item["Sha256"]}
        for item in payload["Data"]["Files"]
        if item["Type"] == "blob" and (item["Path"] in EXACT or item["Path"].startswith(PREFIXES))
    ]
    files.sort(key=lambda item: item["path"])
    if len(files) != 53:
        raise RuntimeError(f"Expected 53 selected files, found {len(files)}")
    for item in files:
        if len(item["sha256"]) != 64:
            raise RuntimeError(f"Missing SHA-256: {item['path']}")
    lock = {
        "transport": "modelscope",
        "repo_id": args.repo,
        "revision": args.revision,
        "files": files,
        "total_bytes": sum(item["bytes"] for item in files),
        "tree_sha256": hashlib.sha256(canonical_json(files)).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n")
    print(f"Locked {len(files)} files, {lock['total_bytes']} bytes, tree {lock['tree_sha256']}")


if __name__ == "__main__":
    main()

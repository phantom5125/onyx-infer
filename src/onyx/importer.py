import argparse
import hashlib
import json
from pathlib import Path

from onyx.artifacts import canonical_json, sha256_file, write_proto_json
from onyx.h3 import build_h3_manifest
from onyx.model.v1 import model_pb2
from onyx.safetensors import SafeTensorFile


def load_source_lock(path: Path):
    lock = json.loads(path.read_text())
    if lock.get("transport") != "modelscope":
        raise ValueError("Only ModelScope source locks are accepted")
    files = lock["files"]
    if len(files) != len({item["path"] for item in files}):
        raise ValueError("Duplicate source paths")
    if hashlib.sha256(canonical_json(files)).hexdigest() != lock["tree_sha256"]:
        raise ValueError(f"Source tree digest mismatch: {path}")
    parsed = model_pb2.SourceLock(
        transport=lock["transport"], repo_id=lock["repo_id"], revision=lock["revision"],
        files=[model_pb2.SourceFile(path=item["path"], byte_size=item["bytes"], sha256=bytes.fromhex(item["sha256"])) for item in files],
        tree_digest=bytes.fromhex(lock["tree_sha256"]),
    )
    return lock, parsed


def verify_files(source: Path, lock, verify_hashes=False, metadata_only=False):
    report = {"files_checked": 0, "bytes_checked": 0, "hashes_checked": 0, "tensors_checked": 0, "metadata_only": metadata_only}
    if metadata_only:
        return report
    for item in lock["files"]:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe source path: {relative}")
        path = source / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing locked source file: {path}")
        size = path.stat().st_size
        if size != item["bytes"]:
            raise ValueError(f"Size mismatch for {path}: {size} != {item['bytes']}")
        report["files_checked"] += 1
        report["bytes_checked"] += size
        if verify_hashes:
            if sha256_file(path) != item["sha256"]:
                raise ValueError(f"SHA-256 mismatch: {path}")
            report["hashes_checked"] += 1
        if path.suffix == ".safetensors":
            parsed = SafeTensorFile(path)
            report["tensors_checked"] += len(parsed.tensors)
    return report


def import_h3(source: Path, lock_path: Path, output: Path, config=None, verify_hashes=False, metadata_only=False):
    lock, _ = load_source_lock(lock_path)
    report = verify_files(source, lock, verify_hashes, metadata_only)
    manifest = build_h3_manifest(config or source, lock)
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.pb").write_bytes(manifest.SerializeToString(deterministic=True))
    write_proto_json(manifest, output / "manifest.json")
    report.update({
        "model_id": manifest.model_id,
        "source_revision": manifest.source_revision,
        "semantic_digest": manifest.semantic_digest.hex(),
        "source_tree_sha256": lock["tree_sha256"],
    })
    (output / "import-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return manifest, report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="onyx import")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args(argv)
    _, report = import_h3(args.source, args.lock, args.output, args.config, args.verify_hashes, args.metadata_only)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen


CHUNK_SIZE = 256 * 1024 * 1024


def plan_ranges(total, existing, chunk_size=CHUNK_SIZE):
    if not 0 <= existing <= total:
        raise ValueError("Invalid existing size")
    ranges = []
    start = existing
    while start < total:
        if start % chunk_size:
            end = min(total, ((start // chunk_size) + 1) * chunk_size)
        else:
            end = min(total, start + chunk_size)
        ranges.append((start, end))
        start = end
    return ranges


def download_range(url, start, end, path: Path, retries=6):
    expected = end - start
    if path.exists() and path.stat().st_size == expected:
        return path, 0
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".tmp")
    for attempt in range(retries):
        try:
            request = Request(url, headers={"Range": f"bytes={start}-{end - 1}"})
            with urlopen(request, timeout=120) as response:
                full_response = response.status == 200 and start == 0 and response.headers.get("Content-Length") == str(expected)
                if response.status != 206 and not full_response:
                    raise RuntimeError(f"Range request returned {response.status} for {start}-{end}")
                content_range = response.headers.get("Content-Range", "")
                if response.status == 206 and not content_range.startswith(f"bytes {start}-{end - 1}/"):
                    raise RuntimeError(f"Invalid Content-Range: {content_range}")
                with partial.open("wb") as output:
                    while True:
                        data = response.read(1024 * 1024)
                        if not data:
                            break
                        output.write(data)
            if partial.stat().st_size != expected:
                raise OSError(f"Range size mismatch: {partial.stat().st_size} != {expected}")
            os.replace(partial, path)
            return path, expected
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def fetch_file(entry, directory: Path, workers: int):
    destination = directory / entry["path"]
    if destination.exists():
        if destination.stat().st_size != entry["bytes"]:
            raise ValueError(f"Existing file size mismatch: {destination}")
        with destination.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"Existing file hash mismatch: {destination}")
        print(f"Verified existing {entry['path']}", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    legacy = destination.with_name(destination.name + ".incomplete")
    if not partial.exists() and legacy.exists() and legacy.stat().st_size <= entry["bytes"]:
        os.replace(legacy, partial)
    existing = partial.stat().st_size if partial.exists() else 0
    if existing > entry["bytes"]:
        raise ValueError(f"Oversized partial: {partial}")
    chunks = destination.with_name(destination.name + ".chunks")
    ranges = plan_ranges(entry["bytes"], existing)
    remaining = sum(end - start for start, end in ranges)
    if shutil.disk_usage(destination.parent).free < remaining + entry["bytes"]:
        raise OSError(f"Insufficient disk space for parallel assembly of {destination}")
    print(f"{entry['path']}: existing {existing / 1e9:.2f} GB, remaining {remaining / 1e9:.2f} GB in {len(ranges)} ranges", flush=True)
    resumed = 0
    for index, (start, end) in enumerate(ranges):
        path = chunks / f"{index:05d}-{start}-{end}"
        if path.exists() and path.stat().st_size == end - start:
            resumed += end - start
    began = time.monotonic()
    downloaded = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = []
        for index, (start, end) in enumerate(ranges):
            path = chunks / f"{index:05d}-{start}-{end}"
            futures.append(pool.submit(download_range, entry["modelscope"], start, end, path))
        for future in as_completed(futures):
            _, size = future.result()
            downloaded += size
            elapsed = time.monotonic() - began
            completed = resumed + downloaded
            print(f"  completed {completed / 1e9:.2f}/{remaining / 1e9:.2f} GB, new-data rate {downloaded / max(elapsed, 1e-6) / 1e6:.1f} MB/s", flush=True)
    assembled = destination.with_name(destination.name + ".assembled")
    digest = hashlib.sha256()
    with assembled.open("wb") as output:
        if existing:
            with partial.open("rb") as stream:
                while data := stream.read(8 * 1024 * 1024):
                    output.write(data)
                    digest.update(data)
        for index, (start, end) in enumerate(ranges):
            path = chunks / f"{index:05d}-{start}-{end}"
            if path.stat().st_size != end - start:
                raise OSError(f"Chunk size changed: {path}")
            with path.open("rb") as stream:
                while data := stream.read(8 * 1024 * 1024):
                    output.write(data)
                    digest.update(data)
    if assembled.stat().st_size != entry["bytes"]:
        raise OSError(f" assembled size mismatch: {assembled}")
    if digest.hexdigest() != entry["sha256"]:
        raise ValueError(f"SHA-256 mismatch after assembly: {destination}")
    os.replace(assembled, destination)
    partial.unlink(missing_ok=True)
    legacy.unlink(missing_ok=True)
    shutil.rmtree(chunks)
    print(f"Verified {destination} in {time.monotonic() - began:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--source-lock", type=Path)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--only")
    args = parser.parse_args()
    if args.manifest:
        manifest = json.loads(args.manifest.read_text())
    else:
        lock = json.loads(args.source_lock.read_text())
        manifest = {"files": [{
            "path": item["path"],
            "bytes": item["bytes"],
            "sha256": item["sha256"],
            "modelscope": f"https://modelscope.cn/models/{quote(lock['repo_id'])}/resolve/{lock['revision']}/{quote(item['path'])}",
        } for item in lock["files"]]}
    args.directory.mkdir(parents=True, exist_ok=True)
    with (args.directory / ".onyx-fetch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for entry in manifest["files"]:
            if args.only and entry["path"] != args.only:
                continue
            fetch_file(entry, args.directory, args.workers)


if __name__ == "__main__":
    main()

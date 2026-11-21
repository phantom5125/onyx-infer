import hashlib
import json
from pathlib import Path

import rfc8785
from google.protobuf import json_format

from onyx.common.v1 import types_pb2


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_json(value) -> bytes:
    encoded = rfc8785.dumps(value)
    return encoded if isinstance(encoded, bytes) else encoded.encode("utf-8")


def message_digest(message) -> bytes:
    value = json_format.MessageToDict(message, preserving_proto_field_name=True)
    return hashlib.sha256(canonical_json(value)).digest()


def artifact_from_path(path: Path, kind=types_pb2.ARTIFACT_KIND_OTHER, media_type="application/octet-stream", locator=None):
    digest = sha256_file(path)
    return types_pb2.ArtifactRef(
        artifact_id=f"sha256:{digest}",
        media_type=media_type,
        byte_size=path.stat().st_size,
        sha256=bytes.fromhex(digest),
        local_locator=str(locator if locator is not None else path),
        kind=kind,
    )


def artifact_from_lock(item, locator=None):
    return types_pb2.ArtifactRef(
        artifact_id=f"sha256:{item['sha256']}",
        media_type="application/x-safetensors" if item["path"].endswith(".safetensors") else "application/json",
        byte_size=item["bytes"],
        sha256=bytes.fromhex(item["sha256"]),
        local_locator=str(locator if locator is not None else item["path"]),
        kind=types_pb2.ARTIFACT_KIND_WEIGHTS if item["path"].endswith(".safetensors") else types_pb2.ARTIFACT_KIND_OTHER,
    )


def write_proto_json(message, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = json_format.MessageToDict(message, preserving_proto_field_name=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")

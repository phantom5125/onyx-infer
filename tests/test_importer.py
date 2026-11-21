import json
from pathlib import Path

import pytest

from onyx.importer import import_h3, load_source_lock
from onyx.model.v1 import model_pb2


ROOT = Path(__file__).resolve().parents[1]


def test_metadata_import_is_deterministic(tmp_path, h3_test_config):
    source = h3_test_config
    lock = ROOT / "config/sources/h3-image-files.json"
    first, report = import_h3(source, lock, tmp_path / "first", metadata_only=True)
    second, _ = import_h3(source, lock, tmp_path / "second", metadata_only=True)
    assert first == second
    assert first.protocol == "onyx.model.v1"
    assert first.source_revision == "df334556f585a596a710cf4f7ce3a567c0738ee5"
    assert first.recipes[0].sigmas == [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.0]
    assert len(first.graphs[1].nodes) == 53
    assert len(first.components) == 5
    assert report["metadata_only"] is True
    decoded = model_pb2.ModelManifest.FromString((tmp_path / "first/manifest.pb").read_bytes())
    assert decoded.semantic_digest.hex() == report["semantic_digest"]
    assert len(decoded.semantic_digest) == 32


def test_source_lock_digest_detects_tampering(tmp_path):
    lock_path = ROOT / "config/sources/h3-image-files.json"
    lock, parsed = load_source_lock(lock_path)
    assert len(parsed.files) == 53
    assert len(parsed.tree_digest) == 32
    lock["files"][0]["bytes"] += 1
    tampered = tmp_path / "lock.json"
    tampered.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="tree digest"):
        load_source_lock(tampered)

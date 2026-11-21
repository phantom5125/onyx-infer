import json

import pytest


@pytest.fixture
def h3_test_config(tmp_path):
    """Minimal synthetic configuration for the importer tests."""
    root = tmp_path / "h3-test-config"
    files = {
        "text_encoder/config.json": {"hidden_size": 16},
        "transformer/config.json": {"hidden_size": 32},
        "vae/config.json": {"latent_channels": 24},
        "pdd_config.json": {"sigmas": [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.0]},
    }
    for filename, data in files.items():
        path = root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
    return root

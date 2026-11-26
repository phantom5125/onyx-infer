import json
from types import SimpleNamespace

import numpy as np
import pytest

from onyx.native import main


def test_native_rejects_more_than_16_references(capsys):
    args = ["--models", "models", "--prompt", "Edit", "--output", "output.png"]
    args.extend(value for i in range(17) for value in ("--reference", f"image-{i}.png"))
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
    assert "up to 16" in capsys.readouterr().err


def test_conditioning_preserves_all_16_references(monkeypatch, tmp_path):
    mx = pytest.importorskip("mlx.core")
    from PIL import Image
    from onyx.reference import conditioning

    class Weights:
        def __init__(self, path, resident):
            pass

        def close(self):
            pass

    class Tokenizer:
        @classmethod
        def from_file(cls, path):
            return cls()

        def encode(self, prompt, add_special_tokens):
            return SimpleNamespace(ids=[100 + int(prompt.split()[1][:-1])])

    monkeypatch.setattr(conditioning, "Weights", Weights)
    monkeypatch.setattr(conditioning, "Tokenizer", Tokenizer)

    def vision(_weights, image):
        pixel = image.getpixel((0, 0))[0]
        features = mx.full((2, 3), pixel)
        return features, [features + offset for offset in (1, 2, 3)], (1, 2)

    def latent(_weights, image, _stats, _rng):
        return mx.full((24, 2, 4), image.getpixel((0, 0))[0], dtype=mx.float32)

    monkeypatch.setattr(conditioning, "visual_features", vision)
    monkeypatch.setattr(conditioning, "encode_reference", latent)
    models = tmp_path / "models"
    (models / "vae").mkdir(parents=True)
    (models / "vae/config.json").write_text("{}")
    paths = []
    for i in reversed(range(16)):
        path = tmp_path / f"image-{i}.png"
        Image.new("RGB", (128, 128), (i, 0, 0)).save(path)
        paths.append(path)

    conditioning.prepare(models, tmp_path, [77], paths, 42)
    tokens = json.loads((tmp_path / "tokens.json").read_text())
    assert tokens["count"] == 81
    assert tokens["ids"][::5] == [*range(101, 117), 77]
    vis = mx.load(str(tmp_path / "vision.safetensors"))
    refs = mx.load(str(tmp_path / "references.safetensors"))
    assert vis["rows"].tolist() == [5 * i + row for i in range(16) for row in (2, 3)]
    assert vis["positions"][:5].tolist() == [[0, 0, 0], [1, 1, 1], [2, 2, 2], [2, 2, 3], [4, 4, 4]]
    assert vis["vision"][::2, 0].tolist() == list(reversed(range(16)))
    assert vis["deepstack_2"][::2, 0].tolist() == list(reversed(range(3, 19)))
    assert refs["grids"].tolist() == [[1, 2]] * 16
    assert refs["text_tags"].tolist() == [1, 0, 0, 0, 0] * 16 + [1]
    np.testing.assert_allclose(np.asarray(refs["patches"])[::2, 0], np.arange(15, -1, -1) * 0.999, atol=0.004)

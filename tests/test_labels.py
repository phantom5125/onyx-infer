import json

import pytest
from PIL import Image, ImageChops

from onyx.labels import DEFAULT_FONT, compose_labels, load_labels, main


def write_layout(path, **changes):
    label = {"text": "中文 A-102", "box": [0.1, 0.1, 0.45, 0.18], "color": "#eeeeee"}
    label.update(changes)
    path.write_text(json.dumps({"labels": [label]}, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "changes",
    [
        {"text": "中文\n"},
        {"box": [0.9, 0.1, 0.45, 0.18]},
        {"box": [0.1, float("nan"), 0.45, 0.18]},
        {"leader": [0.2, 1.1]},
    ],
)
def test_reject_invalid_label_geometry_or_text(tmp_path, changes):
    with pytest.raises(ValueError):
        load_labels(write_layout(tmp_path / "layout.json", **changes))


@pytest.mark.skipif(not DEFAULT_FONT.is_file(), reason="Needs the macOS Chinese font")
def test_compose_preserves_pixels_outside_label(tmp_path):
    labels = load_labels(write_layout(tmp_path / "layout.json"))
    image = Image.new("RGBA", (320, 240), "#142d43")

    first = compose_labels(image, labels, DEFAULT_FONT)
    second = compose_labels(image, labels, DEFAULT_FONT)

    assert labels[0].text == "中文 A-102"
    assert first.tobytes() == second.tobytes()
    change = ImageChops.difference(first, image).convert("RGB").getbbox()
    assert change is not None
    left, top, right, bottom = change
    assert 32 <= left < right <= 176
    assert 24 <= top < bottom <= 67


@pytest.mark.skipif(not DEFAULT_FONT.is_file(), reason="Needs the macOS Chinese font")
def test_cli_writes_repeatable_png_with_leader(tmp_path):
    layout = write_layout(tmp_path / "layout.json", leader=[0.8, 0.8])
    original = tmp_path / "original.png"
    Image.new("RGB", (320, 240), "#142d43").save(original)
    first, second = tmp_path / "first.png", tmp_path / "second.png"
    for output in (first, second):
        main(["--input", str(original), "--layout", str(layout), "--output", str(output)])

    assert first.read_bytes() == second.read_bytes()
    with Image.open(first) as image:
        assert image.getpixel((255, 191)) == (238, 238, 238, 255)

"""Render exact labels on an existing image using normalized positions."""

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw, ImageFont


DEFAULT_FONT = Path("/System/Library/Fonts/Hiragino Sans GB.ttc")


@dataclass(frozen=True)
class Label:
    text: str
    box: tuple[float, float, float, float]
    leader: tuple[float, float] | None
    color: str
    background: str | None


def _coordinates(value, count, field):
    if not isinstance(value, list) or len(value) != count or any(
        type(component) not in (float, int) or not 0 <= component <= 1
        for component in value
    ):
        raise ValueError(f"{field} must contain {count} coordinates from 0 to 1")
    return tuple(value)


def load_labels(path: Path) -> list[Label]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("labels"), list) or not data["labels"]:
        raise ValueError("The layout needs a nonempty labels array")
    labels = []
    for index, entry in enumerate(data["labels"], 1):
        if not isinstance(entry, dict):
            raise ValueError(f"Label {index} must be an object")
        text = entry.get("text")
        if not isinstance(text, str) or not text.strip() or text.splitlines() != [text] or "\x00" in text:
            raise ValueError(f"Label {index} needs single-line text")
        x, y, width, height = _coordinates(entry.get("box"), 4, f"Label {index} box")
        if not width or not height or x + width > 1 or y + height > 1:
            raise ValueError(f"Label {index} box must have area and fit inside the image")
        leader = entry.get("leader")
        if leader is not None:
            leader = _coordinates(leader, 2, f"Label {index} leader")
        color = entry.get("color", "#ffffff")
        background = entry.get("background")
        for value in (color, background):
            if value is not None:
                ImageColor.getcolor(value, "RGBA")
        if color is None:
            raise ValueError(f"Label {index} color must be an image color")
        labels.append(Label(text, (x, y, width, height), leader, color, background))
    return labels


def _fit_text(draw, text, font_path, width, height):
    low, high = 1, height
    best = None
    while low <= high:
        size = (low + high) // 2
        font = ImageFont.truetype(font_path, size)
        bounds = draw.textbbox((0, 0), text, font=font)
        if bounds[2] - bounds[0] <= width and bounds[3] - bounds[1] <= height:
            best = (font, bounds)
            low = size + 1
        else:
            high = size - 1
    if best is None or best[1][0] == best[1][2] or best[1][1] == best[1][3]:
        raise ValueError(f"Text does not fit the label box: {text!r}")
    return best


def compose_labels(image: Image.Image, labels: list[Label], font_path: Path) -> Image.Image:
    width, height = image.size
    overlay = Image.new("RGBA", image.size)
    draw = ImageDraw.Draw(overlay)
    line_width = max(1, round(min(width, height) / 512))
    boxes = []
    for label in labels:
        x, y, box_width, box_height = label.box
        left, top = round(x * width), round(y * height)
        right, bottom = round((x + box_width) * width), round((y + box_height) * height)
        padding = min(line_width * 2, (right - left) // 4, (bottom - top) // 4)
        font, bounds = _fit_text(draw, label.text, font_path, right - left - 2 * padding, bottom - top - 2 * padding)
        boxes.append((label, left, top, right, bottom, font, bounds))
        if label.leader is not None:
            anchor_x, anchor_y = round(label.leader[0] * (width - 1)), round(label.leader[1] * (height - 1))
            edge_x = max(left, min(anchor_x, right - 1))
            edge_y = max(top, min(anchor_y, bottom - 1))
            draw.line(((anchor_x, anchor_y), (edge_x, edge_y)), fill=label.color, width=line_width)
            draw.ellipse((anchor_x - line_width, anchor_y - line_width,
                          anchor_x + line_width, anchor_y + line_width), fill=label.color)

    for label, left, top, right, bottom, font, bounds in boxes:
        if label.background is not None:
            draw.rectangle((left, top, right - 1, bottom - 1), fill=label.background)
        x = (left + right - bounds[0] - bounds[2]) // 2
        y = (top + bottom - bounds[1] - bounds[3]) // 2
        draw.text((x, y), label.text, font=font, fill=label.color)
    return Image.alpha_composite(image.convert("RGBA"), overlay)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="onyx compose-labels")
    parser.add_argument("--input", type=Path, required=True, help="Base image")
    parser.add_argument("--layout", type=Path, required=True, help="JSON label layout")
    parser.add_argument("--font", type=Path, default=DEFAULT_FONT, help="OpenType or TrueType font for the text")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG")
    args = parser.parse_args(argv)
    try:
        labels = load_labels(args.layout)
        with Image.open(args.input) as image:
            result = compose_labels(image, labels, args.font)
            profile = image.info.get("icc_profile")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=args.output.parent, suffix=".png", delete=False) as temporary:
            path = Path(temporary.name)
            try:
                result.save(temporary, format="PNG", icc_profile=profile)
                temporary.flush()
                os.replace(path, args.output)
            finally:
                path.unlink(missing_ok=True)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

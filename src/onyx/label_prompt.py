"""Write an H3 prompt with one quoted string per visual label."""

import argparse
import json
from pathlib import Path


DEFAULT_INSTRUCTIONS = (
    "Use the quoted text literally, once per label. "
    "Preserve its spelling, script, punctuation and capitalization."
)


def build_label_prompt(scene: str | None, layout: dict) -> str:
    if not isinstance(layout, dict) or not isinstance(layout.get("labels"), list) or not layout["labels"]:
        raise ValueError("The layout needs a nonempty labels array")
    if scene is None:
        scene = layout.get("scene")
    if not isinstance(scene, str) or not scene.strip():
        raise ValueError("The scene description is required")
    instructions = layout.get("instructions", DEFAULT_INSTRUCTIONS)
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError("The layout instructions must be text")

    lines = [f"{scene.rstrip()} {instructions.strip()}"]
    for index, label in enumerate(layout["labels"], 1):
        if not isinstance(label, dict):
            raise ValueError(f"Label {index} must be an object")
        values = [label.get(field) for field in ("position", "subject", "text")]
        if any(not isinstance(value, str) or not value.strip() or value.splitlines() != [value] for value in values):
            raise ValueError(f"Label {index} needs single-line position, subject, and text")
        position, subject, literal = values
        position = position.strip()
        position = position[0].upper() + position[1:]
        lines.append(f"{position}, pointing to {subject.strip()}: {json.dumps(literal, ensure_ascii=False)}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="onyx label-prompt")
    parser.add_argument("--scene", help="Override the scene description from the layout")
    parser.add_argument("--layout", type=Path, required=True, help="JSON with each label's position, subject, and literal text")
    args = parser.parse_args(argv)
    try:
        print(build_label_prompt(args.scene, json.loads(args.layout.read_text(encoding="utf-8"))))
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

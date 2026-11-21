from onyx.label_prompt import build_label_prompt


def test_literal_text_is_quoted_without_rewriting():
    prompt = build_label_prompt(
        "Exploded view of a tea kettle.",
        {
            "instructions": "Use two columns with leader lines.",
            "labels": [
                {"text": "进气罩", "position": "left, Zone A", "subject": "intake"},
                {"text": "A-102, 螺丝", "position": "right, under the title", "subject": "fastener"},
            ],
        },
    )
    assert prompt.splitlines() == [
        "Exploded view of a tea kettle. Use two columns with leader lines.",
        'Left, Zone A, pointing to intake: "进气罩"',
        'Right, under the title, pointing to fastener: "A-102, 螺丝"',
    ]

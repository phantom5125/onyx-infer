# Text on generated images

## Guide H3 with a structured prompt

Use `onyx label-prompt` when H3 should draw the lettering as part of the image. Put the scene in `scene` and give each item in `labels` a literal `text`, a descriptive `position`, and a `subject` that names the object it should point to. Optional `instructions` specify layout and styling. The command writes the composed prompt to standard output.

```bash
uv run onyx label-prompt --layout benchmarks/requests/text-baseline-jet-engine-structured.json
```

Paste the result into Studio or pass it as a single `--prompt` argument to `native-generate`. The dense Chinese label benchmark measured how many complete labels Apple Vision read exactly once. Inspect lettering and leader placement before using the image. [Text rendering experiments](../benchmarks/requests/text-baseline-notes.md)

## Place exact lettering on an existing image

Use `onyx compose-labels` to place exact lettering over a PNG from `native-generate`. A JSON layout keeps the text, boxes, and leader anchors editable. Change the layout and rerun the compositor to revise the lettering on the same image.

In Onyx Studio, open **Exact text labels** below a generation result. Edit the layout and select **Render labels** to preview and open the labeled PNG. Studio saves the layout and labeled image beside the generation record.

```json
{
  "labels": [
    {
      "text": "搪瓷壶身",
      "box": [0.04, 0.72, 0.22, 0.052],
      "leader": [0.43, 0.64],
      "background": "#293e52"
    }
  ]
}
```

`box` contains left, top, width, and height as fractions of the image dimensions. The origin is the top left. The optional `leader` gives the normalized point the label should identify. Each label takes a single line of Unicode text. By default the text and leader are white and the background is transparent. Set `color` and `background` to choose other colors.

```bash
uv run onyx compose-labels \
  --input artifacts/native-output.png \
  --layout labels.json \
  --output artifacts/annotated.png
```

The default font on macOS is Hiragino Sans GB. Pass an installed font with the glyphs your layout needs using `--font`. Reuse the base image, layout, font file, and Pillow version for repeatable output.

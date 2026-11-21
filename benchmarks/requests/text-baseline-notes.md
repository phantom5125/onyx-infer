# Text rendering regression cases

The 1024×1024 MiniMax-H3-Image text cases exercise rendering accuracy with seed 42 and the default sampler for each path. Keep the original prompts and expected glyphs as benchmark inputs.

- The [jet engine case](text-baseline-jet-engine.json) checks 11 Chinese labels for exact glyph matches. The November 17 Onyx W4 run scored 4/11. The MLX W8A8 run scored about 6/11. A later streamed BF16 reference scored about 6.5/11.
- The [moka pot case](text-baseline-moka-pot.json) checks the short title `MOKA POT`. Its title matched in the W4 run.

Use straight quotation marks when testing the community W8A8 path. In the original experiment, the archived curved-quote form produced an unrelated image on that implementation. Preserve the prompt variant and sampler settings when comparing results.

The jet engine result documents a regression case for dense labels. The sparse moka pot title provides evidence for one short English string. [Detailed Chinese observations](text-baseline-notes.zh-CN.md)

## Structured prompts for dense Chinese labels

The same W4 model and six-step recipe rendered the jet engine at 1024×1024 and 832×1248, with three seeds at each size and prompt setting. The structured prompt names each label's position and object, quotes the exact text, and requests medium-size lettering in two columns.

| Image size | Original prompt | Structured prompt |
| --- | ---: | ---: |
| 1024×1024 | 4 of 33 | 21 of 33 |
| 832×1248 | 5 of 33 | 21 of 33 |

The score counts target labels Apple Vision read exactly once with language correction disabled. The recognizer can miss or misread stylized glyphs. One structured output duplicated a label, and inspection found incorrect leader lines. The study covers one scene and three seeds. [Raw OCR transcripts and image hashes](../results/h3-structured-label-prompt-v1.json)

Print the tested prompt to use with `native-generate --prompt` or paste it into Studio.

```bash
uv run onyx label-prompt --layout benchmarks/requests/text-baseline-jet-engine-structured.json
swift benchmarks/check_label_ocr.swift /path/to/jet-engine-output.png
```

## Isolating text-encoder and DiT precision

An isolated native build streamed the original model's shards. Each comparison used the same structured prompt, token IDs, initial noise, six PDD steps, and VAE. The two W4 control PNGs matched the structured-prompt study byte for byte. A fresh replay reproduced all six latent and PNG hashes.

| Seed | W4 text and W4 DiT | BF16 text and W4 DiT | W4 text and BF16 DiT |
| --- | ---: | ---: | ---: |
| 42 | 9 of 11 | 9 of 11 | 8 of 11 |
| 741 | 7 of 11 | 2 of 11 | 1 of 11 |

These are exact-once Apple Vision reads. Higher weight precision changed the generated results and the set of recognizable labels. The experiment supports keeping the W4 configuration for this case, with the structured prompt and optional exact overlay. [Stage and image hashes](../results/h3-text-precision-ablation-v1.json)

Apply the [diagnostic shard reader](../patches/precision-sharded-weights.patch) in a disposable checkout of `59679ff` and build `onyx-native` with the usual [native build requirements](../../docs/VALIDATION.md). Then run the same two seeds.

```bash
lab=.worktrees/h3-precision
git worktree add --detach "$lab" 59679ff
git -C "$lab" apply --unidiff-zero "$PWD/benchmarks/patches/precision-sharded-weights.patch"
cmake -S "$lab" -B "$lab/.build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$lab/.build" --target onyx-native
uv run python benchmarks/run_text_precision_ablation.py \
  --native-bin "$lab/.build/onyx-native" \
  --original artifacts/models/h3-image-source \
  --w4 artifacts/models/h3-w4 \
  --output artifacts/text-precision-replay
swift benchmarks/check_label_ocr.swift artifacts/text-precision-replay/*.png
```

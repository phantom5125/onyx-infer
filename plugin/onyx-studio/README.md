# Onyx Studio for DeepSeek Harness

Onyx Studio runs the local Onyx Infer engine inside DeepSeek Harness. The workbench supports prompt-based image generation, analysis of engine runs, and an illustrated book reader. [Chinese guide](README.zh-CN.md)

- **Generate** renders images with the native engine, shows live stage progress and memory, and saves a replayable RunRecord.
- **Inspect** shows stage timings, comparable runs, Q4 operator benchmarks, model and precision breakdowns, memory traces, and target hardware.
- **Read** imports whole books in TXT, Markdown, and EPUB. It splits them into chapters and pages and pairs each text page with an illustration slot.

## Install and use

In DeepSeek Harness, open **Plugins → Add plugin** and select this directory. The package is `@local/onyx-studio`. Build the Onyx engine and stage the W4 pack in the engine checkout. Set `engineRoot` in the plugin configuration when the checkout cannot be resolved automatically.

Below a generation result, open **Exact text labels** to edit and render captions and leader lines. The labeled PNG and its layout are saved alongside the generation record. [Layout format and CLI examples](../../docs/IMAGE_TEXT.md)

The reader can render or redraw a page, edit its illustration prompt, and pre-generate the next pages while the engine is idle. It supports ink wash, cinematic, storybook, and minimalist illustration styles. Use Left, Right, and Space to turn pages and Escape to close an open panel.

Agents can call `onyx_generate` to render, `onyx_analyze` to inspect runs, and `onyx_book` to import, list, read, or illustrate pages. Illustration jobs use the same serial queue as image generation. Books and rendered pages live under `artifacts/studio/books/`. Native run records live under `artifacts/benchmarks/onyx-native-h3/`.

## Implementation and checks

- `index.js` registers the host routes and tools and launches generation through `onyx.replay`.
- `client.js` registers the views and sidebar panels using the host-provided React runtime.
- `book_import.py` parses books with the Python standard library. It handles UTF-8 and GB18030, Markdown headings, EPUB spine order, and Chinese chapter markers.
- `sampler.py` records stage and process memory samples during Studio runs.

```bash
node --check index.js
node --check client.js
python3 -m py_compile sampler.py book_import.py
python3 tests/run_book_import_test.py
node e2e.mjs
node smoke.mjs
node e2e-book.mjs
```

The generation and illustration checks require a built engine and W4 pack. Run `e2e.mjs` before the smoke test to create a real run record. Set `ONYX_ENGINE_ROOT` to test a different engine checkout.

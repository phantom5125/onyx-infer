# Contributing to Onyx Infer

Small, verifiable fixes are welcome, especially for Metal kernels, numerical correctness, tests, diagnostics, and documentation.

## Choose where to start

Open a pull request for an individual kernel variant, tile or layout optimization, bug fix, regression test, or documentation improvement that fits the existing module boundaries. Use an issue to track a bug or a small feature.

Start a Discussion for new models, backends, quantization or package formats, changes to the Semantic IR or public contracts, scheduler or memory-management rewrites, major dependencies, and changes to default numerical behavior. Maintainers consider implementation and maintenance cost before taking on cross-module work.

## Kernel changes

1. Start from `main` and give each commit one logical purpose.
2. Define the reference behavior, including the `OpDefinition`, inputs, outputs, dtypes, shapes, quantization, and numerical tolerance.
3. Register the implementation as a separate `KernelVariant`. Keep the reference semantics and tolerances fixed when measuring speed.
4. Cover small matrices, boundary shapes, and at least one production H3 shape. Save the commands and raw results.
5. Run the relevant checks and attach the `onyx.ci.v1` JSON and Markdown reports.
6. Report wall time, peak memory, output hashes, and uncovered cases. Use the median of at least three measured runs for performance claims and retain every sample.

```bash
python tools/native_validate.py \
  --suite kernel \
  --variant tile32 \
  --m 1024 --n 21504 --k 5376 --group 64 --iterations 10 \
  --label pr-123-q4-tile32
```

## Validation

| Level | Checks | Changes that need it |
| --- | --- | --- |
| L0 Contract | Schema, compatibility, Python tests, build | All pull requests |
| L1 Metal | MPSGraph probes, Q4 dequantization and GEMM correctness | Metal, weights, kernels |
| L2 Module | Shape, tile, quantization, and memory benchmarks | Operators, layout, quantization, loading |
| L3 Native | 1024×1024 native generation and replay | Model execution, scheduling, numerics, memory |
| L3 Stress | 4096×4096 native generation | Large shapes, attention, VAE, memory, release validation |

```bash
python tools/native_validate.py --suite contract
python tools/native_validate.py --suite metal
python tools/native_validate.py --suite h3 --case teapot-1024 --runs 3 --label pr-123-native
python tools/native_validate.py --suite h3 --case teapot-4096 --label pr-123-native-4k
```

GitHub-hosted macOS runners handle L0 and L1. Maintainers and contributors with suitable Apple Silicon hardware can run L2 and L3. Attach the reports from `artifacts/ci/` to the pull request and name the cases that need maintainer hardware. The source revision and binary in a report need to match the proposed change. Setup and workflow details are in [Hardware and local validation](docs/VALIDATION.md).

## Generated-image evidence

Any pull request that could change numerical accuracy or visual output must include generated-image evidence. This applies to kernels, MPSGraph changes, dtypes, quantization, model import, execution order, sampling, caching, streaming, memory planning, VAE, and image post-processing—even when operator tests pass or byte-identical output is expected.

- Generate baseline and candidate images from the same request, model, seed, resolution, sampling steps, and execution profile. A new capability without a baseline should explain that exception.
- Upload the original candidate PNG directly to the pull request so it renders in the discussion. For an existing pipeline, upload the baseline PNG beside it; a local path or output hash alone is not visual evidence.
- Use the native 1024×1024 case for general execution changes. Add a text-rendering case when conditioning, quantization, or decoding could affect lettering, and the 4096×4096 case for large-shape, attention, VAE, or memory changes.
- Record the base and candidate revisions, hardware, request, seed, resolution, profile, and both SHA-256 hashes next to the images. Attach the corresponding `onyx.ci.v1` reports and explain every expected visual difference.
- If suitable hardware is unavailable, leave the evidence item unchecked and request maintainer validation. Do not claim that visual accuracy has been verified until someone has inspected the generated output.

Attach ad hoc evidence images and reports to the pull request or workflow instead of committing them. Curated, compressed examples used by documentation are separate from this per-change evidence.

## Pull request checklist

- Keep commits focused and exclude models, generated reports, local environments, logs, and credentials.
- Keep the golden output, tolerance, seed, timing boundaries, and protocol baseline fixed in performance comparisons.
- Discuss public contract changes before proposing them. Regenerate schema bindings with `python tools/compile_schema.py` and run the compatibility check.
- For execution profile changes, follow the [profile evidence rules](docs/PROFILES.md). Results are bound to the profile registry used for the measurement.
- Include reproducible evidence for numerical correctness, boundary cases, peak memory, and replay. Explain any requested follow-up validation.

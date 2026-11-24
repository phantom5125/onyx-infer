# Execution profiles

The profiles share the same W4 model, six-evaluation PDD recipe, and model identity. The profile registry controls execution choices. A new weight format, RNG, or sampling recipe requires a separate identity and validation. [Registry](../config/profiles/profiles.v1.json) · [Evidence report](BENCHMARKS.md)

## Settings

| Profile | Text and DiT weights | VAE weights | VAE tiles | DiT weight cache | Weight prefetch | Q4 linear path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `max-quality` | Stream | Resident | `tile16` | None | Next block | MPSGraph |
| `memory-min` | Stream | Stream | `tile16` | None | Off | MPSGraph |
| `m5-fast` | Stream | Resident | `tile16` | None | Next block | M5 neural accelerators |
| `speed-focus` | Stream | Resident | `tile16` | LRU, 24 GiB | Off | BF16 SIMD |

- The 21.86 GB text and 37.47 GB DiT packages stream on the 36 GB reference machine. The native stage rejects a resident request for either component.
- VAE decoding uses batches of up to 25 tiles by default with `tile16` geometry. Batches with multiple tiles use fused attention, gated activation, and residual graphs. Two rolling FP32 strips retain the rows needed for overlap blending. The streaming mode loads weights per batch and the resident mode reuses weights across batches. Batch size can affect rounding in the output. Use `ONYX_VAE_BATCH=1` to decode tiles individually. [VAE batching measurements](../benchmarks/results/campaign-20261123/round2-vae-tile-batching.json)
- The default MPSGraph Q4 path uses FP32 dequantization. `speed-focus` tests the BF16 SIMD kernel with the DiT weight cache. The vendored Steel FP16 port is available to probes.
- `m5-fast` uses the MLX affine Q4 kernel on M5 neural accelerators through Metal Performance Primitives. The engine falls back to MPSGraph on GPUs without neural accelerators. The kernel uses BF16 scales and biases, which adds resident memory and can change the generated image. [Latency, memory, and output comparison](../benchmarks/results/campaign-20261123/round4-nax-q4-kernel.json)
- The LRU reuses original packed weights across six DiT steps. A 24 GiB limit covers about 64% of the DiT package. The cache test checks equivalent bytes across streaming, LRU, and resident access. Its lifetime relies on synchronous MPSGraph completion.
- The `weight_prefetch` knob overlaps reads for the next text encoder layer or DiT block with GPU computation. It holds an extra set of weights in memory. `memory-min` defaults to serial reads for a smaller memory footprint. [Prefetch measurements](../benchmarks/results/campaign-20261123/round1-prefetch-pipeline.json)

## Validation gates

| Gate | Scope |
| --- | --- |
| `output_replay_identical` | Repeated runs of the same profile and request have the same SHA-256 |
| `profile_echo_must_match` | Native stages declare the expected profile and weights |
| `peak_process_bytes_max` | Process-tree physical footprint, with RSS fallback. Limits are 28.08 GiB for `max-quality`, `m5-fast`, and `speed-focus`, and a provisional 2 GiB for `memory-min`. |
| `min_samples` | Three measured runs per group |
| `bitwise_equal_to` | `memory-min` matches `max-quality` byte for byte |
| `numeric_equal_to` | `m5-fast` and `speed-focus` declare their tolerance to the `max-quality` output |
| `interleaved` | Profiles rotate through the measurement order |

The `speed-focus` speed gate uses a 3% median margin against comparable groups. A result within the margin is accepted. A slower result beyond both the margin and observed dispersion is rejected. Missing comparisons or excessive dispersion are inconclusive. Report inconclusive results explicitly.

Matrices bind to the registry SHA-256, the native binary hash, and the source revision. The checker rejects a mismatched registry. A historical VAE matrix remains available for comparison of the original experiment. Run warmups per resolution and rotate profile order during measured runs. The initial fixed order biased the 4096×4096 resident comparison by 12.2%.

## Running the checks

`onyx native-generate --profile <id>` resolves the registry, sends stage options, checks the returned settings, and records the resolved knobs and registry SHA-256. Each native stage requires explicit `--profile` and `--weights` arguments.

Weight prefetch follows the profile's `weight_prefetch` setting. Use `--prefetch-weights` or `--no-prefetch-weights` to override both text and DiT. Use `--dit-prefetch` or `--no-dit-prefetch` to override DiT alone. These stage-specific flags take precedence for DiT. Explicitly enabling DiT prefetch requires a zero DiT weight-cache budget. The launcher verifies each stage's `prefetch_weights` setting.

```bash
uv run python benchmarks/run_profiles.py --sizes 1024x1024 --runs 3 --output artifacts/benchmarks/profile-stage-matrix.json
uv run python tools/check_profile_gates.py artifacts/benchmarks/profile-stage-matrix.json
uv run python tools/native_validate.py --suite profiles
```

The runner above measures the VAE stage. The hosted profile suite checks registry and evidence with the Python standard library.

## Historical results and next questions

The November 18 VAE matrix used the same FP32 source weights, one warmup per resolution, and three rotated samples per configuration.

| Resolution | `max-quality` | `memory-min` | `speed-focus` |
| --- | --- | --- | --- |
| 1024×1024 | 6.30 s, 1.58 GiB | 10.25 s, 1.57 GiB | 5.55 s, 1.58 GiB |
| 4096×4096 | 85.23 s, 1.87 GiB | 155.28 s, 1.78 GiB | 97.40 s, 1.87 GiB |

All three VAE outputs matched per resolution. The two resident groups had the same VAE settings. At 4096×4096 their within-group dispersion was 39–41%, and the 14.3% median gap was inconclusive. The streaming group dispersion was about 0.3%. [Historical matrix](../benchmarks/results/profile-stage-matrix-v1.json)

Larger VAE tiles, conditioning reuse, and DiT cache tuning need independent evidence. Measure DiT-specific knobs end to end using the full W4 package. The 0.3.0 release comparison used one matched 1024×1024 request per profile and favored the default MPSGraph profile. [Optimization campaign](OPTIMIZATION_CAMPAIGN.md)

# W4 operator optimization

The campaign compared Onyx kernels with the MIT-licensed MLX Steel affine QMM implementation. The custom SIMD candidate improved isolated kernels, and cached MPSGraph remained the fastest measured end-to-end H3 path. [Kernel evidence](../benchmarks/results/m5-max-36gb-q4-kernel-variants-v1.json)

The experiment archive records the Q4 MMA and small-M split-K cases, their acceptance criteria, and the evidence behind each decision. [Kernel experiment contracts](../benchmarks/kernel-experiments/README.md)

## Local single-image campaign

Five rounds on one working condition: a single 1024×1024 request on the M5 Max with 36 GB, the default `max-quality` profile, no on-disk caches, and weights streaming from the internal SSD. Each round is measured with the engine-level interleaved A/B in `benchmarks/ab_compare.py` against the binary of the previous round: one warmup and three measured runs per arm in rotating order, with output hashes compared. [Round records](../benchmarks/results/campaign-20261123/)

The f90b0ce baseline took 42.68 s in a profiled run. DiT took 32.79 s, of which 13.97 s was serial weight reads, 12.44 s Q4 linears, and 4.2 s other graph operations. The text stage took 5.14 s with 3.10 s of reads, and the VAE 4.73 s.

### Round 1: overlap weight reads with GPU compute

The weight store now queues background reads. Two reader threads stay one block ahead of the GPU for every DiT block tensor, for each text encoder layer, and for the AdaLN projections, and the pipeline continues across sampling steps. Without an on-disk modulation cache, the first sampling step computes each AdaLN table just before its block uses it, so its 520 MB projection weight streams alongside the block weights instead of in a serial pass.

| Build | End-to-end samples | Median total | Text | DiT | VAE | Peak RSS |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| f90b0ce, prefetch off | 41.74 / 41.88 / 41.95 s | 41.88 s | 5.23 s | 31.98 s | 4.71 s | 1.79 GiB |
| Round 1, prefetch on | 26.40 / 26.23 / 26.27 s | **26.27 s** | 2.23 s | 19.64 s | 4.39 s | 1.86 GiB |

All six PNGs had the same SHA-256, and both arms read 85,656,876,608 weight bytes. A second interleaved session measured 38.86 → 27.45 s (1.42×) on a warmer page cache, so compare arms within a session and not across sessions. [Round record](../benchmarks/results/campaign-20261123/round1-prefetch-pipeline.json)

Cold model files deliver about 6.4 GiB/s on this machine with any number of reader threads, so the remaining I/O time follows from the bytes read. The AdaLN projections alone stream 26 GB, and an on-disk modulation cache skips them.

### Round 2: decode VAE tiles in batches

The VAE decoder ran one 16×16 latent tile at a time, so a 1024×1024 image made about 14,000 small graph executions with 261 rows in each matrix product. Tiles now decode in batches of 25 in raster order, which multiplies the rows of every matrix product and cuts the dispatch count 25-fold. The tile geometry and the overlap blending are unchanged, and `ONYX_VAE_BATCH=1` restores one tile at a time.

| Build | End-to-end samples | Median total | DiT | VAE | Peak RSS |
| --- | --- | ---: | ---: | ---: | ---: |
| One tile at a time | 24.22 / 24.11 / 24.51 s | 24.22 s | 17.81 s | 4.18 s | 2.15 GiB |
| Batches of 25 | 22.08 / 22.39 / 21.92 s | **22.08 s** | 17.87 s | 2.05 s | 2.10 GiB |

Both arms are the same binary with the batch size switched by environment. **This round changes the output bytes.** Batched matrix products sum in a different order, so the PNG differs from the earlier rounds: 36 of 3,145,728 channel values differ, each by one level of 255 (MAE 4.5e-8, PSNR 97.5 dB). Repeated runs of the batched decoder reproduce the same PNG. [Round record](../benchmarks/results/campaign-20261123/round2-vae-tile-batching.json)

### Round 3: gather the AdaLN modulation on the GPU

Each block and step used to build six per-token modulation buffers on the CPU: six allocations and about 70 MB of copies per block, or 21 GB of copies and 1,800 allocations per image. Every token belongs to one of two modalities, so one graph execution now gathers the six rows that each token needs, with the same values as the copies. The norm that opens a block also moves to the end of the previous block, where it follows the gated residual.

| Build | End-to-end samples | Median total | DiT | Peak RSS |
| --- | --- | ---: | ---: | ---: |
| Round 1 | 26.23 / 26.73 / 26.70 s | 26.70 s | 19.94 s | 2.12 GiB |
| Round 3, gather | 25.38 / 25.92 / 24.96 s | **25.38 s** | 18.33 s | 1.82 GiB |

Both arms decode one VAE tile at a time, so only the gather differs. All six PNGs keep the SHA-256 of the earlier rounds. Sampling steps two to six fell from 2.7–3.0 s to 2.3–2.4 s each, while the first step is bounded by weight reads and did not move. The VAE stage moved between 4.2 s and 5.3 s in both arms. [Round record](../benchmarks/results/campaign-20261123/round3-modulation-gather.json)

A single graph for each gated residual and the norm that follows it was tried and rejected. It changed the PNG, because the fused graph keeps the residual sum at higher precision before the norm reads it, and steps two to six took 2.4–2.6 s in one uncontended run, no better than the separate graphs. The two operations stay separate executions.

### Round 4: Q4 linear layers on the M5 neural accelerators

The MPSGraph Q4 path dequantizes each weight to FP32, casts it to BF16, transposes it, and multiplies. MLX takes another route on M5: `quantized_matmul` dispatches a kernel that dequantizes inside the matrix product and runs the product on the neural accelerators through Metal Performance Primitives. The kernel is vendored with the MLX sources, so the build now compiles it into the Onyx metallib and the engine dispatches it directly. The new `m5-fast` profile selects it and otherwise matches `max-quality`.

| Profile | End-to-end samples | Median total | DiT | Peak RSS |
| --- | --- | ---: | ---: | ---: |
| `max-quality` | 22.94 / 22.69 / 22.71 s | 22.71 s | 18.30 s | 1.67 GiB |
| `m5-fast` | 20.98 / 20.95 / 21.08 s | **20.98 s** | 16.71 s | 3.31 GiB |

The gain is much smaller than the earlier operator comparison suggested, because that comparison ran an MPSGraph probe which included graph compilation. At the engine entry on real block weights, the neural-accelerator kernel is 27% faster for the output projection and 12% faster for the FFN input at 1093 rows, and level for the fused QKV projection: each threadgroup still stages and dequantizes its weight tile in software. The text stage is bounded by weight reads after round 1 and did not move. [Round record](../benchmarks/results/campaign-20261123/round4-nax-q4-kernel.json)

**This round changes the numeric path.** The kernel takes BF16 scales and biases and sums each product in another order. Per layer its outputs differ from the MPSGraph path by one BF16 ulp, and against FP64 reference products on real weights it is as accurate (maximum error 0.068 against 0.082). Across 300 block-steps the differences change the fine detail of the image: correlation 0.987 with the `max-quality` image and PSNR 28.3 dB, with the same composition. The `m5-fast` output replays identically. The BF16 scales and biases raise the peak RSS by about 1.6 GiB.

Two MPSGraph changes were tried at the engine entry. Dequantizing straight to a 16-bit type gave wrong results and is rejected. Moving the transpose off the weight is bit-identical, 6–10% slower at 1093 rows and 8–12% faster at 16384 rows. It stays behind `ONYX_Q4_GRAPH=f32-t` until the 16 MP path is measured end to end.

## Production and isolated measurements

| Configuration | DiT stage | End-to-end per image | Observation |
| --- | ---: | ---: | --- |
| MPSGraph, streamed weights | 26.6 s | 36.8 s | Production default |
| MPSGraph with LRU | 26.6 s | — | Streaming already overlaps I/O |
| BF16 SIMD with LRU | 37.3 s | About 50 s | Experimental path |
| SIMD, streamed weights | About 41 s | — | Experimental path |

One isolated case measured 26.5 ms for tile16, 15.7 ms for tile32, 5.01 ms for SIMD (11.8 TFLOPS), and 5.26 ms for Steel. Across the measured DiT Q4 shapes, the SIMD candidate was about 5.3× faster than tile16 and 4% faster than Steel. Cached production MPSGraph fuses dequantization and matmul and uses shape-specialized graphs.

### Release 0.3.0 retest

One matching 1024×1024 teapot request per profile used the same W4 pack and seed 42 on M5 Max with 36 GB. The default profile took 44.1 s overall and 33.8 s in the DiT stage. The BF16 SIMD profile took 52.1 s overall and 41.8 s in the DiT stage. The default PNG matched the earlier result byte for byte. SSIM between profiles was 0.9580.

### Reuse DiT timestep modulation tables

The six PDD timesteps and the AdaLN projections depend on the model and schedule. Native DiT computes the 50 block tables and the final-layer table before denoising. With `--modulation-cache <directory>`, it can store their BF16 bytes in a 55.5 MiB entry and reuse them across requests. The cache binds the model file identity, schedule, native build, GPU, and macOS version. Cache writes are atomic and entries carry a payload checksum.

On the 36 GiB M5 Max, the fixed 1024×1024 teapot request gave these medians in three rotated blocks after warmup. Cold entries started in empty cache directories while the OS file cache remained warm.

| Modulation cache | End to end | Native DiT | DiT weight bytes read |
| --- | ---: | ---: | ---: |
| Off | 43.14 s | 32.93 s | 85,656,876,608 |
| First use | 42.96 s | 32.78 s | 85,656,876,608 |
| Hit | 38.21 s | 28.11 s | 59,514,797,120 |

All nine measured requests and both warmups produced the same PNG SHA-256. Peak sampled process-tree RSS was 1.78–1.80 GiB across the three measured configurations. System swap usage was unchanged at sample boundaries.

For a 512×768 native stage check, the uncached, cached, and rebuilt latents had the same byte hash. A payload bit flip forced a rebuild and the following request hit the repaired entry. A 4096×4096 request reused the 1024×1024 modulation entry and matched the PNG from the current uncached baseline byte for byte. [Samples and build identities](../benchmarks/results/m5-max-36gb-dit-modulation-cache-v1.json)

### Cache the repeated DiT weights

`native-generate --dit-cache-mib 10240` uses a 10 GiB budget with the default MPSGraph kernel. With the timestep modulation table already cached, this budget covers the repeatedly read W4 weights on the benchmark model. The DiT stage reports the applied budget and the observed weight reads.

With the same 1024×1024 request, same native binary, and a warm modulation table, the median of three paired end-to-end runs fell from **30.69 s to 24.96 s**. Native DiT fell from 20.11 s to 16.76 s. The cached runs recorded 2,015 weight hits and read 11,329,849,920 bytes per run, compared with 59,514,797,120 bytes with weight caching off. Peak sampled process-tree RSS rose from 1.79 GiB to 7.09 GiB. All PNGs matched byte for byte and system swap usage was unchanged at sample boundaries.

The 4096×4096 stress run produced the same PNG hash with this budget. It increased system swap by about 853 MiB and took 318.46 s overall, including 105.26 s in VAE. Keep the default weight budget for 16 MP on this machine. [Paired measurements and stress run](../benchmarks/results/m5-max-36gb-dit-weight-residency-v1.json)

## DiT weight prefetch on M5 Max

Profiling the `f8f73f8` streaming baseline on the 36 GiB M5 Max attributed the 33.14 s DiT stage to 14.20 s of weight reads and 12.98 s of Q4 linear operations. The six steps read 85,656,876,608 bytes. Modulated RMSNorm, fused attention, gated activation, and gated residuals accounted for another 3.20 s. The existing one-block prefetch was opt-in and restarted at each sampling-step boundary.

The candidate enables prefetch by default with a zero weight-cache budget and keeps the one-block pipeline active across step boundaries. Three samples per mode used the same binary and request in off/on/on/off/off/on order, with profiling disabled.

| Prefetch | End-to-end samples | Median total | Median DiT | Peak process-tree RSS |
| --- | --- | ---: | ---: | ---: |
| Disabled | 42.34 / 42.24 / 42.16 s | 42.24 s | 32.00 s | 1.794 GiB |
| Enabled | 32.81 / 32.81 / 33.11 s | **32.81 s** | **22.95 s** | 1.794 GiB |

The median fell 22.3% end to end and 28.3% in DiT. All six PNGs had SHA-256 `4160adb7ac0aa777d6cfe8bafaa9284dd0202198a3ea289c377edf8e93b577d1`. Prefetch waits were 17–23 ms per request. A paired 4096×4096 run with the same binary also produced byte-identical PNGs (SHA-256 `0b9761b2416b57639d9fe387a528136c072b867fa54f394399a651be36557e3f`); those single runs are correctness and memory evidence, not a 4K performance claim. Use `--no-dit-prefetch` to recover serial reads; a nonzero weight-cache budget keeps prefetch disabled unless explicitly requested. [Samples, binary hash, and reproduction commands](../benchmarks/results/m5-max-36gb-dit-prefetch-default-v1.json)

## DiT weight prefetch on MacBook Air M4

The 1024×1024 teapot retest used an M4 with 10 GPU cores, 24 GiB of unified memory, macOS 26.2, and battery power. Each configuration had one warmup and three measured runs in rotating order. All runs used the same W4 pack, `max-quality` profile, seed 42, and six PDD evaluations.

| Native binary and setting | End-to-end samples | Median total | Median DiT | Peak process-tree RSS |
| --- | --- | ---: | ---: | ---: |
| 0.3.1 release | 174.98 / 192.19 / 173.26 s | 174.98 s | 143.48 s | 1.79 GiB |
| Prefetch branch, disabled | 179.54 / 196.47 / 171.72 s | 179.54 s | 148.80 s | 1.79 GiB |
| Prefetch branch, enabled | 177.23 / 184.61 / 159.54 s | 177.23 s | 145.17 s | 1.79 GiB |

Enabling prefetch lowered the median by 1.3% on the same binary. The median with prefetch disabled was 2.6% higher than the release. The sample ranges overlap across all three configurations, so this run cannot establish either a repeatable speedup or a default-path regression. An earlier two-sample comparison per mode on AC power measured a 6.5% lower median with prefetch. The two comparisons used different power conditions and sample counts. The separate M5 Max comparison above showed a clear improvement on that device.

All teapot PNG hashes matched. Earlier paired Moka pot and jet engine outputs also matched byte for byte. The Moka pot title remained legible, and the dense Chinese labels retained the same spelling errors in both outputs. The prefetch path reorders weight reads for one block ahead and uses the same numerical path. The memory column reports sampled process RSS. Metal allocations are recorded separately at DiT step boundaries. [Per-run timings, hashes, and build details](../benchmarks/results/m4-air-24gb-dit-prefetch-v1.json)

To measure prefetch with the local W4 package, use the following command. Pass `--no-dit-prefetch` for the control, and alternate the modes for repeated measurements.

```bash
uv run python benchmarks/run_native.py --case benchmarks/requests/teapot-1024.json \
  --label m4-prefetch --target config/targets/m4-air-24gb.json --dit-prefetch
```

## Correctness and profiling

MLX reference latents correlated 0.9962 with the native MPSGraph path. MPSGraph and native BF16 SIMD latents correlated 0.9973. Image SSIM against the MLX reference was 0.9812 for native MPSGraph and 0.9553 for native BF16 SIMD. The streamed MLX reference took 225.4 s for this comparison and native MPSGraph took 36.8 s.

- A probe initially interpreted BF16 output as FP16. Correcting the readback recovered agreement with CPU reference, including a Steel runtime check on the `m=69` edge tile.
- On the tested Metal 4 driver, a third encoder that consumed a buffer written by the immediately preceding encoder was silently skipped. A separate output conversion command buffer worked around the issue. Direct BF16 removed the conversion passes.
- FP16 residuals exceeded 65,504 near block 14 and produced NaNs. BF16 kept the range needed by the VAE.
- The isolated MPSGraph result included graph compilation. Production reused compiled graphs and measured roughly 0.4–2 ms for QKV calls that the probe reported at 106 ms.
- An apparent 377 non-finite baseline latent values came from reading FP32 storage as `uint16`. The FP32 values were finite.

`ONYX_SIMD_SKIP_ODD` alternated paths within one run. The measured average was 0.347 s per block on MPSGraph and 0.825 s on SIMD, a 2.38× gap. Cross-check equations agreed within 0.8 s. Thermal drift was +2.6% and +6.8% respectively, leaving a gap in in-context efficiency. Legacy GPU stage counters asserted on the reference device, its runtime did not expose the MTL4 counter-heap selector, and Instruments tracing required Xcode.

## Rejected experiments

| Experiment | Result |
| --- | --- |
| BK=64 tile | About 10% slower. Extra scalar writes and threadgroup storage outweighed fewer barriers. |
| Scaled FP16 input and output | Correct and 3.4 s slower. Conversion passes and allocations outweighed the FP16 MMA benefit. |
| LRU for MPSGraph | No measured improvement in the compute-bound path. |
| Buffer allocation cache | The 672 measured allocations consumed 17 ms, below the dominant cost. |

## Reproduce

The native sources include `q4_gemm.metal`, `q4_gemm_simd.metal`, and `q4_gemm_steel.metal`. The Steel and cache probes exercise real packed weights. `tools/profile_quality_gate.py` generates candidate images and reports SSIM.

```bash
python -m onyx quantize --source artifacts/models/h3-image-source --component transformer --all-linear --output "$output/transformer.onxpack"
python -m onyx quantize --source artifacts/models/h3-image-source --component text_encoder --all-linear --output "$output/text_encoder.onxpack"
python -m onyx package-w4 --source artifacts/models/h3-image-source --original-manifest artifacts/models/h3-w4-meta/manifest.pb --output "$output"

python3 tools/profile_quality_gate.py --candidate speed-focus
```

Diagnostic environment variables include `ONYX_SIMD_SKIP_ODD=1` for mixed-path measurements, `ONYX_SIMD_NOWAIT=1` for deferred waits, `ONYX_GPU_COUNTERS=1` for counter probes on eligible devices, `ONYX_PREFETCH_THREADS=<n>` for the number of background readers, `ONYX_PREFETCH_TRACE=1` to list each weight that a stage still reads synchronously, and `ONYX_VAE_BATCH=<n>` for the number of VAE tiles per decoder batch, where 1 reproduces the output bytes of the builds before round 2.

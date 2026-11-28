# Task Contract: Profile foreground DiT weight reads

- Issue #10. Measure which weights delay the default zero-cache prefetch path before selecting blocks for a joint residency and prefetch budget. Use `a82f0cf`, H3 W4, six PDD steps, a warm modulation cache, the `max-quality` profile, and the 1024² and 4096² teapot requests. The older 10 GiB fixed cache was slower than default prefetch on the M5 Max at 1024² (21.88 s versus 18.70 s). The current M5 Max 1024² baseline reports about 0.5 ms in the prefetch wait counter with 59.5 GB read per image. That counter omits direct reads.
- Record foreground synchronous weight reads and genuinely blocked prefetch lookups separately, grouped by model block. Keep the legacy counters and report queue drain time separately. Run the diagnostic independently of timed A/B pairs.
- Profile the 24 GB M4 with model, profile, VAE settings, build, and power conditions pinned. Use the target-checked benchmark commands in [Native tuning](../docs/NATIVE_TUNING.md#measure-foreground-dit-weight-reads). The default arm captures latency without weight profiling, and the diagnostic arm captures per-block waits. Check the latent and PNG hashes across arms. Use the M5 Max to validate the measurement path and record it as a separate target.
- If repeatable block read waits on the M4 justify a residency plan, rank blocks by foreground wait removed per byte. Compare default prefetch, fixed LRU, and bounded joint residency and prefetch at 1024² and 4096² under matched budgeted arms. Include weight and prefetch buffer accounting, latency, read bytes, physical footprint, and swap. Require bitwise outputs and a repeatable improvement inside the weight budget for adoption.

## Foreground read diagnostic on M5 Max (2026-11-27)

| Resolution | Native generations | Synchronous reads, median | Blocked prefetch, median |
| --- | ---: | ---: | ---: |
| 1024×1024 | 3 | 72.55 ms | 0.180 ms |
| 4096×4096 | 1 | 74.44 ms | 0.182 ms |

No DiT block weights had foreground reads or blocked prefetch lookups in these M5 Max samples. All four PNGs matched the incumbent. The weight-store cache test confirms that the counters attribute synchronous traffic to the direct arm and exclude background reader traffic. The 24 GB M4 target needs a separate profile. [Samples and build identity](../benchmarks/results/m5-max-36gb-dit-weight-read-profile-v1.json)

## M5 Max shared-budget experiment (2026-11-28)

- Pin the diagnostic launcher revision `50fc318`, native Release build, H3 W4 with six PDD steps, `max-quality`, seed 42, VAE batch 25, AC power, and a warm modulation cache. Exercise the 1024² and 4096² teapot requests. The no-cache M5 baseline streamed 59.5 GB per generation. The exposure data above produced no recurring waits among the 50 blocks, so equal-priority blocks use execution order as a deterministic tie-breaker.
- Compare the shipping zero-cache pipeline, fixed LRU, and a static residency plan with a bounded two-reader prefetch queue. Start with 4 GiB of total retained and prefetched payload budget for the joint arm. For the legacy LRU arm measure resident scales separately and adjust its separate data-cache cap to match the total to the resolution of the MiB flag. Report both accounting interpretations.
- Reserve bytes at queue admission across queued, loading, and ready buffers. Transfer reservations to resident payload accounting when a selected entry is consumed. Include retained Q4 scales, track peak usage, and compare actual physical footprint and swap independently. Transient weights in active operations and graph workspace are measured via the process footprint.
- Require identical packed weights, output latent and output PNG hashes, positive prefetch and residency use in the joint arm, and peak charged bytes within the budget. Reuse the cache test plus full CTest and the Python contract suite. Profile exposed waits separately from timed A/B runs.
- Compare warmed native runs with rotated arm order. Promote only for a repeatable end-to-end or footprint benefit over the default with no correctness, budget, or small-resolution regression. If a prototype misses the gate, preserve its evidence and retain the shipping policy.

### Result (2026-11-28)

Rejected for M5 Max. The shared policy retained 13 blocks and charged at most 3.81 GiB across resident and prefetched payloads. The fixed LRU uses a 2901 MiB data-cache cap plus 1.17 GiB of resident scales to match the nominal 4 GiB total.

| Resolution | Policy | DiT median | End-to-end median | Weight reads | DiT peak footprint |
| --- | --- | ---: | ---: | ---: | ---: |
| 1024² | Default prefetch | 13.66 s | 17.31 s | 59.51 GB | 6.44 GiB |
| 1024² | Fixed LRU | 20.28 s | 24.05 s | 59.51 GB | 8.99 GiB |
| 1024² | Shared budget | 14.24 s | 17.97 s | 46.99 GB | 8.81 GiB |
| 4096² | Default prefetch | 191.15 s | 209.81 s | 59.51 GB | 12.30 GiB |
| 4096² | Fixed LRU | 204.56 s | 223.69 s | 59.51 GB | 14.74 GiB |
| 4096² | Shared budget | 196.34 s | 215.50 s | 46.99 GB | 14.62 GiB |

The same historical baseline hashes matched for the latent and PNG in both sizes. The shared budget increased peak physical footprint and end-to-end time in all three 1024² rotations. Keep the default streaming prefetch policy on this M5 Max workload. Native elapsed times come from three rotated, profiling-disabled samples per policy and size. Foreground read diagnostics, process RSS, and swap deltas are recorded in the raw report. The default and shared arms' blocked prefetch totals were 0.216/0.196 ms at 1024² and 0.212/0.170 ms at 4096². The 4096² timing spread reached 6.7%, so those medians are descriptive.

[Prototype source](https://github.com/phantom5125/onyx-infer/commit/c7bc05bda7c9fdd1b087d31eaf39b27140671ac9) · [Raw samples](../benchmarks/results/m5-max-36gb-dit-shared-weight-budget-v1.json)

To reproduce the timed comparison from the prototype revision, build a Release native binary and warm the modulation cache. Run this command with each teapot case.

```bash
env -u ONYX_PROFILE -u ONYX_PREFETCH_THREADS -u ONYX_PREFETCH_TRACE uv run python benchmarks/ab_compare.py \
  --label m5-shared-budget-1024 --case benchmarks/requests/teapot-1024.json \
  --target config/targets/m5-max-36gb.json --measure-dit --warmup 1 --rounds 3 \
  --arm 'default .build/onyx-native --dit-cache-mib 0 --dit-prefetch' \
  --arm 'lru .build/onyx-native --dit-cache-mib 2901 --no-dit-prefetch' \
  --arm 'joint .build/onyx-native --dit-cache-mib 4096 --dit-prefetch --dit-shared-budget' \
  --env default=ONYX_VAE_BATCH=25 --env lru=ONYX_VAE_BATCH=25 --env joint=ONYX_VAE_BATCH=25 \
  --env default=ONYX_DIT_WEIGHT_PROFILE=0 --env lru=ONYX_DIT_WEIGHT_PROFILE=0 --env joint=ONYX_DIT_WEIGHT_PROFILE=0
```

For a separate diagnostic run, set the three weight-profiling assignments to 1.

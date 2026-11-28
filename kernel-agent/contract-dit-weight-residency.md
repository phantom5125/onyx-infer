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

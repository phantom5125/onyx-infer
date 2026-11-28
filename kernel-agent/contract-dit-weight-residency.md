# Task Contract: Profile foreground DiT weight reads

- Issue #10. Measure which weights delay the default zero-cache prefetch path before selecting blocks for a joint residency and prefetch budget. Use `a82f0cf`, H3 W4, six PDD steps, a warm modulation cache, the `max-quality` profile, and the 1024² and 4096² teapot requests. The older 10 GiB fixed cache was slower than default prefetch on the M5 Max at 1024² (21.88 s versus 18.70 s). The current M5 Max 1024² baseline reports about 0.5 ms in the prefetch wait counter with 59.5 GB read per image. That counter omits direct reads.
- Record foreground synchronous weight reads and genuinely blocked prefetch lookups separately, grouped by model block. Keep the legacy counters and report queue drain time separately. Run the diagnostic independently of timed A/B pairs.
- Profile the 24 GB M4 with model, profile, VAE settings, build, and power conditions pinned. Use the M5 Max to validate the measurement path and record it as a separate target. Check that diagnostic runs reproduce the incumbent latent and PNG hashes.
- If repeatable block read waits on the M4 justify a residency plan, rank blocks by foreground wait removed per byte. Compare default prefetch, fixed LRU, and bounded joint residency and prefetch at 1024² and 4096² under matched budgeted arms. Include weight and prefetch buffer accounting, latency, read bytes, physical footprint, and swap. Require bitwise outputs and a repeatable improvement inside the weight budget for adoption.

## Foreground read diagnostic on M5 Max (2026-11-27)

| Resolution | Native generations | Synchronous reads, median | Blocked prefetch, median |
| --- | ---: | ---: | ---: |
| 1024×1024 | 3 | 72.55 ms | 0.180 ms |
| 4096×4096 | 1 | 74.44 ms | 0.182 ms |

No DiT block weights had foreground reads or blocked prefetch lookups in these M5 Max samples. All four PNGs matched the incumbent. The weight-store cache test confirms that the counters attribute synchronous traffic to the direct arm and exclude background reader traffic. The target M4 24 GB needs its own profile before a joint policy can be evaluated. [Samples and build identity](../benchmarks/results/m5-max-36gb-dit-weight-read-profile-v1.json)

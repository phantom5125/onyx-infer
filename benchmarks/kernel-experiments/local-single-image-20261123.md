# Local single-image optimization campaign

This case records the 2026-11-23 contract and the outcome of each round. See the
[candidate ledger](candidates.jsonl) for individual trials and the
[round records](../results/campaign-20261123/) for samples, output hashes, and binary
identities. The [benchmark summary](../../docs/OPTIMIZATION_CAMPAIGN.md#single-image-performance)
compares the measured builds.

## Task Contract: local single-image campaign

- Task name: `local-single-image-20261123`
- Working condition: one 1024×1024 image per request on the M5 Max with 36 GB, the
  default `max-quality` profile, no on-disk caches, and W4 weights streaming from
  the internal SSD. The teapot request with seed 42 is the fixed workload.
- Baseline attribution (f90b0ce, 42.68 s): DiT 32.79 s (13.97 s of serial weight
  reads, 12.44 s of Q4 linears, 4.2 s of other graph operations), text 5.14 s
  (3.10 s of reads), VAE 4.73 s.
- Objective: five rounds that each remove the largest measured bottleneck. The PNG
  stays byte-identical unless a round changes the numeric path by design.
- Method: engine-level interleaved A/B with `benchmarks/ab_compare.py`, one warmup
  and three measured runs per arm in rotating order, against the binary of the
  previous accepted round. An isolated kernel win counts only after this check.
- Constraints: the W4 pack and its packing format stay unchanged, the model recipe
  stays unchanged, nothing writes outside the request's output and cache
  directories, and peak memory stays within the profile gates.
- Gates: ctest and pytest at L0. At L3, the PNG SHA-256 equals the f90b0ce output
  for rounds that keep the numeric path; a round that changes it declares a
  tolerance and reports the pixel differences.
- Promotion: a median improvement wider than the spread of the interleaved samples,
  with both arms sharing the page-cache state of the model files.

## Outcome, round 1: weight I/O overlapped with GPU compute

Promoted: a queue-based reader pipeline in `WeightStore`. Every DiT block tensor,
each text encoder layer, and the AdaLN projections stream one block ahead, and the
pipeline stays full across sampling steps. Without an on-disk modulation cache, the
first sampling step computes each AdaLN table just before its block uses it.

- Interleaved A/B on M5 Max: 41.88 → 26.27 s median (1.59×), text 5.23 → 2.23 s,
  DiT 31.98 → 19.64 s. A second session measured 1.42× on a warmer page cache.
- PNG SHA-256 unchanged; 85,656,876,608 weight bytes read in both arms.
- Cold model files deliver 6.4 GiB/s regardless of reader count, so I/O volume, not
  reader parallelism, bounds what remains: the 26 GB AdaLN pass takes at least 3.9 s.
- Evidence: `benchmarks/results/campaign-20261123/round1-prefetch-pipeline.json`.

## Outcome, round 2: VAE tiles decode in batches

Promoted as a numeric change: the VAE decoder takes 25 tiles per pass instead of one.
The tile geometry and the overlap blending are unchanged.

- Same binary with the batch size switched by environment, interleaved A/B: VAE
  4.18 → 2.05 s median, total 24.22 → 22.08 s (1.10×). The DiT stage matches across arms.
- The PNG is no longer byte-identical to round 1. Batched matrix products sum in a
  different order: 36 of 3,145,728 channel values differ by one level of 255, MAE
  4.5e-8, PSNR 97.5 dB. Replay within the batched decoder is identical.
- Gate declared for this round: at most one 8-bit level per channel value and at
  most 0.01% of the values differing.
- `ONYX_VAE_BATCH=1` restores one tile at a time and the earlier output bytes.
- At 512×768, batching cut VAE time from 2.45 to 1.51 s. It changed 17 of
  1,179,648 channel values by one level.
- Evidence: `benchmarks/results/campaign-20261123/round2-vae-tile-batching.json`.

## Outcome, round 3: AdaLN modulation gathered on the GPU

Promoted: one MPSGraph gather per block replaces six per-token CPU copies of the
modulation rows, which cost about 70 MB of copies and six allocations per block
and step.

- Interleaved against the round-1 binary with the VAE unbatched in both arms:
  DiT 19.94 → 18.33 s median, steps two to six 2.7–3.0 → 2.3–2.4 s, peak RSS
  2.12 → 1.82 GiB. The PNG keeps the SHA-256 of round 1.
- Rejected: a fused gated-residual-plus-norm graph. It changed the PNG and
  showed no gain beyond noise, so a numeric change bought nothing.

## Outcome, round 4: Q4 linears on the M5 neural accelerators

Promoted as a numeric change, behind the new `m5-fast` profile: MLX's
`affine_qmm_t_nax` is compiled from the vendored source into the Onyx metallib
and dispatched directly by the engine through Metal Performance Primitives.

- Interleaved against `max-quality`: 22.71 → 20.98 s median (1.08×), DiT
  18.30 → 16.71 s.
- The historical 16.5× probe comparison included MPSGraph graph compilation. At the
  engine entry on real weights the neural-accelerator kernel is about 25%
  faster for the output projection and 12–24% faster for the FFN input, and
  level within noise for QKV (13–18 ms against 14–15 ms over five runs),
  because each threadgroup still stages and dequantizes its tile in software.
- Numeric evidence: one BF16 ulp in 46% of the outputs of every layer against
  MPSGraph. Versus FP64 reference products both paths sit 2–5 half-ulps of the
  typical output from exact; the kernel is closer on QKV and farther, by up to
  1.6×, on the FFN input. Image correlation 0.987, replay identical.
  Peak RSS 1.67 → 3.31 GiB.
- Rejected: dequantizing to a 16-bit type (wrong results). Deferred: moving the
  transpose off the weight (`ONYX_Q4_GRAPH=f32-t`). The output is bit-identical.
  Latency increased 6–10% at 1093 rows and decreased 8–12% at 16384 rows.
- Reproduce the per-layer timings and FP64 reference comparison with
  `onyx-q4-path-compare`.
- Evidence: `benchmarks/results/campaign-20261123/round4-nax-q4-kernel.json`.

## Outcome, round 5: modulation cache on by default

Promoted: the launcher caches the AdaLN tables in the user cache directory,
keeps the two newest entries, and offers `--no-modulation-cache`.

- Same binary with the cache switched by flag, interleaved: 22.71 → 19.35 s
  median (1.17×), DiT 18.19 → 14.84 s. Weight bytes read 85.7 → 59.5 GB and the
  first sampling step 5.7 → 2.4 s.
- The PNG keeps the SHA-256 of the uncached path. A first request with a new
  engine build is a miss and costs the same as before.
- Evidence: `benchmarks/results/campaign-20261123/round5-default-modulation-cache.json`.

## Measurement note: another session on the same machine

A second session ran native benchmarks from another worktree during this campaign.
Two samples of a first round-2 session overlapped it and ran 50% slower, so that
session was discarded. `benchmarks/ab_compare.py` now waits before each sample
until no other native stage process runs and records any that appear, which makes
a contended sample visible in the report.

## Comparison with DiT prefetch build

The `e0b4244` build was compared with `35bc6f9`, which already streams DiT weights one block ahead. Each build ran from its own checkout with its own launcher.

- 1024×1024 teapot, three runs per arm: 33.54 → 19.58 s (1.71×). The PNG differs in 36 of 3,145,728 channel values, each by one level.
- 4096×4096 teapot, one run per arm: 323.04 → 227.93 s (1.42×), VAE 99.6 → 29.0 s, peak RSS 2.66 → 1.91 GiB. The PNG differs in 653 of 50,331,648 channel values, each by one level.
- In this comparison, `m5-fast` took 19.96 s at 1024×1024 and 218.44 s at 4096×4096.
  Its image correlation with the matching `max-quality` outputs was 0.987 and 0.996, respectively.
- Text-rendering cases at 1024 keep their lettering: the moka-pot title is legible in both, and the jet-engine labels carry the same misspellings in both.
- Evidence: `benchmarks/results/campaign-20261123/against-main-*.json` and `generated-image-evidence.json`.

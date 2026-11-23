# Local single-image optimization campaign

This case records the 2026-11-23 contract and the outcome of each round. See the
[candidate ledger](candidates.jsonl) for individual trials and the
[round records](../results/campaign-20261123/) for samples, output hashes, and binary
identities. The [optimization campaign](../../docs/OPTIMIZATION_CAMPAIGN.md#local-single-image-campaign)
holds the narrative.

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
- Evidence: `benchmarks/results/campaign-20261123/round2-vae-tile-batching.json`.

## Measurement note: another session on the same machine

A second session ran native benchmarks from another worktree during this campaign.
Two samples of a first round-2 session overlapped it and ran 50% slower, so that
session was discarded. `benchmarks/ab_compare.py` now waits before each sample
until no other native stage process runs and records any that appear, which makes
a contended sample visible in the report.

---
title: Four rounds, one byte-identical teapot
date: 2026-11-18
summary: 42.30 s → 34.65 s across four optimization rounds on the same 1024 px teapot request — with every accepted change proving byte-identical output. Round 1 made things slower, and we published that too.
tags: Optimization
---

## The rule

No performance conclusion without evidence. Every optimization candidate must bring three
things: a committed measurement JSON, passing correctness gates, and — for the native
pipeline — the output hash of a fixed reference request (the teapot, seed 42). If a candidate
is accepted, the diff proves the pixels did not move.

## The trajectory

Same machine, same request, four rounds:

| Round | Commit | Change | 1024 median | Output |
| --- | --- | --- | --- | --- |
| Baseline | `b5bf491` | Full C++/Metal pipeline | 42.30 s | `d3e32281…` |
| 1 | `4bd6893` | Q4 grouped zero-point caching | 42.63 s | identical — and slower |
| 2 | `e3d1b87` | Weights `pread` directly into `MTLBuffer` | 38.74 s | identical, peak memory −46.7% |
| 3 | `fb4c0aa` | Per-token BF16 modulation packed in shared CPU memory | 37.50 s | identical |
| 4 | `40420aa` | Fused QKV transforms + Q/K RMSNorm + RoPE + SDPA; native offset views kill tensor copies | **34.65 s** | identical |

Cumulative: **1.22×** over the v1 baseline. The stage split at round 4 is
text 3.83 s / DiT 26.02 s / VAE 4.41 s, with peak GPU allocation around 2.0 GB for a
1024×1024 render.

## Round 1 made things slower

Theory said caching grouped zero-points should help; measurement said 42.63 s against a
42.30 s baseline. The v1 review publishes that row anyway. We would rather have a small
truthful number than a big story — an optimization log you cannot trust is worse than no log.

## Why byte-identical matters

- It converts a subjective quality debate into one hex string:
  `d3e3228138e309c627d0c48e2bcbb1597a18fdf5c2ebccc1767c1ecef232b8c8`.
- It lets us chase aggressive tricks — round 4 fuses five operations into one dispatch and
  replaces copies with `MPSNDArray` offset views — at full speed, because if the hash moves,
  the change moves out.
- It makes automated optimization safe. Our Loop controller can only submit candidates
  through the public protocols and must attach evidence; it cannot redefine what "correct"
  means.

## Where this goes

The next rounds aim at the two gaps we publish rather than hide: our own Q4 GEMM kernel is
still ~16× slower than MLX's `quantized_matmul` on the same shape, and the community MLX
W8A8 path is ~1.64× faster end-to-end on the same request. Those numbers live in
`benchmarks/results/`, right next to the code that produced them.

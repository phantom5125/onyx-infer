---
title: Sixteen megapixels on 36 GB of unified memory
date: 2026-11-18
summary: 4096×4096 direct generation — no upscaler — in 329.6 s with a 2.14 GB peak on a 36 GB Mac Studio. The memory choreography that makes it fit.
tags: Memory
---

## The budget is smaller than the weights

The target machine grants 28.08 GiB of recommended GPU working set and caps a single buffer
at 21.06 GiB. Against that: a 21.86 GB W4 text pack and a 37.47 GB W4 DiT pack (high-precision
AdaLN projections included). Neither pair fits resident at once; the DiT does not even fit by
itself. "Unified memory" does not mean "no memory management" — it means the management is
visible, plannable, and ours.

## Phase residency, not one resident graph

The engine executes in phases with explicit handoffs:

1. **Text phase** — the Qwen3-VL text tower streams block-by-block, keeps the layer-50 hidden
   states, then releases the pack.
2. **DiT phase** — the 37.47 GB pack streams block-by-block. Timestep and AdaLN projections
   are precomputed once, their weights released; each transformer block's weights are evicted
   after use.
3. **VAE phase** — the FP32 2D-ViT decoder (2.34 GB) runs with tiling on the 16×-downsampled
   latent grid.

## Attention must be chunked

At 4096×4096, H3's design — one image token per 32×32 pixel patch — yields
(4096/32)² = **16,384 image tokens**, before text and reference tokens lengthen the sequence.
A materialized attention matrix is out of the question at 16 MP; SDPA runs on chunked windows.
The tile experiments behind this live in `benchmarks/results/m5-max-36gb-q4-tile32-v1.json`.

## The result

The 4096×4096 teapot request: **329.61 s**, peak **2,144,190,464 bytes (~2.14 GB)**, full
RunRecord stored and replayed. This is direct generation — the DiT saw a native 4K token grid
— not a diffusion pass at 1024 followed by an upscaler, which is why structure and label text
survive at 16 MP.

## What we will not claim

W4 output at these resolutions passes our functional and visual checks on reference fixtures.
That is not a broad quality-equivalence claim: multi-prompt evaluation, more resolutions and
more seeds are queued in the Loop. Published numbers first; adjectives later.

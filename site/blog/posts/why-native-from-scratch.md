---
title: Why we're building an inference engine from scratch
date: 2026-11-18
summary: 135 GB of checkpoint weights, a 36 GB machine, and a hard rule — no open-source runtime in the dependency graph. The math and the principles behind Onyx Infer.
tags: Design
---

## The math forces the architecture

MiniMax-H3-Image is a big checkpoint. The original weights add up to roughly **135 GB**:
the Qwen3-VL text tower at ~66.7 GB, the 33.1B DiT at ~66.28 GB, and a 2.34 GB FP32 VAE.

Our first target machine is a Mac Studio (M5 Max, 36 GB unified memory), and the operating
budget is not "36 GB" — it is what Metal actually grants:

| Constraint | Value on the target machine |
| --- | --- |
| `recommendedMaxWorkingSetSize` | 28.08 GiB |
| `maxBufferLength` | 21.06 GiB |
| Ideal W8 DiT weights | ~33.14 GB — already over budget, before activations |

So lossy weight quantization is not a stylistic choice, it is a **requirement**. We picked
group-wise W4A16 as the *first candidate*, and we are careful with that word: candidate, not
proven conclusion. Measured W4 packs come out at **37.47 GB for the DiT** (AdaLN projections
kept in high precision, per the official recipe) and **21.86 GB for the text encoder** — each
fits alone, never together. That single fact dictates the whole execution model: phases, not
one resident graph.

## Semantics and execution are different artifacts

The engine splits cleanly:

- The **semantic IR** says *what* to compute: a pure dataflow graph with symbolic shapes and
  explicit numeric + RNG semantics.
- The **compiler/planner** says *how to compute it on this machine*: phase residency, layouts,
  fusion, tiling, kernel selection.

That split is why our frozen v1 contracts cover requests, artifact identity, tensor semantics,
run events and evidence — but deliberately do **not** freeze Metal objects, thread counts,
tile sizes or cache formats. Those are optimization territory, and they must stay free.

## Why not wrap an existing engine?

Not NIH — three concrete engineering reasons:

1. **Claims need proofs.** Every performance claim in this repository ships with a measurement
   JSON file and, for the native pipeline, the output SHA-256 of a fixed reference request.
   Saying "byte-identical" honestly requires owning the entire path; a borrowed runtime makes
   byte-identity someone else's bug.
2. **The optimization Loop needs bypass-proof interfaces.** Candidates — whether authored by a
   human or an automated agent — enter through the same public protocols and must attach
   evidence. Nobody edits production semantics under the table.
3. **References should stay references.** The official PyTorch/Diffusers implementation is used
   *offline*: to import model semantics and to produce golden tensors we compare against. It
   never enters the runtime dependency graph.

## Default: no implicit behavior

No prompt rewriting. No silent network calls. No silent quantization. No silent spilling to
disk. No capability guesses from device strings — the runtime queries the actual Metal budgets
and plans within them. If the engine wants to do something you didn't ask for, it has to ask.

## What this buys

A development loop where "faster" always comes with receipts — measurement files, correctness
gates, and output hashes that prove quality did not move. The next post shows what that looks
like in practice: four optimization rounds against one teapot.

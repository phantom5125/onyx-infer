# Task Contract — H3 AdaLN for 16 reference images

- Task name: `h3-ref16-adaln-v1`
- Budget: one Episode.
- Objective: reduce the DiT memory traffic and live allocations from broadcasting the six AdaLN parts out of MPSGraph. The incumbent H3 reference path feeds modality IDs 0 (target and vision tokens), 1 (text), and 2 (reference latents) into the standalone gather before each block.
- Baseline: the earlier M5 Max 36 GB block profile on the local project checkout measured 66.99 ms at 1,024 rows, including 39.71 ms for Q4 linears (see the November 24 fused-block contract). Its RMS calls consumed 10.18 ms per block. That profile predates the current gather and has no 16-reference timing. At 33,942 rows the current code explicitly allocates 2,189,666,304 bytes for six BF16 results per block, or 656,899,891,200 bytes of output writes across six × 50 blocks. These byte counts follow the allocated shapes, independent of timing or driver allocation policy.
- Inputs: compact BF16 table [18, 5376], int32 indices [rows] with all three modality IDs, BF16 activation and residual tensors [rows, 5376], BF16 RMS weights [5376]. The 33,942-row case uses 16 square references at 1,024 pixels after encoding resize, a 1,024-square target, 119 label and delimiter tokens, and a 31-token prompt.
- Output: BF16 activations with the incumbent RMS, scale, shift, and residual arithmetic and unchanged DiT attention and Q4 linears.
- Correctness: every output word of the two normalization and two residual operations must equal the incumbent BF16 result at 1, 17, and 33,942 rows. The real-weight reference-edit output must match the incumbent PNG hash. A difference in either check rejects promotion.
- Constraints: fold each compact-table lookup into its normalization or residual consumer. Reuse MPSGraph RMS and residual arithmetic. Keep graph cache keys shape-based, use the existing table layout and weights, and introduce no persistent tensor allocations. Enable the candidate only for reference edits when `ONYX_DIT_INDEXED_MODULATION=1`.
- Validation: `cmake --build .build --parallel 6`, `ctest --test-dir .build --output-on-failure`, and `.build/onyx-dit-modulation-probe 33942`. The CTest cases cover 1 and 17 rows.
- Evaluation: the target-shape probe warms both variants then alternates A/B/B/A across three blocks. Report the per-block medians, between-block drift, and explicit bytes of modulation outputs. For the end-to-end anchor, run the same real-weight 16-reference edit with the flag off and on, record the paired DiT wall times, and compare PNG SHA-256.
- Promotion: strict ABBA median improvement with at most 5% between-block drift, full CTest success, matching real-weight output, and an end-to-end DiT gain. Retain the incumbent by default if the evidence is incomplete.

## Design

TensorRT's pointwise fusion and TorchInductor's fusion guidance motivate moving each lookup inside the consumer graph to avoid a separate program boundary for the expanded values. The six explicit shared-buffer results cover the whole block in the baseline. The candidate feeds compact rows and token indices to each consumer. MPSGraph may still use internal memory for a gather. Measure latency and physical allocation before claiming engine-level gains.

The existing Metal SDPA path already composes QKV layout, RoPE and fused attention. FlashAttention motivates avoiding quadratic intermediate traffic, but this Episode leaves SDPA and the 16 input images intact. Replacing attention, shortening the reference sequence, and reusing a prefix across DiT denoising steps would each require a separate contract. DiT attention is bidirectional, so reference states change as the target evolves.

References: [TensorRT layer fusion](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/optimization.html), [TorchInductor fusion](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_get_started.html), [FlashAttention](https://proceedings.neurips.cc/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract.html).

## Episode 1 (2026-11-25)

Rejected for default promotion. Keep the indexed modulation path available with
`ONYX_DIT_INDEXED_MODULATION=1` for reference edits.

- The accepted 33,942-row ABBA batch measured 79.250 ms → 45.389 ms for the
  four AdaLN consumers and the incumbent standalone gather. The three block
  medians drifted 2.61% and 3.25% respectively. The first three timing batches
  exceeded the drift limit and were discarded. Warmup and sample counts are in
  the results file.
- BF16 mismatch count was zero at 1, 17, and 33,942 rows. All 21 configured
  CTest cases and 78 Python tests passed. With 16 real reference images and a
  1024-square canvas, both runs generated the same PNG SHA-256
  `e2ff4e1eede9c0373b38cd21f711d7877f5634e4ab44929123b169179c678e90`.
- Peak physical footprint of the DiT child measured 20.22 GiB for the baseline
  and 18.35 GiB for the indexed path in this pair. The step-end Metal counters
  and the exact byte counts are in the results file.
- Another native 1024-square DiT job overlapped the indexed run. Keep both
  raw timings for traceability and exclude them from speed claims and default
  deployment decisions. The isolated operator ABBA was complete before the
  reference-edit pair started.

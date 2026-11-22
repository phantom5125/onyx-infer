# Task Contract: q4_gemm simdgroup MMA candidate

- Task name: `q4-mma-v1`
- Objective: Replace the scalar-FMA inner loop of `onyx_q4_gemm_fp16acc32` with
  simdgroup-MMA (Apple tensor-core) execution to close the measured 16.5x gap
  against MLX quantized_matmul (baseline 28.1 ms on m1024 n5376 k5376 g64).
- Inputs: fp16 activation [m,k]; packed q4 weights [n,k] (unsigned nibble
  q+8, low=even-k, high=odd-k); fp32 scales [n, k/group].
- Output: fp16 C[m,n] with fp32 accumulation (numeric profile
  `fp16.activation.fp32.accumulation` must not regress).
- Correctness requirements: `onyx-metal-bench` CPU-reference check passes
  (absolute <= 0.05 + 0.005*|expected|); max_relative_error must stay within
  2x of the tile16 baseline (0.00147).
- Constraints: single Metal kernel, no extra buffers beyond threadgroup
  storage, packing format unchanged (no re-quantization), same shared-memory
  storage mode, dispatchable from `tools/metal_bench.mm --variant mma`.
- Validation command: `cmake --build .build && ctest -R mma` plus
  `onyx-metal-bench --m 1024 --n 5376 --k 5376 --group 64 --iterations 10 --variant mma`.
- Evaluation command: ABBA alternation tile16/mma, 4 blocks x 10 iterations,
  compare gpu_median_ms geomeans.
- Promotion criteria: strict gpu_median improvement over tile16 with matching
  correctness; regression on any CI shape (ctest suite) rejects the candidate.

# Draft

Baseline (recorded, m5-max-36gb-q4-operator-v1.json):
- tile16: 28.1 ms median, ~2.1 effective TFLOPS (~7% of MMA peak).
- MLX quantized_matmul: 1.7 ms median.
- Kernel defects identified by static review of `ops/metal/q4_gemm.metal`:
  scalar FMA inner loop, weights dequantized to fp32 in threadgroup memory,
  one output per thread, single-byte packed-weight loads.

Candidate directions ranked:
1. (this candidate) simdgroup MMA: one simdgroup (32 threads) per threadgroup,
   32x32 output tile, k stepped by 8; A staged as fp16, weights unpacked
   directly to fp16 (no fp32 round-trip), uchar4-aligned nibble loads when
   k % 8 == 0, scalar nibble loads otherwise.
2. Wider tile / 2 simdgroups per threadgroup (64x32) — defer, needs more
   registers; only if candidate 1 lands and profiling shows occupancy slack.
3. Persistent kernel + fused epilogue — out of scope for v1.

Risks: fragment-layout assumptions (simdgroup_load/store byte strides),
fp16 staging precision of weights (nibbles are exact in fp16, scales applied
in fp16 lose bits — mitigation: apply scale in fp16 only when group<=128 and
validate; otherwise keep weights pre-scaled per k-step in fp16, accepting the
validated tolerance). First step: implement, run ctest small shapes, then the
1024x5376x5376 evaluation.

# Outcome (episodes E1-E8, 2026-11-17)

Promoted: mma (E1, 32x32 single-simdgroup), mma64 (E2, 64x64 four-simdgroup),
per-shape dispatch (E6: m>=512 -> mma64, m<=32 -> tile16).
Final vs incumbent tile16: linear 3.2x (26.9 -> 8.4 ms), qkv 3.23x
(107.2 -> 33.2 ms). decode-linear 0.473 ms beats MLX 0.726 ms.
Open gap: decode-qkv 1.81 ms vs MLX 0.520 ms; requires an MMA-based
split-k small-M kernel (scalar FMA tops out near 8 TFLOPS; E7/E8 documented).

Runtime integration is not wired (bench-level evidence only); the runtime
still reaches these operators through MPSGraph. Wiring mma64 into
runtime/native dispatch is the follow-up step before end-to-end gains.

# Engine integration + MLX gap analysis (E9-E10, 2026-11-18)

mma64 is now wired into GraphSession::linear for quantized weights with
m >= 512 (small m keeps the MPSGraph dequantize+matmul path); verified by
the new onyx_q4_linear_dispatch ctest (both paths vs CPU reference through
the real WeightStore format). Bias handled via broadcast add.

MLX gap root cause, measured: a 4-layout minimal MMA probe proves
thread_elements() never reaches MMA inputs on the metal4 toolchain, so
register-side dequant is impossible with public APIs; software pipelining
at BK=8 regressed 2.2x. Practical public-API ceiling ~9 TFLOPS effective
for this q4 GEMM vs MLX ~43 TFLOPS. Closing further requires Apple-internal
primitives (MLX is Apple's own) or toolchain changes; documented as
anti-strategies in the atrex gpu-wiki.

# Runtime selection in 0.3.0

The native engine defaults to MPSGraph based on the end-to-end H3 W4 measurements
in [the optimization campaign](../docs/OPTIMIZATION_CAMPAIGN.md). The Python launcher
selects the Q4 kernel from the requested profile. For direct native probes,
`ONYX_Q4_BACKEND=mlx` selects the optional FP16 MLX path on supported shapes,
`native` selects FP16 MMA64 at large row counts, and `auto` tries both in that order.
Other inputs use MPSGraph. Build MLX support with the installed wheel or disable it
with `-DONYX_ENABLE_MLX=OFF`.

# MLX backend (E12, 2026-11-18)

The 0.2.0 experiment used the MLX backend for FP16 quantized linears with
mlx::core::quantized_matmul (C++ API, links .venv libmlx.dylib):
- Engine m1024 linear 4.10 ms median (6.6x vs tile16 baseline, 3.2x vs our
  best native kernel); m16 decode 0.374 ms (beats tile16 0.473 and
  python-mlx 0.726). ctest 7/7.
- Format bridge: biases = -8 * scales, uint32 repack (8 nibbles/word).
- Pitfalls recorded in gpu-wiki: converting-constructor semantics, no-sync
  fp16 read-back garbage, foreign-pointer zero-copy segfault, metallib must
  be registered via metal::set_metallib_path, group sizes 32/64/128 only.
Remaining per-call overheads: fp16->float conversion of activations and
fp32 read-back (~1 ms total at m=1024); true zero-copy needs MLX-allocator
buffers end-to-end (future work).

# Block-level fusion (E13, 2026-11-18)

Whole-block mlx lazy graph (fused q4 linears + fast SDPA + fast rms_norm +
fused elementwise, residual resident in mlx) validated stage-by-stage vs
fp32 references; 26.9 ms/block at m1024 = 2.17x over the per-op MLX path
(58.8 ms) => DiT 17.6 -> 8.1 s. Time is now ~90% the four quantized_matmul
kernel invocations - at MLX-kernel SOTA parity per E9 ceiling analysis.
Remaining 1.5x-over-SOTA would need beating MLX's own qmm kernels (blocked
by E9 public-API evidence) or zero-copy bridge (E12 follow-up).
Engine wiring (dit.mm block loop -> per-step mlx graph with rope via
fast::rope + freqs) is the pending integration step; contract + probe
validated separately.


# Release 0.2.0 (2026-11-19)

Highlights over 0.1.0a0:
- Quantized linears: MLX C++ backend as default (ONYX_Q4_BACKEND=mlx|native|auto),
  with per-layer weight cache and the exact (q-8)*scale = q*scale + (-8*scale)
  format bridge. Engine-level medians at m1024: 27 ms -> 4.10 ms (6.6x vs the
  0.1.0 baseline path).
- simdgroup-MMA kernels (mma 32x32, mma64 64x64 four-simdgroup) as the native
  fallback path, ABBA-validated (2.0-3.2x over the scalar tile baseline).
- Decode shapes (m<=32) keep the scalar tile path and beat python-mlx
  (0.473 ms vs 0.726 ms at n=5376).
- New engine-level integration test (onyx_q4_linear_dispatch) plus kernel
  tests; ctest 7/7.
- kernel-agent/ evidence chain: E1-E13 episodes, candidates.jsonl with
  promoted and rejected candidates, fp32-referenced stage validation of the
  fused mlx block (26.9 ms/block, 2.17x over the per-op composition).
- Known limits: decode-qkv gap vs MLX requires an MMA+split-k small-m kernel;
zero-copy mlx bridge and full fused-block engine wiring are follow-ups.

# Task Contract: q4 decode split-k small-M candidate

- Task name: `q4-decode-splitk-v1`
- Objective: Close the small/mid-M native gap by a split-k MMA kernel for
  m<=128 q4 GEMM shapes. Fresh 2026-11-21 matrix (M5 Max 36 GB, this harness,
  gpu_median_ms, group 64) shows the incumbent tile16 loses every small-m
  shape to the in-tree mlx-steel kernel, and the native dispatch (rows>=512
  -> mma64, else MPSGraph) leaves m16/m32/m69 with no native option:
  - m16 n5376 k5376 (decode-linear): tile16 1.093, simd 0.308, steel 0.469
  - m16 n21504 k5376 (decode-qkv): tile16 1.825, mma 2.088, simd 1.396, steel 0.837
  - m32 n21504 k5376: tile16 3.598, simd 1.200, steel 1.095
  - m69 n8192 k5120 (text qproj): tile16 3.264, mma 2.707, steel 0.950
  Matrix JSONs: benchmarks/results/campaign-20261121/.
- Approach: deterministic split-k — parallelize over k slices, per-slice fp32
  partials in scratch, one reduce+convert encoder in the same command buffer
  (no atomics, bit-stable). E8 proved scalar-FMA split-k tops out near
  8 TFLOPS; E9 proved register-side dequant impossible, so weights are
  pre-staged to fp16 in threadgroup memory per slice (mma kernel pattern).
- Baselines: ABBA vs tile16 (incumbent native small-m kernel); bar to beat:
  mlx-steel gpu_median on the same shape. E7/E8 failure map applies (n-capped
  parallelism, scalar-FMA ceiling); E9 applies (no thread_elements dequant).
- Correctness: onyx-metal-bench CPU-reference gate (abs <= 0.05 + 0.005*|v|)
  on all ctest shapes incl. new ragged gates (m17 n33 k30 g30); max_relative
  error must stay <= 1e-2 (fp16-dequant class, same as current mma at qkv).
- Constraints: packing format unchanged; existing kernels untouched;
  new kernel + bench variant registration + scratch reuse; dispatchable as
  one command buffer from `tools/metal_bench.mm --variant splitk16`.
- Validation: full ctest green; ABBA 4 blocks x 10 iterations per shape.
- Promotion criteria: strict gpu_median improvement over tile16 on every
  dispatch-bucket shape AND over mlx-steel on at least the decode-qkv shape,
  with matching correctness. Fallback if not beaten within budget: promote
  per-shape dispatch of the existing mlx-steel kernel for 33<=m<512 and the
  small-m buckets (still 2-3x isolated over tile16), recorded as separate
  candidate. Either outcome wires dispatch via a CPU-reference probe
  (onyx_q4_linear_dispatch pattern); end-to-end runs stay pending human
  validation (no model on this machine).

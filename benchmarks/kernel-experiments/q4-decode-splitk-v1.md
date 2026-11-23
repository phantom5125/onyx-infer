# Small-M Q4 split-K case

This case reconstructs the 2026-11-21 contract and decision from the
[candidate ledger](candidates.jsonl) and
[published run records](../results/campaign-20261121/).
The published base is `f90b0ce` and the retained implementation is
`f8f73f8`. The exact revisions of rejected runtime trials were not recorded
in the ledger.

## Goal and workload

Reduce FP16 Q4 GEMM latency at small or intermediate row counts (`m <= 128`)
by splitting the reduction over K. Hold the existing kernels and packing
format fixed. The candidate uses FP32 scratch partials and a deterministic
reduce-and-convert encoder in the same command buffer.

The benchmark uses FP16 activations `[m,k]`, Q4 weights with logical shape
`[n,k]` (two encoded values per byte, low nibble first), and FP32 scales
`[n,k/group]`. Output is FP16 `[m,n]` with FP32 accumulation.
`onyx-metal-bench` generates the fixture with seed 42.

| Shape `m × n × k` | Group | Workload |
| --- | ---: | --- |
| 16 × 5376 × 5376 | 64 | Decode linear |
| 16 × 21504 × 5376 | 64 | Decode QKV |
| 32 × 21504 × 5376 | 64 | Larger decode QKV |
| 69 × 8192 × 5120 | 64 | Text projection |

Use `tile16` as the isolated operator baseline and the vendored `mlx-steel`
variant as the isolated performance target. At engine level, compare against
MPSGraph, the incumbent for these row counts.

## Correctness and measurement

The operator harness checks sampled outputs against the CPU reference with
`abs(error) <= 0.05 + 0.005 * abs(expected)`. The original case also required
`max_relative_error <= 0.01`, with each relative error computed using
`max(1e-6, abs(expected))` as the denominator. Check the reported field
separately from the harness's `correct` flag. Run CTest, including the
`17 × 33 × 30` ragged shape (group 30) and the multi-row tile case.

The case used an M5 Max with 36 GB of unified memory. Compare
`gpu_median_ms` in four ABBA blocks of ten iterations per run. The case
tested multiple split counts and selected `splitk8` for the paired decode
comparisons.

```bash
cmake -S . -B .build -DCMAKE_BUILD_TYPE=Release -DONYX_ENABLE_MLX=OFF
cmake --build .build
ctest --test-dir .build --output-on-failure

.build/onyx-metal-bench --m 16 --n 21504 --k 5376 --group 64 \
  --iterations 10 --variant splitk8
```

Repeat the final command with `tile16` and `splitk8` in ABBA order for
the paired operator comparison. Run `mlx-steel` on the same shape and
under comparable thermal conditions.

## Acceptance and decision

The original operator target required a win over `tile16` in every intended
dispatch bucket and over `mlx-steel` on decode QKV, with the numerical gates
met. The proposed fallback was an engine experiment dispatching Steel for
small and intermediate row counts.

- **Operator-library retention.** The split-K kernel and reduce entry points
  remained available to the benchmark harness after the campaign reported
  15 passing CTests. The ledger reports about 1.2× on decode QKV versus
  `tile16` in its clean session, with decode-linear inconclusive. Raw data
  from the hotter published QKV rerun shows a win in three of four blocks.
  The isolated split-K kernel trailed Steel in the clean-session comparison.
- **Numerical gate.** All four archived `splitk8` decode-linear samples report
  `correct: true` and `max_relative_error: 0.0772658902`. The harness gate
  passed and the original 1% relative-error gate failed.
  [Decode-linear sample](../results/campaign-20261121/abba-linear-b1-splitk8.json)
- **Runtime adoption rejected.** The ledger records 2.96 ms for cached FP16
  Steel versus 2.55 ms for MPSGraph at `m=16` in the engine probe.
  MPSGraph fuses dequantization and matmul and reuses the compiled graph.
  Split-K stayed available for operator benchmarks. L1–L3 validation with
  a real W4 package was pending at the close of the case.

# Optimization workflow

A fixed workload and its acceptance criteria anchor every optimization. Candidates can change a kernel, plan, or runtime implementation within the case's scope. Changes to model semantics, goldens, stable protocols, and releases have their own review process. [Chinese design notes](zh-CN/LOOP.md)

## 1. Workflow

Freeze an OptimizationCase, build a candidate from a clean checkout, run progressively more expensive correctness checks, collect replayable measurements, and ask an independent evaluator to compare the results. Accepted candidates enter a local registry with an explicit rollback identity. Publishing is a separate decision.

Typical changes include layout, fusion, tiling, residency, and kernel selection. Each candidate includes an exact patch, a comparison baseline, and the evidence needed for the next experiment.

Use the Q4 cases as examples when writing a kernel experiment contract. [Kernel experiments](../benchmarks/kernel-experiments/README.md)

## 2. Persistent records

- **OptimizationCase** pins the case ID, model semantic identity and source revision, inputs, shapes, recipe, seeds and RNG, device, baseline, tolerances, benchmark method, metric, budget, and stop criteria. Distinguish operator probes, 4 MP functional work, and 16 MP stress tests.
- **Candidate** records the case, base commit, patch or artifact SHA-256, scope, build and toolchain, and plan or kernel identities. Keep the change attributable to one main variable and pin its dependencies. Source, inputs, commands, and logs are the reproduction material.
- **Experiment** relates the case and candidate to the original RunRecords and environment. Use the normal submission path. Track cold model and plan startup, warm execution, stage times, memory, transfers, compilation, device conditions, and competing work.
- **Evaluation** records correctness at every level, complete run references, latency distributions, memory and available energy or thermal readings, quality when numerics change, comparison method, and variability. Its decision is `accepted`, `rejected`, or `inconclusive`. Excessive noise and unusable references are reasons to mark a result inconclusive.

## 3. Correctness gates

Run the least expensive checks first. A failed gate ends evaluation before expensive image runs.

| Level | Compare | Examples |
| --- | --- | --- |
| L0 Operator | CPU or reference values, edge shapes, finite values, accumulation, and layout | GEMM, attention, normalization, RoPE, and quantization |
| L1 Stage | Fixed intermediate inputs and stage outputs | Text conditioning, a DiT step, PDD heads, and VAE tiles |
| L2 Replay | Same model, noise, plan, and build | Async scheduling, arena reuse, and atomic operations |
| L3 End-to-end | Final images from fixed requests and resource budgets | Release candidates and numeric changes |

A bitwise result is scoped to its build, device, and plan. Numerical equivalence uses predefined operation- and dtype-specific tolerances. Quantization and larger numerical changes require an image set and quality review. Every candidate that could affect numerical accuracy or visual output must attach viewable baseline and candidate generation PNGs to its pull request, even when replay hashes match. The fixed-request controls and review record follow the [generated-image evidence rules](../CONTRIBUTING.md#generated-image-evidence).

## 4. Measurement

- Record OS, toolchain, metallib, power and thermal state, and other workloads. Keep downloads, builds, and other GPU work outside the run.
- Predeclare warmups, measured runs, time limits, and stopping criteria. Interleave the baseline and candidate to reduce thermal drift and position bias.
- Retain every RunRecord and report medians, tails, and dispersion. Measure cold startup, warm latency, sustained load, and first-time compilation separately.
- Use wall time and stage times for fixed-size H3 images. Enforce peak memory and swap constraints. Mark noisy results inconclusive or use additional measurements allowed by the case.

## 5. Registration and rollback

An accepted candidate passes applicable L0–L3 gates, respects regression and memory bounds, and shows a reproducible improvement above observed noise. A refactor may have a declared performance-neutral gate. Rebuild from pinned artifacts and a clean checkout, attach complete RunRecords, and retain the previous plan or kernel identity for rollback. External and internal candidates use the same process.

## 6. Isolation

Run candidates in a separate worktree and process with bounded caches and no default network, credentials, or write access to production artifacts. Validate data artifacts and establish explicit trust for third-party native code. The controller operates within a fixed experiment budget and lacks authority to alter gates, release secrets, or stable contracts. Replay cases include actual private inputs only after explicit export.

## 7. Initial H3 queue

1. Derive W4 packages from a pinned BF16 source. Record calibration identity, independent acceptance data, exceptions, and actual packed size.
2. Compare fixed-noise tensors and final images to establish a numerical and image-quality baseline.
3. Prove stage residency, conditioning reuse, and memory feasibility on the 36 GB machine.
4. Measure text, individual DiT steps, all six steps, VAE, and image encoding separately.
5. Compare MPP and MSL Q4 operators across shapes and group sizes. Benchmark any format conversion separately.
6. Validate tiled attention and bounded temporary memory at high resolution.
7. Explore dequantization, QKV, RoPE, normalization, and PDD fusion inside the declared numeric contract.
8. Validate tiled FP32 VAE output and overlap behavior.
9. Evaluate W3 or mixed formats as separate model identities if the W4 path misses a quality or memory requirement.

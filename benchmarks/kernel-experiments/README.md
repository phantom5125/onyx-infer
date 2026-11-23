# Kernel experiments

Each experiment records an Onyx `OptimizationCase`, its candidates, and the
evidence behind the decision. Following the Atrex Kernel Agent (AKA) workflow,
freeze the evaluator inputs, bind measurements to exact candidate source,
and have an independent reviewer compare the candidate with the incumbent.
Use the local Onyx build and validation tools to reproduce the cases below.
[AKA workflow](https://github.com/alibaba/atrex-kernel-agent#how-aka-works)
· [Onyx optimization workflow](../../docs/LOOP.md)

| Case | Outcome |
| --- | --- |
| [Q4 simdgroup MMA](q4-mma-v1.md) | Historical experiments on large FP16 Q4 GEMMs, shape dispatch, and MLX integration. |
| [Small-M Q4 split-K](q4-decode-splitk-v1.md) | Split-K retained for operator benchmarks. Engine integration was rejected. |
| [Local single-image campaign](local-single-image-20261123.md) | Weight reads overlapped with GPU compute on the 1024×1024 request, and the rounds that follow. |

[Candidate ledger](candidates.jsonl) preserves trial order, parent candidates,
historical decisions, and the available evidence. Some entries summarize
temporary probes whose source and raw outputs are absent from the archive.

## Writing a case contract

Freeze these entries before the first candidate. Record revisions to the
contract alongside the evidence collected under each revision.

| Entry | What to record |
| --- | --- |
| Goal and scope | Operator and integration point, intended benefit, allowed source changes, and memory or scratch budget. |
| Workload and baseline | Base revision, incumbent at the target integration point, reference implementation, tensor shapes and layouts, dtypes, packing, fixed input fixtures or seed, and edge cases. Include model, request, and execution profile for engine or end-to-end claims. |
| Correctness | Reference command, checked outputs, tolerance formula and thresholds, and applicable L0–L3 gates. Specify how relative error is computed near zero. |
| Measurement | Build and run commands, GPU, OS, toolchain, build or metallib identity, power and thermal conditions, warmups, iterations, and stopping criteria. On one device, alternate baseline and candidate in ABBA blocks. Pin the primary latency metric, comparison statistic, and noise rule. |
| Acceptance | Threshold for improvement, allowed regressions in other shapes and memory, and the evidence needed for operator-library retention or runtime adoption. A runtime change must beat the actual incumbent through the engine. Apply the relevant L0–L3 checks and include image-quality evidence for precision-sensitive changes. |
| Decision and reproduction | Link the exact patch or candidate revision, raw samples, numerical results, and integration evidence. Record whether the candidate was accepted, rejected, or inconclusive, which scope the decision covers, and which checks remain pending. Keep the rollback identity for adopted candidates. |

The benchmark reports correctness and timings. The reviewer applies the
contract's acceptance criteria to those results. See the
[validation guide](../../docs/VALIDATION.md) and
[contribution guide](../../CONTRIBUTING.md) for the Onyx review process.

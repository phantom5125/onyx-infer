# Acceleration roadmap

Acceleration options for the six-evaluation H3 PDD pipeline on Apple Silicon. [Native inference benchmarks](OPTIMIZATION_CAMPAIGN.md) · [Chinese research notes](zh-CN/ACCELERATION.md)

## 1. Sampling and caching

Draft-and-verify decoders amortize verification across multiple tokens in an autoregressive sequence. H3 runs six non-autoregressive evaluations, each with a full DiT forward pass. The image result in SJD2 concerns autoregressive token generation. [MLX draft-model implementation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py) · [SJD2](https://arxiv.org/abs/2510.08994)

Cross-step feature caching can accelerate diffusion models but changes the numerical path. Published results range widely with schedule length. The Delta-DiT evaluation cited in the original review reports 1.6× at 20 steps and 1.12× at four steps. For H3, any cached approximation needs its own numeric identity and quality evidence across prompts. A distilled schedule likewise requires a new recipe and independent validation. [DeepCache](https://arxiv.org/abs/2312.00858) · [TeaCache](https://arxiv.org/abs/2411.19108) · [Learning to Cache](https://arxiv.org/abs/2406.01733) · [ToCa](https://arxiv.org/abs/2410.05317) · [Delta-DiT](https://arxiv.org/abs/2406.01125)

A conditioning cache reuses the exact text encoder output. Its key includes encoder semantics, prompt, references, preprocessing, and numerical settings. The text stage represents 3.83 s of the historical 34.65 s teapot benchmark. The DiT weight LRU reuses the original packed bytes within the current generation. Size and residency are tracked independently. [Profile settings](PROFILES.md)

## 2. Apple Silicon and operator choices

The historical 16.5× operator comparison included MPSGraph graph compilation. With graph reuse in the native pipeline, the `m5-fast` profile reduced median wall time from 22.71 to 20.98 s in a three-run 1024×1024 comparison on an M5 Max. It uses the vendored MLX Q4 kernel through Metal Performance Primitives and trades additional memory and a different numerical path for the measured speedup. [Operator probe](../benchmarks/results/m5-max-36gb-q4-operator-v1.json) · [Native comparison](../benchmarks/results/campaign-20261123/round4-nax-q4-kernel.json)

The engine already caches MPSGraph graphs by shape. Other proposals include encoding more of a sampling step in one submission and measuring the effect of device memory budgets. System-wide memory settings would need their own stability assessment. [MLX lazy evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html) · [TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/)

## 3. Validation and priorities

- **Q4 operators.** Compare MPP and MSL variants with references by shape. In the earlier probe, reading BF16 output as FP16 falsely suggested a decoding error. Corrected real-weight checks confirmed Steel and the production path. At the `m=69` edge the Steel CPU-reference maximum absolute error was 0.00028.
- **Weight cache and conditioning reuse.** Check bitwise replay, measured peak memory, and interleaved latency. The experimental `speed-focus` profile uses a 24 GiB cache. On the 36 GB M5 Max, the default streaming budget with prefetch ran faster than the 10 GiB override on the 1024×1024 teapot request and used less memory. [Tuning and verification](NATIVE_TUNING.md)
- **Command encoding.** Measure operator grouping inside the actual pipeline after verifying lifetime and synchronization.
- **Cross-step features.** Gate these candidates by new numerical identity, multiple prompts, and an end-to-end quality and latency matrix.

The default profile uses MPSGraph. It outperformed the custom BF16 SIMD variant in the full H3 pipeline despite the SIMD variant's isolated kernel gains. [Native inference benchmarks](OPTIMIZATION_CAMPAIGN.md)

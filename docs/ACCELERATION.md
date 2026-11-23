# Acceleration roadmap

The initial review on November 18, 2026 evaluated methods for the six-evaluation H3 PDD pipeline on M5 Max. [Measurements and corrections](OPTIMIZATION_CAMPAIGN.md) · [Chinese research notes](zh-CN/ACCELERATION.md)

## 1. Sampling and caching

Draft-and-verify decoders amortize verification across multiple tokens in an autoregressive sequence. H3 runs six non-autoregressive evaluations, each with a full DiT forward pass. The image result in SJD2 concerns autoregressive token generation. [MLX draft-model implementation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py) · [SJD2](https://arxiv.org/abs/2510.08994)

Cross-step feature caching can accelerate diffusion models but changes the numerical path. Published results range widely with schedule length. The Delta-DiT evaluation cited in the original review reports 1.6× at 20 steps and 1.12× at four steps. For H3, any cached approximation needs its own numeric identity and quality evidence across prompts. A distilled schedule likewise requires a new recipe and independent validation. [DeepCache](https://arxiv.org/abs/2312.00858) · [TeaCache](https://arxiv.org/abs/2411.19108) · [Learning to Cache](https://arxiv.org/abs/2406.01733) · [ToCa](https://arxiv.org/abs/2410.05317) · [Delta-DiT](https://arxiv.org/abs/2406.01125)

A conditioning cache reuses the exact text encoder output. Its key includes encoder semantics, prompt, references, preprocessing, and numerical settings. The text stage represents 3.83 s of the historical 34.65 s teapot benchmark. The DiT weight LRU reuses the original packed bytes within the current generation. Size and residency are tracked independently. [Profile settings](PROFILES.md)

## 2. Apple Silicon and operator choices

The initial standalone MPSGraph Q4 operator measurements showed large gaps against MLX, including 16.5× for the 1024×5376×5376 case. The candidate tracks include Metal Performance Primitives quantized tensor ops, device-tuned MSL, and operator fusion. Kernel probes need real packed weight and edge-tile coverage. Production measurements include graph compilation behavior and caching. [Operator measurements](../benchmarks/results/m5-max-36gb-q4-operator-v1.json) · [Metal features](https://developer.apple.com/metal/whats-new/)

Round 4 of the [local single-image campaign](OPTIMIZATION_CAMPAIGN.md#local-single-image-campaign) settled this track. The 16.5× figure came from an MPSGraph probe that included graph compilation: with compiled graphs the gap is within 9% end to end. MLX's quantized kernel, which runs on the M5 neural accelerators through Metal Performance Primitives, is now compiled into the Onyx metallib and dispatched by the engine behind the `m5-fast` profile. [Round record](../benchmarks/results/campaign-20261123/round4-nax-q4-kernel.json)

The engine already caches MPSGraph graphs by shape. Other proposals include encoding more of a sampling step in one submission and measuring the effect of device memory budgets. System-wide memory settings would need their own stability assessment. [MLX lazy evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html) · [TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/)

## 3. Validation and priorities

- **Q4 operators.** Compare MPP and MSL variants with references by shape. In the earlier probe, reading BF16 output as FP16 falsely suggested a decoding error. Corrected real-weight checks confirmed Steel and the production path. At the `m=69` edge the Steel CPU-reference maximum absolute error was 0.00028.
- **Weight cache and conditioning reuse.** Check bitwise replay, measured peak memory, and interleaved latency. The 24 GiB setting belongs to the experimental `speed-focus` profile. The measured 10 GiB override uses the default MPSGraph kernel and improved the 1024×1024 case. [Tuning and verification](NATIVE_TUNING.md)
- **Command encoding.** Measure operator grouping inside the actual pipeline after verifying lifetime and synchronization.
- **Cross-step features.** Gate these candidates by new numerical identity, multiple prompts, and an end-to-end quality and latency matrix.

The custom BF16 SIMD variant improved isolated Q4 measurements. In the tested H3 pipeline, MPSGraph graph fusion and cached compilation provided better end-to-end wall time. Current defaults use that measured pipeline. [Optimization campaign](OPTIMIZATION_CAMPAIGN.md)

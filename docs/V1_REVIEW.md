# V1 stability review

The v1 contract passed review. The frozen compatibility baseline allows checked additive extensions within the major version. [Compatibility baseline](../schema/v1.compatibility.json)

## Scope

The review covers the six Protobuf families, native H3 text-to-image on the M5 Max with 36 GB, the C++/Objective-C++ Metal runtime, MPSGraph, Onyx Q4 packages, staged streaming, VAE tiling, and the 1024×1024 and 4096×4096 replay and performance baselines. A future binary plugin ABI has its own review. MLX reference results use a separate measurement scope.

## Acceptance evidence

- SHA-256 and format checks covered 53 source files and 2,007 tensors.
- The derived W4 packages measured 21.86 GB for text and 37.47 GB for DiT.
- Native 50-layer text encoding correlated 0.9999987 with the W4 reference. Six-step DiT sampling completed with a stable 6.59 GiB Metal allocation. Native VAE pixels correlated 0.999993 with the reference at the same latent and 1024×1024 size.
- Native 1024×1024 replay passed. The 4096×4096 native generation took 329.61 s at a peak of 2.14 GB.
- At freeze, 17 Python tests, four Metal CTests, and the schema compatibility check passed.

## Optimization evidence at freeze

| Configuration | Commit | 1024×1024 median | Result |
| --- | --- | ---: | --- |
| Native streaming baseline | `b5bf491` | 42.30 s | Functional and replay checks passed |
| Grouped zero-point cache | `4bd6893` | 42.63 s | Bitwise output, slight latency regression |
| Direct reads into Metal buffers | `e3d1b87` | 38.74 s | 9.2% faster and 46.7% lower peak, same SHA-256 on three runs |

On the same teapot request, the community MLX W8A8 path took 23.61 s. Native Onyx took 38.74 s with a different quantization and runtime. The teapot result provides functional and visual evidence for that request. General quality equivalence requires a larger prompt set.

## Records

- [Native stability](../benchmarks/results/native-v1-stability.json)
- [Native baseline](../benchmarks/results/native-baseline-v1.json)
- [Grouped zero-point cache](../benchmarks/results/native-round1-zero-cache.json)
- [Direct weight reads](../benchmarks/results/native-round2-direct-pread.json)
- [Teapot comparison](../benchmarks/results/h3-teapot-1024-comparison-v1.json)
- [Original model layout fixes](../benchmarks/results/h3-original-layout-fixes-v1.json)

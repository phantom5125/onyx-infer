# Onyx Infer architecture

The v1 contracts are stable. The runtime and its plan and kernel optimizations continue to evolve. [Protocol reference](PROTOCOLS.md) · [Optimization workflow](LOOP.md) · [Chinese design notes](zh-CN/ARCHITECTURE.md)

## 1. Product scope

Onyx Infer is a local image-generation engine for Apple Silicon. The initial target is one M5 Max Mac Studio with one GPU, native execution, explicit memory planning, and replayable runs. MiniMax-H3-Image text-to-image is the first supported workload. Image editing with the same checkpoint and independently distributable operators are follow-up work. The initial scope excludes training, distributed execution, hosted inference, and a public binary plugin ABI.

## 2. Model and hardware constraints

| Item | Reference configuration |
| --- | --- |
| H3 DiT | Dense 33.1B single-stream transformer, 50 blocks, hidden width 5376, 56 attention heads, FFN width 14336 |
| Sampling | Six PDD evaluations with a per-step output head and no CFG. The model recipe supplies these settings. |
| Text encoder | Qwen3-VL, hidden states from layer 50, about 66.7 GB of source weights |
| DiT source weights | About 66.28 GB, mainly BF16 with FP32 embedding, timestep, and output components |
| VAE | About 2.34 GB in FP32, 16× spatial downsampling, 24 latent channels, and a 24-layer ViT decoder |
| Output | Direct generation up to about 16 MP. Each DiT image token covers 32×32 pixels. |
| Mac Studio | Mac17,14, 18 CPU cores, 32 GPU cores, 36 GB unified memory, and on-GPU Neural Accelerators |
| Metal limits | 30,150,672,384-byte recommended working set and 22,613,000,192-byte maximum buffer on the reference machine |
| Reference software | macOS 27 and Metal 4 on the benchmark machine |

The original checkpoint is about 135 GB. Even the ideal 33.14 GB estimate for W8 DiT weights exceeds the 28.08 GiB recommended working set before activations. Ideal W4 estimates are 16.57 GB for the DiT and 16.68 GB for the text encoder. The actual packages are 37.47 GB and 21.86 GB respectively because scales, exceptions, and packing add overhead. The DiT streams one block at a time and releases timestep and AdaLN weights after precomputation.

The initial numeric profile uses group-wise W4 weight-only quantization, 16-bit activations, and FP32 matrix and reduction accumulation. Sensitive operations and model exceptions retain higher precision, including the FP32 VAE. A different quantization format creates a derived model identity with its own calibration and quality evidence. The offline BF16 path provides reference results.

At 4096×4096 the image alone adds 16,384 tokens, so attention requires tiled or online reduction. Unified memory can eliminate a physical copy where the backend supports it. The planner still tracks transfers, dependencies, and allocation lifetimes against the actual device limits.

## 3. Design principles

- The semantic model describes computation. The execution plan selects the physical implementation.
- Stable contracts cover requests, artifact identity, tensor semantics, job events, and evidence. Metal objects, thread groups, tile sizes, and caches can evolve independently.
- Protobuf carries control and metadata. Owned artifact and buffer handles carry tensors, weights, and images.
- Requests use the final prompt provided by the caller and expose choices that affect model semantics, network access, storage, and numerical behavior.
- Optimizations require replayable inputs, correctness evidence, raw measurements, and a rollback identity.
- PyTorch and Diffusers serve as offline import and reference tools.

## 4. Execution structure

```text
CLI / local app / agent → Session and Job API → RunEvent and Artifact
                                │
                   Model resolver and compiler
                      │                     │
               Semantic model         Execution plan
                                            │
                              Native runtime and scheduler
                                            │
                                   Internal backend SPI
                                    ├─ Metal and MSL operators
                                    └─ CPU reference
```

The Loop controller submits candidates through the same contracts and reads the same evidence. The operator layer has no dependency on the frontend, importer, Python, or Loop controller.

## 5. Module boundaries

| Module | Responsibilities |
| --- | --- |
| Public API | Validate requests and expose jobs, events, and artifacts |
| Model importer | Read source configuration and SafeTensors and produce model packages |
| Semantic IR | Dataflow, symbolic shapes, versioned numerical rules, and RNG semantics |
| Compiler and planner | Resolve shapes, memory, layouts, stage boundaries, and operator variants |
| Native runtime | Session lifetimes, asynchronous dependencies, cancellation, and trace |
| Backend | Device capabilities, command encoding, memory domains, and completion events |
| `onyx-ops` | Operator implementations with testable numerical contracts |
| Loop controller | Candidate construction, validation, measurement, registration, and rollback |

## 6. Semantic IR and execution plans

Semantic IR persists with the model package. It describes pure dataflow, symbolic dimensions, explicit sampling and stage boundaries, and versioned accumulation, rounding, epsilon, and noise layout. The model recipe describes iteration.

Execution plans capture device and kernel identities, concrete layout, quantization encoding, fusion, tiling, stage residency, arena liveness, transfers, and command dependencies. Each optimized node retains a source mapping to the IR for validation. Cache identity includes model semantics, capabilities, toolchain, operator build, and numeric configuration. Deleting a cached plan changes compilation cost only.

## 7. H3 execution

1. **Prepare.** Validate the resolution and model capabilities, tokenize the final prompt, and generate noise with the declared RNG. An external prompt expansion step records the original prompt, expansion, and model version.
2. **Condition.** Run the W4 text encoder, collect the chosen hidden state, and record a conditioning artifact. The reference plan releases text weights before denoising. A conditioning cache key includes the encoder semantic identity, prompt, reference inputs, preprocessing, and numeric configuration.
3. **Sample.** Stream W4 DiT blocks across the six-step PDD schedule. Include temporary attention memory in the working-set budget.
4. **Decode.** Release DiT resources and decode with the FP32 VAE. Overlapping tiles bound the peak at large resolutions.
5. **Publish.** Write the image artifact and RunRecord. Output encoding has a separate contract from sampling.

## 8. Technology choices

The native backend uses C++20, Apple Clang, and a thin Objective-C++ Metal bridge. CMake builds native executables and precompiled MSL metallibs with the Xcode toolchain. Metal, MPSGraph, and eligible Metal Performance Primitives are backend choices driven by capability checks. System frameworks such as Accelerate and ImageIO can support CPU reference operations and image output. Python 3.12+ and uv support import, reference generation, validation, and benchmarking. Traces, signposts, and Metal counters provide local diagnostics.

MLX can supply a reference or a specific operator backend. Core ML or ANE execution needs measurement against the same workload. ONNX and GGUF can serve as import sources where appropriate. The internal model and plan contracts remain backend-independent.

## 9. Operator interface

`OpDefinition` binds mathematical semantics, inputs, outputs, and tolerances. A `KernelVariant` adds layout, quantization, capability and shape ranges, tile parameters, and implementation SHA. Selection uses measured cost and actual capability. Reference, general Metal, MPP, and device-tuned variants use the same shape and correctness gates. The in-tree C++ SPI can evolve before a future `onyx.abi.v1` is reviewed against real workloads.

## 10. Delivery and acceptance

The initial design milestones covered the v1 contracts and pinned toolchain, the W4 native vertical slice, 16 MP memory and execution, repeatable optimization, and an independently buildable operator library. The baseline pins H3 source revision `df334556f585a596a710cf4f7ce3a567c0738ee5` and Metal compiler `32023.921`. Release evidence includes operation and stage accuracy, replay, image quality, cold and warm latency, peak memory, clean builds, and compatibility checks.

## 11. Working assumptions

The reference hardware is one Apple Silicon machine and one GPU. Capability queries cover other Apple Silicon configurations. BF16 reference data remains separate from the W4 runtime package. Calibration and acceptance datasets are independent. The runtime receives the final prompt from the caller.

## 12. Reference material

- [MiniMax H3 Image](https://github.com/MiniMax-AI/MiniMax-H3-Image)
- [ModelScope checkpoint](https://modelscope.cn/models/MiniMax/MiniMax-H3-Image)
- [Apple Mac Studio announcement](https://www.apple.com/ee/newsroom/2026/08/apple-introduces-mac-studio-with-m5-max-and-m5-ultra/)
- [Apple Metal guides](https://developer.apple.com/guides/metal/)

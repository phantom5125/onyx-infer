# 加速路线与"投机解码"适配性评估

本文记录 Apple Silicon 上六步 H3 PDD 管线的加速方案和验证要求。[原生推理基准](../OPTIMIZATION_CAMPAIGN.md) · [英文路线图](../ACCELERATION.md)

## 1. "投机解码"映射到扩散模型：三条路线的适配性

LLM 投机解码（draft model 起草 + 大模型验证，MLX 官方支持 `--draft-model`，见 [mlx-lm/generate.py](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py)）的成立前提是：存在廉价的逐步起草器、验证一次即可接受多个 token、接受率足够高。H3 的 6-NFE PDD 扩散不具备这些前提：

| 路线 | 代表工作 | 对本项目的适配性 |
| --- | --- | --- |
| draft-verify 投机解码 | MLX/SGLang/HuggingFace 均为 LLM 专用；图像侧仅见 [SJD2（arXiv:2510.08994）](https://arxiv.org/abs/2510.08994)——把去噪嵌入 Jacobi 迭代，服务于**自回归逐 token** 图像模型（数千次前向），且需微调 | **不适用**。H3 是 6 步非自回归模型：没有可起草的 token 链；"验证"本身要花一次完整 DiT 前向，6 步里没有任何可省的步 |
| 跨步特征缓存（扩散投机化的实际对应物） | DeepCache [arXiv:2312.00858](https://arxiv.org/abs/2312.00858)：SD1.5 上 2.3×（CLIP Score 降 0.05）；TeaCache [arXiv:2411.19108](https://arxiv.org/abs/2411.19108)：最高 4.41×（-0.07% VBench）；Learning-to-Cache [arXiv:2406.01733](https://arxiv.org/abs/2406.01733)：缓存步移除 93.68% 计算（全步 46.84%）；ToCa [arXiv:2410.05317](https://arxiv.org/abs/2410.05317)：2.36× | **有条件适用，收益缩水**。公开数据一致显示步数越少收益越小：Δ-DiT [arXiv:2406.01125](https://arxiv.org/abs/2406.01125) 在 20 步模型 1.6×，4 步模型仅 1.12×；FLUX.1-schnell（4 步）上最好 2.46×。H3 是 6 步，预期收益远低于多步模型，且**改变数值路径**——必须注册为新 numeric profile 身份并走完整质量门（LOOP.md §3），不能作为计划优化混入 |
| 步数蒸馏 | ADD/SDXL-Turbo [arXiv:2311.17042](https://arxiv.org/abs/2311.17042)、LCM [arXiv:2310.04378](https://arxiv.org/abs/2310.04378)：1-4 步达标 | **属于新模型身份**，不是引擎优化。H3 recipe 是模型制品；改步数 = 新 recipe + 完整质量证据，与档位系统无关 |

## 2. Mac（M5 Max）特性与本机瓶颈的对齐

早期 Q4 算子探针测得 16.5× 的差距，其中 MPSGraph 耗时包含图编译。完整原生管线会复用已编译的图。在 M5 Max 的 1024×1024 茶壶对照中，`m5-fast` 使用 Metal Performance Primitives 调用随项目提供的 MLX Q4 内核，将三次运行的耗时中位数从 22.71 秒降至 20.98 秒。该内核使用 BF16 scale 和 bias，增加内存占用，也会改变输出数值。[算子探针](../../benchmarks/results/m5-max-36gb-q4-operator-v1.json) · [端到端对照](../../benchmarks/results/campaign-20261123/round4-nax-q4-kernel.json)

| Mac 特性 | 对应实现 | 状态 |
| --- | --- | --- |
| M5 神经网络加速器与 MPP 量化张量 | 通过 `m5-fast` 在 Q4 线性层使用 MLX 量化内核 | 已提供可选档位 |
| 统一内存与页缓存 | 在单次生成的多个采样步之间复用 Q4 权重 | 已实现 DiT LRU 权重缓存。实测条件见[调优指南](../NATIVE_TUNING.md) |
| `iogpu.wired_limit_mb`（[MLX README](https://raw.githubusercontent.com/ml-explore/mlx-lm/main/README.md) 记载的 wired memory 手段） | 可提高 GPU 可驻留预算；属系统级调优，需单独验证稳定性后才能进基准口径 | 备选，暂不动基准口径 |
| MLX lazy evaluation / `mx.compile`（[官方文档](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html)） | MPSGraph 等价物我们已有（按 shape 键控的图缓存 + metallib 预编译） | 已具备 |
| TensorRT-LLM CUDA Graphs（[docs](https://nvidia.github.io/TensorRT-LLM/)） | MPSGraph 按需建图已是同思路；进阶是步内多算子合并编码，属 kernel 层候选 | P2 |

## 3. 相关实现与研究

- **SGLang / RadixAttention**（[blog](https://www.lmsys.org/blog/2024-01-17-sglang/)、[arXiv:2312.07104](https://arxiv.org/abs/2312.07104)）：radix tree 前缀缓存带来最高 5× 吞吐。迁移到扩散管线 = **conditioning cache**：以"文本编码器语义哈希 + prompt + 预处理"为键缓存文本编码输出（占端到端 ~11%：3.83 s / 34.65 s），同 prompt 重复生成直接跳过文本塔。这是缓存路线里唯一**数值不变**（复用同一 artifact）的一项，性价比最高。
- **TensorRT-LLM**（[docs](https://nvidia.github.io/TensorRT-LLM/)）：官方把 **GEMM/attention 量化 + graph replay** 作为视觉生成首选——与我们的 W4 + MPSGraph 图缓存同构；他们的经验确认"先修 kernel 效率，再玩调度技巧"。
- **Apple MLX 团队**：lazy evaluation、`mx.compile` 融合、wired memory 调优（URL 见 §2）。
- **缓存研究线（DeepCache/TeaCache/L2C/ToCa/Δ-DiT）**：跨步缓存在多步模型 2.3-4.4×，**在 4 步模型缩水到 1.12-2.5×**；对 6-NFE 的 H3 属于"有质量风险的候选"，必须新 numeric profile 身份 + 多 prompt 质量评估后才可能启用。

## 4. 优先级队列（与门禁对齐）

| 优先级 | 项 | 数值影响 | 门禁路径 |
| --- | --- | --- | --- |
| Q4 神经网络加速器 | `m5-fast` 使用 vendored MLX 内核 `affine_qmm_t_nax`。可用 `onyx-q4-path-compare` 对比真实权重上的延迟和误差。 | BF16 scale 和 bias 带来数值差异 | `onyx_q4_linear_nax` CTest、FP64 参考对照与端到端交错测量 |
| Q4 探针验证 | 早期探针将 BF16 输出按 FP16 读回，造成参考值偏差。修正后，真实量化权重与参考对照通过。 | 在 `m=69` 边界形状中，Steel 与 CPU 参考的最大绝对误差为 0.00028 | `onyx_linear_steel_probe` CTest |
| P0 进展（2026-11-18） | ① MLX steel qmm 内核（社区最优，MIT）vendored 并在我们 bench 内跑通：5.27/20.74/1.19 ms（三个真实 shape），成为新基线 variant `mlx-steel`（数据：`benchmarks/results/m5-max-36gb-q4-kernel-variants-v1.json`）；② 自研蒸馏版 `simd` 9.64/38.52/2.50 ms（较原 tile16 快 2.8×，尚未达 steel 水平）；③ steel 与 MLX 独立参照仍有 ~3× 差距（Shared 存储模式 + 计时口径），private buffer + blit 上传是下一步 | — | 两个新内核均有 CTest 正确性门（onyx_q4_gemm_simd / onyx_q4_gemm_steel） |
| P1 | DiT block LRU 缓存（已实现） | 无（bitwise） | `must_be_fastest_within` + 峰值内存 ≤ 28.08 GiB |
| P1 | **MLX steel affine_qmm_t 接入运行时**（q4_kernel=steel） | 数值等价（fp16 舍入内，非逐位） | `numeric_equal_to`（linear_steel_probe CTest 全对拍）+ `must_be_fastest_within` |
| P1 实测 | DiT qkv_proj 形状 (1024×8192×5120)：tile16 38.4ms → simd 13.9ms → **steel 7.6ms**（**5.1× vs 基线**）。linear_q4_simd 运行时含 GPU bf16→fp16 转换内核（消除 host CPU 串行转换），m=1024 GPU 17.1ms。 |
| P1 运行时集成 | linear_q4_simd + GPU bf16→fp16 转换内核 + 三路 dispatch（simd/steel/mpsgraph）已集成并通过 CTest/pytest 验证。profile knob q4_kernel=simd 启用 steel/simd 快速路径。 | | |
| P1 DiT 全链路 | 3 个 Q4 GEMM 层（qkv+out_proj+fc1）m=1024 总耗时：tile16 203.9ms → simd **38.5ms**（**5.3×**）→ steel 40.0ms（5.1×）。**simd 快于 steel 4%**。DiT 单步节省 165ms × 6 NFE = **每张图节省 ~1.0 秒**（仅 Q4 层）。数据：kernel-variants-v1.json DiT 全链路对比。 |
| P1 运行时集成完成 | linear_q4_simd + linear_steel + mpsgraph 三路径全部通过真实包 CPU 全对拍（simd/steel 0.02017 / mpsgraph 0.01819）。修复：linear_q4_simd 的 m/k 绑定互换（内核契约检查静默拦截 → 全零）。simd 在全部形状（含 m=69 text tower 0.71ms = 4.3×）均为最快，q4_kernel=simd 无需按 m 分派。检查点提交 fa910f7。 |
| P0 生产验证 | 端到端管线实测（W4 全量包 37.5GB 在位）：基线 36.8s/图。快路径（simd/steel）DiT 阶段 37-41s **慢于**基线 26s——production MPSGraph 图级融合在真实管线中更快，bench 级 5.3× 不translate到端到端。快内核保留为基准实现与验证基础设施，生产路径维持 mpsgraph。bf16 变体根治 fp16 溢出（VAE 通过 ✓）。 |
| P0 质量验证 | bf16 路径 latent 与基线**相关系数 0.9973**、均值差 0.53%（值域归一）、完全有限、范围吻合（±4）——统计上同等有效的生成。更正：此前"基线 377 非有限值"为 uint16 视角读 F32 的测量伪影，模型稳定。像素差异 mean 5.3/255 为混沌扩散对数值扰动的预期放大。 |
| P0 感知门 | **SSIM 0.9601**（阈值 0.85，pass）——感知近似相同。工具 tools/ssim_check.py；门语义正式从数值等价转向感知质量指标。 |
| P0 验收流 | tools/profile_quality_gate.py：候选 profile 生成 → SSIM 感知门 → verdict JSON，一条命令完成（--make-reference 生成基线）。热稳定性实测：两路径轻度热衰减（bf16 +2.6% / mps +6.8%），内核差距 1.35× 非热效应主导。 |
| P0 按类分解 | ONYX_SIMD_SKIP 按类回退实验：qkv/fc1/fc2 各慢 ~16-20ms/调用，out_proj 持平。**simd 内核 in-context 与 bench 一致 ✓**；production MPSGraph qkv 仅 ~0.4-2ms（bench 的 106ms 被图编译开销污染，production 缓存编译图）。 | | |
| P0 金标准对拍 | MLX 参照实现（original_w4 读同一 W4 包）vs native production：latent **相关系数 0.9962**（同 embedding、同 seed→同 numpy noise）。三角闭合：MLX 参照 ↔ native mpsgraph（0.9962）↔ native bf16 快路径（0.9973）——三方互洽，生产正确性金标准确认。 | | |
| 重建链路 | W4 包全量重建（已验证 package 步骤）：① `python -m onyx quantize --source artifacts/models/h3-image-source --component transformer --all-linear --output <dir>/transformer.onxpack` ② 同法 text_encoder ③ `python -m onyx package-w4 --source artifacts/models/h3-image-source --original-manifest artifacts/models/h3-w4-meta/manifest.pb --output <dir>`（wrap+link vae/tokenizer/pdd）。子集量化验证用 `--tensor <name>` 替代 --all-linear。 | | |
| P1 注记 | Apple Silicon 统一内存上 Shared/Private 存储模式无 GPU 计算性能差异（实测同形状一致）。MLX 独立参照与 bench 的 ~3× 差距源于流水线调度差异，非存储模式。优化聚焦内核本体。 |
| P1 注记 2 | simd（9.6ms）与 steel（5.27ms）同架构、同数据同形状均正确。差距归因：① simd 的 Ws [k][n] 布局 dequant 写为跨行访问（stride LD=40）而 MLX 用 [n][k] 连续写；② simd 的 mma 操作数用标量读 + thread_elements 重解释（MLX 也用此路径）✓ 已对齐；③ MLX 的 BlockLoader 用 8 元素向量结构拷贝、simd 用 AVec8 ✓ 已对齐；④ 两内核 barrier 数量相同（2/tile）。剩余差距需 GPU profiler（Xcode Metal system trace）定位。 | | |
| P1 | conditioning cache（SGLang 式前缀缓存） | 无（复用 artifact） | 缓存键契约 + replay 语义 + 交错矩阵 |
| P2 | 步内多算子合并编码（CUDA-graph 式） | 无 | 同 P1 |
| P3 | 跨步特征缓存（DeepCache/TeaCache 类） | **有**——新 numeric profile 身份 | 新量化/数值身份评审 + 多 prompt 质量门；预期收益按 6-NFE 折减后评估，允许 inconclusive |
| 不采 | draft-verify 投机解码 | 结构上不适用（§1） | — |
| 不采 | 6 步 → 更少步数的调度技巧 | recipe 属模型制品 | 新 recipe 身份独立评审，不经档位系统 |

## 5. 后续验证

精确的文本条件缓存需要验证缓存键和回放一致性。跨步特征缓存会改变数值路径，需要单独的数值档位，以及覆盖多个提示词的质量和延迟对照。对于 Q4 内核和权重缓存，使用真实权重测量端到端耗时与峰值内存。[验证流程](../VALIDATION.md)

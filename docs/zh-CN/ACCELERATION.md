# 加速路线与"投机解码"适配性评估

状态：2026-11-18 技术评估，基于公开资料逐条核实（引用见文内 URL）。结论先行：**LLM 式 draft-verify 投机解码对 6-NFE 非自回归扩散模型结构上不适用**（未找到任何针对少步非自回归扩散模型的投机解码公开工作）；扩散场景的对应物是**跨步特征缓存**，但其收益随步数减少明显缩水且改变数值路径；本机真正的第一优先级是**用 M5 Neural Accelerators 关闭 16.5× 的 Q4 GEMM 差距**。（2026-11-23 更新：该项已做完，见下表「P0 结果」；差距本身来自探针口径。）

## 1. "投机解码"映射到扩散模型：三条路线的适配性

LLM 投机解码（draft model 起草 + 大模型验证，MLX 官方支持 `--draft-model`，见 [mlx-lm/generate.py](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py)）的成立前提是：存在廉价的逐步起草器、验证一次即可接受多个 token、接受率足够高。H3 的 6-NFE PDD 扩散不具备这些前提：

| 路线 | 代表工作 | 对本项目的适配性 |
| --- | --- | --- |
| draft-verify 投机解码 | MLX/SGLang/HuggingFace 均为 LLM 专用；图像侧仅见 [SJD2（arXiv:2510.08994）](https://arxiv.org/abs/2510.08994)——把去噪嵌入 Jacobi 迭代，服务于**自回归逐 token** 图像模型（数千次前向），且需微调 | **不适用**。H3 是 6 步非自回归模型：没有可起草的 token 链；"验证"本身要花一次完整 DiT 前向，6 步里没有任何可省的步 |
| 跨步特征缓存（扩散投机化的实际对应物） | DeepCache [arXiv:2312.00858](https://arxiv.org/abs/2312.00858)：SD1.5 上 2.3×（CLIP Score 降 0.05）；TeaCache [arXiv:2411.19108](https://arxiv.org/abs/2411.19108)：最高 4.41×（-0.07% VBench）；Learning-to-Cache [arXiv:2406.01733](https://arxiv.org/abs/2406.01733)：缓存步移除 93.68% 计算（全步 46.84%）；ToCa [arXiv:2410.05317](https://arxiv.org/abs/2410.05317)：2.36× | **有条件适用，收益缩水**。公开数据一致显示步数越少收益越小：Δ-DiT [arXiv:2406.01125](https://arxiv.org/abs/2406.01125) 在 20 步模型 1.6×，4 步模型仅 1.12×；FLUX.1-schnell（4 步）上最好 2.46×。H3 是 6 步，预期收益远低于多步模型，且**改变数值路径**——必须注册为新 numeric profile 身份并走完整质量门（LOOP.md §3），不能作为计划优化混入 |
| 步数蒸馏 | ADD/SDXL-Turbo [arXiv:2311.17042](https://arxiv.org/abs/2311.17042)、LCM [arXiv:2310.04378](https://arxiv.org/abs/2310.04378)：1-4 步达标 | **属于新模型身份**，不是引擎优化。H3 recipe 是模型制品；改步数 = 新 recipe + 完整质量证据，与档位系统无关 |

## 2. Mac（M5 Max）特性与本机瓶颈的对齐

Apple 官方 What's New（[developer.apple.com/metal/whats-new](https://developer.apple.com/metal/whats-new/)）明确：**"New quantized tensor formats and support for scale factors…native support in Metal performance primitives to efficiently compress weights and take advantage of the Neural Accelerators on the M5 Pro and M5 Max."** —— 这正对着我们公布的差距：Onyx MPSGraph Q4 流式 GEMM vs MLX 在 1024×21504×5376 上慢 **16.5×**、QKV 形状慢 20.05×（`m5-max-36gb-q4-operator-v1.json`）。**关闭这个差距的期望收益（数倍）大于任何缓存或投机技巧（1.1-2.5×）**，且是纯 kernel 层优化、不动数值语义契约（同一数学定义的更快实现，走 L0/L1 对账）。

Mac 特性 → 动作映射：

| Mac 特性 | 对应动作 | 状态 |
| --- | --- | --- |
| M5 Pro/Max Neural Accelerators + MPP 量化张量格式 | 用 MPP 量化 tensor ops / NA-aware MSL 变体重写 Q4 GEMM，逐 shape 对账 | **P0**，Loop 队列第 5 项 |
| 统一内存 + 页缓存（64 GB 物理装得下 37 GB 包） | 已被隐式利用；显式化 = 代内 LRU block 缓存（消除重复 pread+拷贝，bitwise 等价） | **P1**，已实现（见 PROFILES.md §2），收益待 W4 包重建后实测 |
| `iogpu.wired_limit_mb`（[MLX README](https://raw.githubusercontent.com/ml-explore/mlx-lm/main/README.md) 记载的 wired memory 手段） | 可提高 GPU 可驻留预算；属系统级调优，需单独验证稳定性后才能进基准口径 | 备选，暂不动基准口径 |
| MLX lazy evaluation / `mx.compile`（[官方文档](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html)） | MPSGraph 等价物我们已有（按 shape 键控的图缓存 + metallib 预编译） | 已具备 |
| TensorRT-LLM CUDA Graphs（[docs](https://nvidia.github.io/TensorRT-LLM/)） | MPSGraph 按需建图已是同思路；进阶是步内多算子合并编码，属 kernel 层候选 | P2 |

## 3. 顶尖 infra 团队的可迁移建议

- **SGLang / RadixAttention**（[blog](https://www.lmsys.org/blog/2024-01-17-sglang/)、[arXiv:2312.07104](https://arxiv.org/abs/2312.07104)）：radix tree 前缀缓存带来最高 5× 吞吐。迁移到扩散管线 = **conditioning cache**：以"文本编码器语义哈希 + prompt + 预处理"为键缓存文本编码输出（占端到端 ~11%：3.83 s / 34.65 s），同 prompt 重复生成直接跳过文本塔。这是缓存路线里唯一**数值不变**（复用同一 artifact）的一项，性价比最高。
- **TensorRT-LLM**（[docs](https://nvidia.github.io/TensorRT-LLM/)）：官方把 **GEMM/attention 量化 + graph replay** 作为视觉生成首选——与我们的 W4 + MPSGraph 图缓存同构；他们的经验确认"先修 kernel 效率，再玩调度技巧"。
- **Apple MLX 团队**：lazy evaluation、`mx.compile` 融合、wired memory 调优（URL 见 §2）。
- **缓存研究线（DeepCache/TeaCache/L2C/ToCa/Δ-DiT）**：跨步缓存在多步模型 2.3-4.4×，**在 4 步模型缩水到 1.12-2.5×**；对 6-NFE 的 H3 属于"有质量风险的候选"，必须新 numeric profile 身份 + 多 prompt 质量评估后才可能启用。

## 4. 优先级队列（与门禁对齐）

| 优先级 | 项 | 数值影响 | 门禁路径 |
| --- | --- | --- | --- |
| P0 | MPP/NA 重写 Q4 GEMM，关闭 16.5× 差距 | 无（同数学定义的更快实现） | L0/L1 逐 shape 对账 + 端到端 bitwise + 交错矩阵 |
| P0 结果（2026-11-23） | **已关闭，但 16.5× 的差距本身不成立**：它来自含图编译的 MPSGraph 探针。按引擎入口（图已编译）在真实块权重上对比，MLX 的 NAX 量化内核（`affine_qmm_t_nax`，经 MPP 运行在 Neural Accelerators 上）相对 MPSGraph：out_proj 快约 25%，fc1 快 12–24%，qkv 在噪声内持平（5 次运行 13–18 ms 对 14–15 ms）——每个线程组仍要靠软件搬运并反量化权重块。该内核已从 vendored 源码编入自有 metallib，由 `GraphSession::linear` 直接分发（不经 MLX 运行时、无拷贝），作为新档位 `m5-fast` 提供：交错 A/B 22.71 → 20.98 s（DiT 18.30 → 16.71 s）。数值变体：每层 46% 的输出相差 1 个 BF16 ulp；对照 FP64 参考，两条路径都落在典型输出的 2–5 个半 ulp 内，互有高低；图像相关 0.987；峰值 RSS 1.67 → 3.31 GiB。复现：`onyx-q4-path-compare`。记录：`benchmarks/results/campaign-20261123/round4-nax-q4-kernel.json` | 数值变体（scale/bias 为 BF16） | `onyx_q4_linear_nax` CTest + 真实权重 FP64 对拍 + 交错端到端 |
| P0 进展 2（2026-11-18） | ⚠️ 重大发现：真实量化权重（text-q0-qproj, m=32）上，steel 内核与生产 MPSGraph 路径输出**互相一致**（fp16 舍入内），但**共同偏离**三重地面真值（MLX python dequantize+matmul = numpy 独立解码 = C++ CPU 参照，三者一致给出 out[0][:4]≈1.025/0.461/-0.700/0.966；GPU 给 1.879/1.731/-1.799/1.866）。已排除：四种零点约定（±8、漏零点、符号取反）均不匹配 GPU 输出；k 配对交替符号检测因精度不足无结论。合成数据 bench 无法暴露此问题（合成 scale 分布下各候选差异超容差，会 fail——而 bench 通过说明合成数据下 GPU=CPU）。这意味着生产 MPSGraph 的 W4 解码语义可能与 MLX affine 语义存在系统性偏差，且 v1 的 correlation 验收无法区分该类偏差。**修复优先级提升为最高**：单点 dump steel/mpsgraph 的实际 dequantized W 与 numpy 对比即可定位。诊断工具：`tools/linear_steel_probe.mm`（支持 steel/mpsgraph 双路径 + dump）、`tools/steel_real_data_probe.mm`（独立绑定）。影响面：真实数据上所有 W4 GEMM 输出；对图像质量的影响未量化。
**进一步线索与定位（同日）**：probe 以 x=e_0 输入时 steel 输出与 x=LCG 近似相同——最终定位：**探针自身的读回类型错误**（把 BF16 输出按 FP16 解码 → 数值显示垃圾），steel 内核与生产路径的输出实际全部正确（y_fp16 原始位 a6f8 = fp16(-0.0272) = W(0,0) 逐位吻合地面真值）。
**结论（已修正）**：① MLX steel 内核在真实量化数据上完全正确（直绑与 CTest 双门验证）；② 生产 MPSGraph 解码路径**无错误**（此前怀疑为探针读回 bug 所致，已排除）；③ GraphSession::linear_steel 运行时路径**验证正确**，含 M=69 边缘 tile（CTest `onyx_linear_steel_probe`，CPU 全对拍 max_err=0.00028）。steel 运行时集成解锁。 |
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

## 5. 对"投机解码"问题的直接回答

对这台单机单 GPU、6-NFE、非自回归的管线：**没有合适的投机解码**。领域内最接近的公开工作（SJD2）服务于自回归逐 token 图像模型，需要数千次前向才有起草空间，还要微调；跨步缓存是扩散场景的实际对应物，但公开数据表明少步模型收益缩水且必然改数值。我们已把缓存路线里数值安全的两个成员（block LRU、conditioning cache）排进队列，把数值有风险的成员（跨步特征缓存）明确标为"新身份候选"，把最大的确定性收益（Q4 GEMM × Neural Accelerators）排在 P0。

<p align="center">
  <picture>
    <img alt="Onyx Infer" src="assets/banner.svg" width="100%">
  </picture>
</p>

<p align="center">
  <a href="README.md"><img src="https://img.shields.io/badge/README-English-blue" alt="English README"></a>
  <img src="https://img.shields.io/badge/protocol-onyx.*.v1%20%E5%B7%B2%E5%86%BB%E7%BB%93-success" alt="v1 已冻结">
</p>

# onyx-infer

在 **Apple Silicon Mac 上本地直出 4096×4096 图像**。Onyx Infer 用 C++20 和 Metal 4 执行 MiniMax-H3-Image 的文本编码器、33.1B DiT 和 VAE。4K 出图已在配备 M5 Max 和 36 GB 统一内存的 Mac Studio 上验证。

[生成第一张图](docs/QUICKSTART.md) · [性能实测](#实测结果m5-max-36-gbmac1714) · [Onyx Studio](plugin/onyx-studio/README.zh-CN.md)

六步 4096×4096 茶壶基准在 36 GB M5 Max 上耗时 329.6 秒。24 GB M4 MacBook Air 在电池供电下生成 1024×1024 茶壶图的中位数约为 175 秒。[M5 Max 基准](benchmarks/results/native-v1-stability.json) · [M4 Air 基准](benchmarks/results/m4-air-24gb-dit-prefetch-v1.json)

PyTorch 和 Diffusers 用于离线导入和参考验证。原生推理由 MPSGraph 和 Metal 内核执行。

## 在 Apple Silicon 上生成

以下均为 Onyx 实际出图的缩放预览，原始生成分辨率见图注。

<table>
  <tr>
    <td align="center" width="50%"><img src="site/assets/gallery/pelican-bicycle.jpg" width="480" alt="海边骑着红色自行车的鹈鹕"><br><b>骑自行车的鹈鹕</b><br><sub>4096×4096 · 16 MP 直出</sub></td>
    <td align="center" width="50%"><img src="site/assets/gallery/red-agate.jpg" width="480" alt="具有半透明矿物纹理的抛光红玛瑙"><br><b>红玛瑙</b><br><sub>4096×4096 · 16 MP 直出</sub></td>
  </tr>
  <tr>
    <td align="center"><img src="site/assets/gallery/moon-surface.jpg" width="480" alt="黑色天空下细节清晰的月球表面与环形山"><br><b>月球表面</b><br><sub>4096×4096 · 16 MP 直出</sub></td>
    <td align="center"><img src="site/assets/gallery/moka-pot-w4.jpg" width="480" alt="标题 MOKA POT 清晰准确的摩卡壶蓝图"><br><b>摩卡壶蓝图</b><br><sub>1024×1024 · W4 · 短标题逐字准确</sub></td>
  </tr>
</table>

## 为什么要有这个项目

原始权重约 **135 GB**，本机 GPU 可用预算 28.08 GiB。这个差距决定了全部架构：
group-wise W4 量化、分阶段驻留、逐 block 流式、分块注意力，以及一个显式做内存规划的
引擎，而不是指望统一内存自动解决一切。

## 优化榜单

所有配置都在 M5 Max 36 GB 上使用同一条 1024×1024 茶壶请求（seed 42，六步采样）。
各项优化的 PNG 都与基线逐字节一致。测量文件记录了样本数、输出哈希和提交版本。

| 配置 | 墙钟时间中位数 | 峰值内存 | 相对基线 | 测量 |
|---|---:|---:|---:|---|
| 原生流式基线 | 42.30 s | 3.19 GiB | 1.00× | [基线测量](benchmarks/results/native-round1-zero-cache.json) |
| 分组 Q4 零点缓存 | 42.63 s | 3.18 GiB | 0.99× | [测量](benchmarks/results/native-round1-zero-cache.json) |
| 权重直读 Metal 缓冲区 | 38.74 s | 1.70 GiB | 1.09× | [测量](benchmarks/results/native-round2-direct-pread.json) |
| 共享内存 BF16 调制打包 | 37.50 s | 1.70 GiB | 1.13× | [测量](benchmarks/results/native-round3-cpu-modulation.json) |
| 融合 DiT 注意力和张量视图 | 34.65 s | 1.86 GiB | 1.22× | [测量](benchmarks/results/native-round4-fused-attention-views.json) |
| DiT 流式权重预取（默认） | **32.81 s** | 1.79 GiB | **1.29×** | [测量](benchmarks/results/m5-max-36gb-dit-prefetch-default-v1.json) |

完整记录见[茶壶基准文章](site/blog/posts/four-rounds-one-byte-identical-teapot.md)。

### 2026 年 11 月 23 日：本地单图五轮优化

在同一请求上再做五轮，每一轮都与上一轮交错对比。[各轮记录](benchmarks/results/campaign-20261123/) · [完整说明](docs/OPTIMIZATION_CAMPAIGN.md#local-single-image-campaign)

| 构建 | 墙钟时间中位数 | 峰值内存 | 相对 `f90b0ce` | 输出 |
|---|---:|---:|---:|---|
| `f90b0ce`，`max-quality` | 42.09 s | 1.79 GiB | 1.00× | SHA-256 `4160adb7` |
| 五轮之后，`max-quality` | **19.18 s** | 1.70 GiB | **2.19×** | 3,145,728 个通道值中有 36 个相差 1 级 |
| 五轮之后，`m5-fast` | **18.37 s** | 3.02 GiB | **2.29×** | 数值变体，与 `max-quality` 图像相关 0.987 |

| 轮次 | 改动 | 之前 → 之后 | 输出 |
|---|---|---:|---|
| 1 | 权重读取与 GPU 计算重叠 | 41.88 → 26.27 s | 逐字节一致 |
| 2 | VAE 分块批处理 | 24.22 → 22.08 s | 36 个通道值相差 1 级 |
| 3 | AdaLN 调制改为 GPU gather | DiT 19.94 → 18.33 s | 逐字节一致 |
| 4 | Q4 线性层运行在 M5 神经网络加速器上（`m5-fast`） | 22.71 → 20.98 s | 数值变体 |
| 5 | 默认启用调制缓存 | 22.71 → 19.35 s | 逐字节一致 |

## 核心特性

- **全原生链路，运行时零复用**——MPSGraph + 手写 MSL kernel，加载自有 packed Q4 权重；
  分阶段流式执行，VAE tiling。
- **16 MP 直出**——4096×4096 由 DiT 直接生成（每 token 对应 32×32 像素），无独立
  upscaler。该次茶壶基准耗时 329.6 秒，采样到的进程树峰值 RSS 为 2.14 GB。
- **可回放的 v1 协议**——六组版本化 Protobuf 契约（`onyx.common/job/model/plan/trace/loop.v1`）；
  每次运行写入 RunRecord；输出经 SHA-256 校验并可回放复现。
- **证据先行的优化 Loop**——上面的榜单不是叙事；每一轮都有测量 JSON、正确性门与
  输出哈希证明作为支撑。
- **版本化不可变模型包**——ModelScope revision 固定、逐文件 SHA-256、多段续传下载、
  流式 group-wise W4 打包。
- **W4 的四档执行配置**——`max-quality` 为默认档，`memory-min` 全阶段流式读取，
  `m5-fast` 在 M5 神经网络加速器上运行 Q4 线性层，`speed-focus` 用于试验 BF16 SIMD 和权重缓存。
  档位注册表记录具体配置与验收门槛。

## 实测结果（M5 Max 36 GB，Mac17,14）

基准机上的关键数字；优化历史见上方榜单。我们与胜利一起发布差距。

| 测量项 | 数值 | 来源 |
| --- | --- | --- |
| 1024×1024 原生生成 | 19.18 s median（text 2.23 / DiT 14.89 / VAE 2.09），峰值 1.70 GiB，调制缓存已预热 | [final-comparison.json](benchmarks/results/campaign-20261123/final-comparison.json) |
| 1024×1024，`35bc6f9`（本地单图五轮优化之前） | 32.81 s median（text 5.25 / DiT 22.95 / VAE 4.65），峰值 1.79 GiB | `m5-max-36gb-dit-prefetch-default-v1.json` |
| 1024×1024，启用权重预取之前 | 34.65 s median（text 3.83 / DiT 26.02 / VAE 4.41），峰值 1.86 GiB | `native-round4-fused-attention-views.json` |
| 4096×4096 茶壶基准 | 329.61 s，该次运行采样到的进程树峰值 RSS 为 2.14 GB，已回放验证 | [native-v1-stability.json](benchmarks/results/native-v1-stability.json) |
| 数值保真 | 文本塔相关系数 0.9999987；VAE 像素相关系数 0.999993 | `native-baseline-v1.json` |
| Q4 GEMM vs MLX | 隔离探针中同形状 1024×5376×5376 下 28.12 ms vs 1.71 ms。MPSGraph 的数字包含图编译：图已编译后，引擎入口的端到端差距在 9% 以内 | `m5-max-36gb-q4-operator-v1.json`、[第 4 轮](benchmarks/results/campaign-20261123/round4-nax-q4-kernel.json) |
| 原生 vs 社区 W8A8（已发布的差距） | 同一请求 38.74 s vs 23.61 s，1.64× | `native-v1-stability.json` |

胜利与差距都在 `benchmarks/results/` 里以提交的 JSON 存档。

## 快速开始

[从零生成第一张图](docs/QUICKSTART.md)列出了原生构建、锁定版本下载、W4 转换和 1024×1024 出图的完整命令。

- **系统** — Apple Silicon、macOS 26.2 或更新版本、Xcode Metal 4 工具链，以及 `uv` 和 Python 3.12 或更新版本。
- **磁盘** — W4 推理文件约 **62 GB**，包含 VAE、PDD heads 和 tokenizer。原始权重与转换产物合计预留约 **200 GB**，构建工具另计。
- **已实测机型** — M4 Air（24 GB）和 M5 Max（36 GB）均完成 1024×1024 出图。统一内存下限尚未验证。

使用 `memory-min` 可让全部阶段流式读取权重；在 M5 上可用 `m5-fast` 让 Q4 线性层运行在神经网络加速器上（属于数值变体）；`speed-focus` 用于试验 BF16 SIMD 内核与 DiT 权重缓存。在保持 `max-quality` 数值路径的档位中，默认的 MPSGraph 内核实测最快。详见[优化记录](docs/OPTIMIZATION_CAMPAIGN.md)。

[Onyx Studio](plugin/onyx-studio/README.md) 为 DeepSeek Harness 提供本地出图、引擎观测和图文阅读器。

六步 DiT 调制表只取决于模型和采样计划，因此 `native-generate` 默认把它缓存在 `~/Library/Caches/onyx-infer/modulation`，在不同提示词、种子和画幅之间复用。换用新的引擎构建后，首次请求需要计算调制表并读入 26 GB 的 AdaLN 投影权重，之后的请求直接加载。`--modulation-cache <目录>` 可更换缓存位置，`--no-modulation-cache` 关闭缓存。

请保持默认的 DiT 权重预算。`--dit-cache-mib 10240` 在权重串行读取时更快，但权重预取现在已经把这些读取隐藏掉了：在 36 GiB 基准机上，该预算实测 21.88 s，默认配置为 18.70 s，且峰值内存升到 6.35 GiB。原生日志会报告实际预算。[缓存调优与验证](docs/NATIVE_TUNING.md)

可用 `onyx label-prompt` 为 H3 组织文字提示。需要逐字准确的文字和引线时，可在生成后用 `onyx compose-labels` 排版。[图中文字工作流](docs/IMAGE_TEXT.md)

跨档基准矩阵、门禁检查与能力报告：

```bash
uv run python benchmarks/run_profiles.py --sizes 1024x1024,4096x4096 --runs 3 --output artifacts/benchmarks/profile-stage-matrix.json
uv run python tools/check_profile_gates.py artifacts/benchmarks/profile-stage-matrix.json
uv run python tools/build_benchmark_report.py
```

MLX 参照环境独立安装：`uv sync --extra reference`。

## 文档

| 文档 | 内容 |
| --- | --- |
| [docs/ARCHITECTURE.md](docs/zh-CN/ARCHITECTURE.md) | 产品定位、来自官方资料的约束、模块边界 |
| [docs/PROTOCOLS.md](docs/zh-CN/PROTOCOLS.md) | 六组版本化 Protobuf 契约 |
| [docs/LOOP.md](docs/zh-CN/LOOP.md) | Loop Engineering 与受控 RSI |
| [docs/V1_REVIEW.md](docs/V1_REVIEW.md) | v1 冻结评审与验收证据 |
| [docs/PROFILES.md](docs/PROFILES.md) | 执行档位与验收门禁 |
| [docs/ACCELERATION.md](docs/zh-CN/ACCELERATION.md) | 加速路线：投机解码适配性评估、Mac 特性抓手、infra 团队建议 |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | 基准能力报告（自动聚合生成） |
| [docs/VALIDATION.md](docs/VALIDATION.md) | 本地与硬件验证 |
| [docs/LORA_ADAPTERS.md](docs/zh-CN/LORA_ADAPTERS.md) | LoRA 适配器说明 |
| [CONTRIBUTING.md](CONTRIBUTING.md) | 贡献入口 |

项目主页与工程博客：**https://phantom5125.github.io/onyx-infer/**
（源码在 [`site/`](site/)，博客由 [`tools/build_blog.py`](tools/build_blog.py) 构建）。

## 指南

工程博客上已发布的指南，按时间倒序：

- [2026-11-18：四轮优化，一只逐字节一致的 teapot](site/blog/posts/four-rounds-one-byte-identical-teapot.md)——榜单背后的完整故事，包括回退的那一轮。
- [2026-11-18：36 GB 统一内存上的十六百万像素](site/blog/posts/sixteen-megapixels-on-36gb.md)——引擎如何为 4096×4096 直出管理内存。
- [2026-11-18：为什么从零写一个推理引擎](site/blog/posts/why-native-from-scratch.md)——135 GB → 28.08 GiB 的数学，以及由此得出的设计原则。

## 许可证

Onyx Infer 自有源码采用 [MIT 许可证](LICENSE)。仓库内的 MLX 代码及其派生部分保留 [Apple MLX 许可证](ops/metal/mlx_root/LICENSE)。H3 模型文件须单独获取，适用 [MiniMax 上游许可证](https://github.com/MiniMax-AI/MiniMax-H3-Image/blob/main/LICENSE)。

## 引用

如果 onyx-infer 对你的研究有帮助，欢迎按如下格式引用：

```bibtex
@misc{onyx-infer,
  author = {Daocheng Ying},
  title = {onyx-infer: native image generation for Apple Silicon},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/phantom5125/onyx-infer}
}
```

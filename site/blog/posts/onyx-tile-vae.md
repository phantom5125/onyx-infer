---
title: OnyxTileVAE 在批处理之上把 4K VAE 延迟再降约三成
date: 2026-11-24
summary: H3 的 441 个 tile 已经分成 18 批。合并注意力子图、滚动拼接之后，4K VAE 阶段从 26.35 秒降到 18.87 秒。
tags: VAE, Metal, Optimization
lang: zh-CN
image: assets/gallery/teapot-1024.jpg
caption: H3 茶壶测试请求示例
---

H3 解码一张 4096×4096 图像要处理 441 个重叠的 VAE tile。主线已经会按 25 个 tile 一组来解码，也就是 18 批。OnyxTileVAE 在这个基础上合并解码子图，再用两块滚动条带拼接像素。在 M5 Max 上，固定同一个 latent 后，从载入 latent 到写完 PNG 的中位耗时由 **26.35 秒降到 18.87 秒**，三轮配对的中位降幅是 **30.4%**。[主线的批处理改动](https://github.com/phantom5125/onyx-infer/pull/6)

## 批处理先消掉了重复提交和权重读取

一个 tile 覆盖 16×16 个 latent 位置，解码成 256×256 像素。解码器在 256 个位置 token 后追加 4 个 register token 和 1 个空 token，形成 261 个 token。相邻 tile 保留重叠，在解码后拼接。tile 在解码时互相独立，可以放到同一个 batch 轴上，每个注意力头仍在自己的 261 个 token 范围内计算。

单 tile 的历史路径为每个 tile 单独执行 24 层解码器。在 1024×1024 用例中，25 个 tile 共调用 14,628 次 MPSGraph。批处理把这个数字降到 587。流式权重也能让每批共享一次读取，1024 的文件读取量从 **40.39 GB 降到 1.62 GB**。这些是运行时请求读取的字节数，实际 SSD 流量取决于系统页缓存。

## 图融合吃掉剩余的提交开销

批处理之后，每层仍分别执行 QKV 拆分与转置、Q/K RMSNorm、RoPE、注意力和输出转置。新子图从 head 内连续存放的 Q、K、V 开始，在图内完成这些步骤，再把结果交回输出投影。FFN 复用现有的门控激活与带缩放的残差图。1024 图像的 MPSGraph 执行次数从 **587 降到 251**。

这两个数字来自应用侧。一次 MPSGraph 执行可能包含多个 GPU kernel。我们还试过把线性层显式展开成二维 token 矩阵。该版本的筛查时间与融合版本重叠，PNG 哈希相同，最终集成继续复用现有线性层。[注意力子图](https://github.com/phantom5125/onyx-infer/blob/022551dd015642ccc0c3e2714fa23a811d3b94d5/runtime/native/graph.mm)

## 两个条带足够保留拼接状态

拼接沿用先纵向、再横向的融合顺序。下一行 tile 开始时，它上方的像素已经定型，可以转换成 RGB8。当前行需要上一行保留的浮点结果来计算纵向重叠。处理下一行时交换两块条带，解码批次的边界和尾批都沿用同一个光栅顺序。

原来保留 192 MiB 的 FP32 画布和 12 MiB 的当前条带。两块条带共 **24 MiB**，外加最终的 RGB8 图像。常驻权重的 4K VAE 进程物理峰值在三个版本上分别是 **2.90、2.57 和 2.38 GiB**。这里的峰值包含 GPU 分配和 CPU 缓冲。[VAE 拼接代码](https://github.com/phantom5125/onyx-infer/blob/022551dd015642ccc0c3e2714fa23a811d3b94d5/runtime/native/vae.mm)

## 把两项改动分开量

在 M5 Max 36 GiB、macOS 27.2.1 上，两种分辨率都以固定 latent、常驻权重和默认 batch 25 做三组交错测量。第二组只加入滚动拼接和打包改动，第三组再启用图融合。计时包含 PNG 编码。

| 用例 | 已批处理的主线 | 加拼接与打包 | 再加图融合 | 配对中位降幅 |
| --- | ---: | ---: | ---: | ---: |
| 1024 VAE | 1.688 s | 1.679 s | 1.192 s | 29.4% |
| 4096 VAE | 26.349 s | 25.779 s | 18.867 s | 30.4% |

滚动拼接的主要收益是缩小缓冲，图融合贡献了大部分延迟降幅。早先从单 tile 主线开始的测试记录了 67.71 → 17.47 秒。这段历史对照同时包含批处理，当前 PR 的增量以 batch 25 对照为准。端到端耗时还包含 Text 和 DiT，两段耗时会随权重缓存和执行环境波动。[PR 的实验记录](https://github.com/phantom5125/onyx-infer/pull/8)

## 保留现有批大小和 PNG 输出

候选沿用主线的默认 batch 25 和 `ONYX_VAE_BATCH=1` 覆盖。1024 和 4096 固定 latent 的三组 PNG 哈希一致，单 tile 和 4K 流式对照也逐字节一致。这里的参考是已经批处理的主线。最初从 batch 1 升到 batch 25 时，4K 茶壶有 653 个 RGB 通道值跨过一个 8-bit 色阶，说明跨批大小的舍入仍需单独比较。

4K 流式单次试验的物理峰值从 2.80 降到 2.38 GiB。2 GiB 的 `memory-min` 目标需要继续在不同批大小和设备上验证。当前改动保留批大小策略，把解码和拼接优化留在 VAE 内。[评审议题](https://github.com/phantom5125/onyx-infer/issues/7)

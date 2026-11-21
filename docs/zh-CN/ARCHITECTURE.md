# Onyx Infer 基础架构

状态：v1 稳定协议与原生实现；后续性能优化继续演进。本文确定首发方向、模块边界和核心技术栈；协议细节见 [PROTOCOLS.md](PROTOCOLS.md)，自动优化机制见 [LOOP.md](LOOP.md)。

## 1. 产品定位

Onyx Infer 是面向 Apple Silicon 的本地图像生成引擎，而不是通用 LLM 推理框架、ComfyUI 后端或开源引擎的包装层。

首发范围：

- 单机、单 GPU 的 M5 Max Mac Studio：18 CPU cores、32 GPU cores、36 GB 统一内存，macOS 27，Metal 4。
- MiniMax-H3-Image，先完成文生图，再支持同一 checkpoint 的多图编辑。
- 原生异步执行、显式内存规划、可重放运行记录。
- 从第一天支持可验证的优化 Loop，后续将高性能算子库独立为 `onyx-ops`。

不在首发范围：训练、多机集群、通用模型托管、隐式云端调用、兼容所有 Diffusers pipeline、开放第三方二进制插件。协议不阻止这些能力以后扩展，但不为它们牺牲首发路径。

## 2. 已确认的首发约束

以下来自 Apple 和 MiniMax 官方资料以及本机 Metal 查询，不是性能推测：

| 项目 | 约束 |
| --- | --- |
| H3 DiT | 33.1B 稠密、单流 Transformer，50 层，hidden size 5376，56 个注意力头，FFN 14336 |
| H3 采样 | 6 NFE，PDD 每步输出头，无 CFG；这些属于模型 recipe，不是引擎全局常量 |
| H3 文本编码 | 完整 Qwen3-VL 文本编码器，读取第 50 层 hidden states；权重约 66.7 GB |
| H3 DiT 权重 | BF16 为主，embedding、timestep 和输出层为 FP32；约 66.28 GB |
| H3 VAE | FP32，约 2.34 GB；16 倍空间下采样、24 latent channels；2D ViT decoder 为 24 层 |
| H3 高分辨率 | 约 16 MP 直接生成，无独立 upscaler；DiT 每个 token 对应 32×32 像素 |
| 本机 M5 Max Mac Studio | Mac17,14；18 CPU cores、32 GPU cores、36 GB unified memory；GPU 核内有 Neural Accelerators |
| 本机 Metal 内存预算 | `recommendedMaxWorkingSetSize = 30,150,672,384 bytes`（28.08 GiB）；`maxBufferLength = 22,613,000,192 bytes`（约 21.06 GiB） |
| 软件能力 | macOS 27、Metal 4；Metal 4 提供 tensor resource、机器学习编码和 MPP tensor operations，可利用 M5 Neural Accelerators |

由此得到五个架构结论：

1. **本机实用首发必须是有损权重量化。** 原始权重约 135 GB，DiT 单独的 BF16 权重就有 66.28 GB，超过 36 GB 物理内存；即使 W8 DiT 理想权重约 33.14 GB，也超过本机推荐 GPU working set，尚未计算激活和系统占用。BF16 只作为离线参照，SSD 逐层流式 BF16 只作为实验兜底。
2. **W4 是首个候选，而不是已经证明的质量结论。** 纯权重位数换算下，W4 DiT 和文本编码器分别约 16.57 GB、16.68 GB；实际还要加 group scales/zero-points、高精度例外层和打包开销。它们可分别进入本机预算，但不能同时驻留。
3. **16 MP 的注意力必须分块。** 4096×4096 图像产生 `(4096/32)^2 = 16,384` 个图像 token，文本和参考图像还会继续增加序列长度；不能物化完整注意力矩阵。
4. **执行必须是阶段化、异步和显式同步的。** 文本编码、采样、VAE 解码具有完全不同的驻留集和算子特征，不能用单一 eager pipeline 串起来。
5. **统一内存不等于无需内存管理。** 是否零拷贝是设备能力；逻辑图仍需保留 transfer 和资源生命周期。运行时必须查询实际预算，不能把 28.08 GiB 写成所有机器通用的常量。

首个本机数值 profile 定为 **weight-only W4A16 候选**：大型线性层使用 group-wise 4-bit weights，激活采用 BF16/FP16，矩阵与归约使用 FP32 accumulation；softmax、normalization 以及官方指定的 FP32 组件优先保留高精度。VAE 先保留 FP32。group size、scale/zero-point 格式、clipping 和例外层由量化实验确定，不能预先宣称无损。

这里的 A16 是 16-bit 激活计算，不是激活量化。W3、混合 W3/W4 或更低 bit 只有在 W4 的内存/质量实测后作为新候选评估；W8 不作为本机主路径。

实测首个原始 checkpoint 的 group-wise W4 派生包中，DiT 为 37.47 GB、文本编码器为 21.86 GB。高精度 AdaLN 投影使 DiT 明显超过理想权重位数换算，也超过本机物理内存；因此首发执行必须支持按 block 流式读取，并在 timestep/AdaLN 预计算后释放其权重。不能把 W4 等同于可 resident，也不能只用量化 bits 估算最终 working set。

## 3. 核心设计原则

- **语义与优化分离。** 模型图描述“算什么”，执行计划描述“在这台机器上怎样算”。
- **小稳定内核。** 永久兼容只覆盖请求、制品身份、张量语义、运行事件和实验证据，不冻结 Metal 对象、线程数、tile size 或缓存格式。
- **数据面绕过序列化。** Protobuf 只承载控制和元数据；大张量、权重和图像通过带所有权的 Artifact/Buffer handle 传递。
- **默认无隐式行为。** 不隐式扩写 prompt、不隐式联网、不隐式量化、不隐式 swap 到磁盘，也不根据设备字符串猜测能力。
- **每个优化都可回放和撤销。** 没有稳定复现材料、正确性证据和原始测量数据，就没有性能结论。
- **原生不等于排斥参考实现。** 官方 PyTorch/Diffusers 实现只用于离线导入和 golden 对照，不进入原生运行时依赖图。

## 4. 总体结构

```text
CLI / 本地应用 / Agent
          │  onyx.job.v1
          ▼
   Session / Job Service ───────────────► RunEvent / Artifact
          │
          ▼
 Model Resolver → Compiler / Planner → ExecutionPlan
          │             │                  │
          │             │                  ├─ phase residency
          │             │                  ├─ layouts / fusion / tiling
          │             │                  └─ kernel selection
          ▼             ▼                  ▼
 Model Package     Semantic IR        Native Runtime
                                      ├─ async scheduler
                                      ├─ memory arena / cache
                                      └─ trace / replay
                                             │
                                      Backend SPI（内部）
                                             │
                              ┌──────────────┴──────────────┐
                              ▼                             ▼
                        Metal 4 backend               CPU reference
                              │
                         onyx-ops variants
                     MSL / MPP / tuned kernels

 Loop Controller 只提交 Candidate，并通过同一组公开协议读取证据；
 它不能绕过 Compiler、Runtime 或正确性门直接修改生产请求语义。
```

依赖方向必须保持单向：前端依赖 Session，Session 依赖编译和运行时，backend 依赖算子接口。`onyx-ops` 不依赖 UI、Python、模型 importer 或 Loop Controller。

## 5. 模块边界

| 模块 | 负责 | 明确不负责 |
| --- | --- | --- |
| Public API | 请求校验、作业、事件、Artifact | Metal 对象、物理布局、模型专属 sampler 常量 |
| Model Importer | 读取原始配置和 SafeTensors，生成模型包 | 原生推理、在线执行自定义模型代码 |
| Semantic IR | 纯数据流、符号 shape、数值和 RNG 语义 | tile、fusion、设备地址、kernel 名称 |
| Compiler / Planner | shape 求解、阶段划分、内存与布局、算子选择 | 改变模型语义；把缓存当作正确性来源 |
| Native Runtime | Session、异步依赖、生命周期、取消、trace | Python eager 调度、逐节点 CPU/GPU 同步 |
| Backend | 设备能力、命令编码、事件、内存域 | 公共作业协议、模型专属流程 |
| `onyx-ops` | 有独立数值契约的 kernel variants | 作业调度、模型下载、prompt 处理 |
| Loop Controller | 候选构建、验证、测量、注册和回滚 | 自行改变 golden、协议或发布规则 |

## 6. 两层计算表示

### 6.1 Semantic IR：长期保留

模型包内的 Semantic IR 是正确性的来源：

- 纯数据流图，所有输入、输出、属性和 shape 可检查。
- 支持符号维度；编译期把分辨率、token 数等绑定为具体值。
- 模型阶段和迭代采样由 recipe 描述，不把 Python 控制流塞进图中。
- 算子有版本化数值定义，包括 accumulation、rounding、epsilon 和 RNG/noise 布局。
- 不包含设备、stride、tile、fusion、线程组织或动态库路径。

### 6.2 ExecutionPlan：可生成、可替换、可丢弃

执行计划保存物理选择：

- 具体 backend、kernel variant、物理 layout、量化编码、fusion 和 tiling。
- 每个阶段的驻留集、arena 大小、复用关系和显式 transfer。
- 命令依赖和可并发的独立工作，但禁止为并发引入未声明的数值不确定性。
- 从优化后节点到原 Semantic IR 节点的映射，便于逐阶段对账。
- 由模型语义哈希、设备能力、编译器/算子构建和数值配置共同键控。

计划缓存随时可以删除；删除后只能损失编译时间，不能改变正确性。Loop 的大多数优化只更换 ExecutionPlan 或 KernelVariant，不修改 Semantic IR。

## 7. H3 的执行映射

首发 H3 recipe 固定为以下阶段，具体实现可优化但语义边界不可丢失：

1. **Prepare**：使用最终 prompt；校验尺寸和模型能力；按指定 RNG 生成初始 noise。Prompt expansion 是独立前置产品能力，原 prompt、展开 prompt 和展开模型版本必须由上层记录。
2. **Condition**：运行 W4 文本编码器，读取指定层；将 conditioning 作为有哈希的中间 Artifact 缓存。36 GB 首发计划在进入采样前释放整个文本编码器驻留集。
3. **Sample**：加载 W4 DiT，执行 6 步 PDD schedule，无 CFG。高分辨率 attention 使用在线归约或分块算法，不保存完整 score matrix；临时 arena 必须与量化权重一起计入 28.08 GiB 推荐预算。
4. **Decode**：采样完成后释放 DiT，再加载 FP32 VAE decoder；大分辨率下以可证明等价的 tiling 控制峰值内存。更大内存机器可选择保留 DiT，但不改变本机基线。
5. **Publish**：输出图像 Artifact 和 RunRecord。编码格式、压缩和显示不属于采样语义。

Conditioning cache 的键至少包含文本编码器语义哈希、最终 prompt、参考输入、预处理和数值配置。不能只用 prompt 字符串做键。

首发先建立可运行的 W4 模型与质量基线，再由测量优化大 GEMM、分块 attention、QKV/RoPE/归一化融合、FFN 数据布局、阶段驻留和 VAE 解码。量化是新的派生模型制品和数值实验，具有独立模型身份、校准来源和质量证据，不能伪装成无损计划优化。

## 8. 核心技术栈决策

| 层 | 决策 | 理由与边界 |
| --- | --- | --- |
| 原生核心 | C++20、Apple Clang | 适合内存/异步/算子开发，与 Objective-C++ 和 C ABI 直接互操作；不在热路径引入第二门系统语言 |
| Apple 桥接 | 极薄 Objective-C++ `.mm` 层 | Metal 对象、API availability 和资源所有权限制在 backend 内，不泄漏到公共协议 |
| GPU | Metal 4、Metal Shading Language、Metal Performance Primitives | 直接覆盖 M5 GPU Neural Accelerators、tensor operations 和权重压缩能力；按运行时能力选择路径 |
| 构建 | CMake + Ninja + Xcode 工具链 | C++/Objective-C++/MSL 可统一构建；shader 预编译为 metallib，运行时缓存只作加速 |
| 控制与元数据 | Protocol Buffers，独立版本命名空间 | 跨语言、可加字段、有 JSON 映射；张量字节不经过 Protobuf |
| 模型输入 | SafeTensors + 原始配置 | 易于 mmap、校验和内容寻址；自定义打包格式只能在实测证明收益后作为派生制品加入 |
| CPU 支撑 | C++ reference kernels；Accelerate/ImageIO 等系统框架按需使用 | 用于确定性对照和系统图像能力，不把 CPU reference 当高性能实现 |
| 离线工具 | Python 3.12、uv；PyTorch/Transformers/Diffusers 为可选开发依赖 | 最适合模型导入、golden 生成、质量评估和 agent 工具；原生运行时无 Python 依赖 |
| 观测 | 自有版本化 trace + `os_signpost` + Metal 计数器 | RunRecord 可跨工具消费，Apple 工具用于深入诊断；热路径不依赖外部 telemetry 服务 |
| 测试 | 后续采用 CTest 驱动原生测试，Python 测试覆盖 importer/Loop | 统一输出协议化验证记录，而不是只打印“通过” |

### 不作为核心运行时的一些选择

- **MLX**：可作为对照或社区适配器，不让其数组运行时成为 Semantic IR、内存规划或公共 ABI 的上位依赖。
- **PyTorch MPS / Diffusers**：用于参考和导入，不作为首发原生执行路径。
- **Core ML / Core AI / ANE**：先测量再决定是否为文本编码器或 VAE 提供独立 backend；不把整个图预先转换成 Core ML，以免失去自定义 fusion、布局和内存控制。
- **MPS Graph**：可作为某个计划的后端候选，不是全局图语义。
- **ONNX / GGUF**：可作为导入来源，不直接承担多阶段图像模型的内部 IR 或制品协议。
- **Rust**：适合以后编写独立工具，但首发核心同时采用 Rust 与 C++ 会增加 Metal FFI 和构建复杂度，没有足够收益。

这些不是永久禁用清单。社区实现只有满足同一接口和相同验证门，才可替换对应模块。

## 9. 面向独立算子库的设计

`onyx-ops` 从逻辑上独立，首发仍可同仓构建：

- `OpDefinition` 定义数学语义、输入输出和数值容差。
- `KernelVariant` 绑定具体物理 layout、设备能力、量化格式、tile 参数和实现哈希。
- 同一 OpDefinition 可有 reference、通用 Metal、MPP、M5 专用和社区 variant。
- 选择器基于能力、shape 和测量成本选择 variant；不能通过修改模型 Python 代码选择优化。
- 每个 variant 都有独立微基准、跨 shape 正确性和端到端回放证据。

首发阶段只冻结算子的**语义契约**。C++ backend SPI 保持仓内源码演进；等真实 H3 工作负载跑通并经过兼容 soak 后，再发布 `onyx.abi.v1` 动态插件 ABI，避免过早冻结错误的 Metal 抽象。

建议的未来源码边界，不在本设计阶段提前生成空壳：

```text
schema/             # job/model/trace/loop 的规范 schema
runtime/            # session、job、memory、scheduler、artifact
compiler/           # semantic IR、planner、plan cache
backends/metal/     # Objective-C++ 桥、Metal command/resource
ops/                # 可独立构建的 onyx-ops
models/h3/          # H3 importer 与 recipe，不含通用引擎分支
tools/              # CLI、replay、inspect、benchmark
loop/               # 候选运行与证据管理
tests/              # golden、数值、协议兼容和端到端验证
```

## 10. 交付顺序与验收

### M0：冻结最小契约

- 评审三份设计文档；基准机已确认为 M5 Max 36 GB，Metal compiler `32023.921` 与 ModelScope H3 revision `df334556f585a596a710cf4f7ce3a567c0738ee5` 已锁定。
- 实现 schema 后再宣布 v1 冻结；当前文档中的签名仍是草案。
- 保存官方 BF16 参考的逐阶段 golden、固定 noise 和测试请求，不只保存最终图片。
- 冻结首个量化评估集、质量门和校准集；校准数据不能兼任最终验收集。

### M1：W4 原生垂直切片

- SafeTensors importer、W4 派生模型包、原生 Session/Job、Metal backend 和 CPU reference。
- 本机运行 weight-only W4A16，敏感层保留高精度，官方 6 步 recipe，无 prompt expansion 和编辑。
- 先以 BF16/FP32 小算子和小阶段对照定位量化误差，再完成官方有效范围内 4 MP 的 W4 端到端文生图与质量评估。

### M2：16 MP 与内存目标

- 分块 attention、VAE tiling、文本编码器/DiT/VAE 分阶段释放和 conditioning cache。
- 在本机 28.08 GiB 推荐 GPU working set 内跑通 4096×4096；若实测必须超出推荐值，应作为明确实验风险记录，不能当作已达标。
- 未经实测不承诺具体秒数、最终模型大小或 W4 质量达标。

### M3：Loop 成为主开发方式

- 自动重放、交错 A/B、冷/热启动区分、峰值内存与热状态记录。
- 优化 candidate 通过数值、端到端质量和性能门后进入本地 registry；可一键回滚。

### M4：独立算子库与社区替换

- 将 `onyx-ops` 的构建、测试和制品身份独立。
- 在 ABI soak 后发布版本化 C ABI 和签名插件机制；社区 backend/variant 走同一验证流程。

统一验收至少包括：逐算子误差、逐阶段误差、固定输入重放、最终图像质量、冷/热延迟分布、峰值内存、干净构建复现和协议兼容测试。任何单一 prompt 的平均耗时都不足以作为发布依据。

## 11. 当前假设

- “H3 Image”指 MiniMax-H3-Image，而非 HunyuanImage 3.0。
- 首发基准机已确认为 M5 Max、32 GPU cores、36 GB；协议仍通过能力查询适配其他 Apple Silicon，不写死本机型号。
- 本机首发以有损 W4 权重为功能前提；BF16 是离线质量参照，W4 的具体格式与质量接受线仍需实验验证。
- 默认使用调用者给出的最终 prompt，不在引擎内隐式调用云端或本地 prompt expander。
- 首个发布为单机单 GPU；Thunderbolt/RDMA 集群不进入当前协议稳定面。

## 12. 官方依据

- MiniMax H3 模型结构、权重组成、采样与本地请求语义：<https://github.com/MiniMax-AI/MiniMax-H3-Image>
- ModelScope 原始制品与本项目锁定来源：<https://modelscope.cn/models/MiniMax/MiniMax-H3-Image>
- Apple M5 Max/M5 Ultra Mac Studio 硬件与统一内存说明：<https://www.apple.com/ee/newsroom/2026/08/apple-introduces-mac-studio-with-m5-max-and-m5-ultra/>
- Apple Metal 开发指南与 Metal 4/MPP 入口：<https://developer.apple.com/guides/metal/>

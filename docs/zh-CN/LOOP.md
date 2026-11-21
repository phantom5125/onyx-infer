# Loop Engineering 与受控 RSI

状态：v1 稳定协议；实现与 kernel variants 持续通过 Loop 优化。这里的 RSI 指在固定语义和验证契约下，反复生成、验证、测量和注册更优实现；不是让 agent 无约束改写协议、测试或生产系统。

## 1. 核心闭环

```text
不可变 OptimizationCase
          │
          ▼
 Candidate（plan / kernel / runtime patch）
          │
          ▼
 干净构建 → 正确性验证 → 可重放基准 → 独立评估
          │                              │
          │失败                          │未达门限
          ▼                              ▼
       Rejected                       Rejected
                                      │
                    通过全部门限 ─────┘
                                      ▼
                         本地 Candidate Registry
                                      │
                         显式发布 / 随时回滚
```

Loop 的主要动作应是：

- 更换 ExecutionPlan 中的 layout、fusion、tiling、驻留或 kernel variant。
- 提交一个有明确基线和回滚点的最小实现 patch。
- 增加新的参考对照或性能 shape，但不能为了让候选通过而修改原 Case。
- 将历史 RunRecord 和 Evaluation 作为下一轮输入，而不是只保留“最快一次”的数字。

Semantic IR、稳定协议、golden 来源和发布策略不属于普通优化 candidate 的可写范围。

## 2. 四类持久记录

### 2.1 OptimizationCase

Case 是冻结的工作负载与验收定义：

```text
OptimizationCase {
  case_id and semantic version
  model semantic_digest + immutable source revision
  actual replay inputs or encrypted/private input references
  resolved sizes, recipes, seeds/RNG and numeric profiles
  target device capability fingerprint
  baseline build/plan/run identity
  correctness suite and tolerances
  benchmark method and primary metric
  experiment budget and stop criteria
}
```

Case 必须区分小 shape 算子测试、4 MP 功能测试和 16 MP 压力测试。只优化一个分辨率、一个 prompt 或热缓存，不代表整个 H3 工作负载改善。

### 2.2 Candidate

```text
Candidate {
  candidate_id
  base source revision and immutable Case ID
  patch/artifact SHA-256
  allowed scope: plan | kernel | runtime | importer
  build configuration and toolchain identity
  resulting plan/kernel identities
  proposer and generation context references
}
```

- 一个 candidate 尽量只改变一个主要变量；融合多个变化时必须能拆分归因。
- 不允许修改 Case、容差、基准次数或统计方法来制造收益。
- 不允许把联网结果、未固定 revision 的下载或本机未声明状态带入构建。
- Agent 的推理文本不是复现依据；实际 patch、输入、配置和日志才是。

### 2.3 Experiment

Experiment 绑定 Case、Candidate、原始 RunRecord 列表和测量环境。每次运行使用相同的公共提交路径，不能为 candidate 调用一条绕过 scheduler 的“专用捷径”后声称端到端获益。

至少区分：

- 冷模型、冷 shader/plan cache 的首次运行。
- 已加载、已编译的热运行。
- 逐阶段时间：condition、每个 sampling step、VAE decode、publish。
- 端到端延迟、峰值驻留、临时内存、传输/同步和编译时间。
- 设备、温度/热状态、其他负载和测量来源。

### 2.4 Evaluation

```text
Evaluation {
  experiment identity
  correctness results per level and shape
  latency distributions and raw run references
  peak memory and optional energy/thermal results
  quality results where numeric profiles changed
  comparison method and confidence/variability
  decision: accepted | rejected | inconclusive
  machine-readable reasons
}
```

`inconclusive` 是一等结果。测量噪声过大、热状态不可比或参考失败时，不得强行提升 candidate。

## 3. 正确性门

按成本从低到高执行，前一门失败就不继续昂贵的图像基准：

| 层级 | 验证内容 | 典型用途 |
| --- | --- | --- |
| L0 Op | 对 CPU/reference 的输入输出、边界 shape、NaN/Inf、accumulation 和 layout | GEMM、attention、norm、RoPE、量化 kernel |
| L1 Stage | 固定中间输入，对文本条件、单步 DiT、PDD head、VAE 分块输出对账 | fusion、tiling、驻留和 VAE 优化 |
| L2 Replay | 相同模型、noise、plan 和构建重复执行；验证声明的确定性等级 | 异步调度、arena 复用、原子操作 |
| L3 End-to-end | 固定请求生成最终图像，验证数值/感知质量和能力约束 | 所有可发布 candidate |

需要区分三种保证：

- **Bitwise deterministic**：仅对明确声明的构建、设备和计划成立。
- **Numerically equivalent**：在 op/stage 容差内，允许归约顺序变化。
- **Quality equivalent**：低 bit 量化或较大数值改动使用固定图像集和质量评审，不能用单张肉眼观察代替。

容差按 op、dtype 和 shape 预先定义。不能统一使用一个宽松像素误差掩盖中间阶段已经发散的问题。

## 4. 性能实验方法

- 基准机和后台负载固定；记录 OS、Xcode、Metal library、供电和热状态。测量期间停止模型下载、构建和其他 GPU 任务，避免磁盘、内存与 page cache 竞争。
- 预热次数、测量次数、最大时长和停止条件由 Case 预先声明，不因结果好坏中途修改。
- baseline 与 candidate 交错 A/B，避免先跑完全部 baseline 再跑 candidate 造成热漂移偏差。
- 同时报告中位数、尾部分位、离散程度和全部原始 RunRecord；不能只保留最小值。
- 冷启动、热运行和持续负载分开给结论。首次 shader 编译不能藏进“初始化不计时”而没有独立指标。
- 主要指标由 Case 指定，例如固定 16 MP 请求的热态端到端 p50；次要指标和保护指标同时检查。
- 统计不显著时增加预先允许的测量或标记 inconclusive，不宣称提升。

对 H3，标题指标应是固定尺寸图像的端到端时间和阶段时间，而不是借用 LLM 的 tokens/s。内存可行性本身是发布门，不能用严重 swap 后的低吞吐结果参与性能排名。

## 5. 提升、注册与回滚

Candidate 进入本地 registry 的最低条件：

1. 所有适用 L0–L3 验证通过，且没有修改原 Case。
2. 在目标 shape 矩阵上没有超过 Case 保护线的回退。
3. 主指标有可复现且超过测量噪声的改善；纯重构可以“性能不降级”为门，但必须明确标注。
4. 峰值内存满足目标配置，且无未声明 swap 或隐式云端回退。
5. 从干净 checkout 和固定制品可重建，RunRecord 完整。
6. 存在唯一身份和可回滚的上一稳定 plan/kernel。

自动提升只进入隔离的本地 candidate registry；正式发布、签名和公共协议变更保留独立发布控制。发现回归时按 identity 回退，不在原制品上覆盖修复。

社区优化与内部 agent 使用同一流程。MPP、Core ML、MLX 或第三方 Metal kernel 若成为候选，只替换其声明层级的模块，并重新走完整验证，不因为来源权威而豁免。

## 6. 隔离与安全边界

- Candidate 在独立 worktree/进程和受限缓存中构建运行；默认无网络、无凭据、无生产 Artifact store 写权限。
- 模型包和 shader 按数据校验；第三方原生代码需显式信任，不因扩展名是 `.metallib` 就视为安全。
- Loop Controller 不能修改自己的预算、正确性门、发布密钥或稳定协议。
- 每个 Loop 有最大实验数、时长、能耗/机器占用或其他明确预算；达到预算返回现有最佳已验证结果，而不是无限搜索。
- 私有 prompt 和图像只在显式导出的 Case 中进入长期 replay 材料；普通 trace 默认记录摘要。

## 7. H3 的初始 Loop 队列

首发机器是 M5 Max 36 GB，Metal 推荐 GPU working set 为 28.08 GiB。以下从量化功能前提开始，而不是先假设 BF16 能完整运行：

1. **建立 W4 派生制品**：从固定 BF16 源生成文本编码器与 DiT 的 group-wise W4 候选；固定校准集、独立验收集、例外层和数值 profile，记录实际打包大小。
2. **建立质量基线**：对固定 noise 做逐层/逐阶段误差分析，并在独立 prompt 集上比较 BF16 与 W4 最终图像。先确定可接受质量，再把通过门的 W4 版本设为本机运行基线。
3. **36 GB 驻留路径**：验证 condition 后释放文本编码器、DiT 采样后释放 DiT、VAE 独立解码和 conditioning cache；每阶段都按实际 Metal 预算验收。
4. **建立分解性能基线**：文本编码、单步 DiT、六次采样、VAE 和图像编码分别测量，确认同步点、swap 和真实峰值内存。
5. **量化大 GEMM**：比较 MPP tensor ops 与自研 MSL variant，覆盖 group size、hidden/FFN 形状和不同 token 数；不假设 4-bit 天然快于 BF16。原生路径必须直接消费 Onyx packed 格式，避免参照路径中逐步重复的格式转换；格式转换开销单独计量。
6. **分块 attention**：避免完整 score matrix，评估在线 softmax、序列分块、Q/K/V layout 和高分辨率临时内存。
7. **局部 fusion**：在数值契约不变时融合 dequant、QKV、RoPE、normalization、timestep/PDD 相关小算子，减少中间读写。
8. **VAE decode**：24 层 ViT 与输出图像的 tiling、边界一致性和 FP32 峰值内存。
9. **更激进的混合精度**：只有 W4 质量或内存不满足目标时，才评估 W3/W4 混合、outlier 通道高精度和更细 group；每个结果都是新模型身份并重新过质量门。

每项优化先证明小 shape 和单阶段正确，再进入 16 MP 长时间实验。Loop 的输出不是“一次跑通”，而是可重放证据、可替换实现和可撤销注册决定。

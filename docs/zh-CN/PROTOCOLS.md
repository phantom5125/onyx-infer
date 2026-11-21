# Onyx 基础协议与接口

状态：v1 稳定。`schema/v1.compatibility.json` 的兼容基线已冻结；同一 major 只允许经过检查的 additive 扩展。目标是让调用方、模型制品、优化 Loop 和未来的算子实现围绕少量稳定契约协作，而不是围绕某个运行时类的内部结构协作。

## 1. “不会更改”的兼容定义

协议不可能停止演进；真正的承诺应是：**同一 major 内只做兼容扩展，语义破坏通过新 major 并行提供，旧制品可迁移或明确拒绝，绝不静默误读。**

稳定面分三层：

| 层 | 内容 | 兼容策略 |
| --- | --- | --- |
| L0 不变量 | 身份与哈希、张量语义、作业终态、所有权、异步完成、数值/RNG 声明、证据链 | 不可在原 major 内改变 |
| L1 稳定协议 | `onyx.job`、`onyx.model`、`onyx.trace`、`onyx.loop` | major 内只加 optional 字段和新能力 |
| L2 可替换实现 | ExecutionPlan、plan cache、backend SPI、KernelVariant、Metal 命令和内存算法 | 独立版本、可重建；不进入公共兼容承诺 |

`onyx.abi.v1` 属于未来公开的二进制插件边界。首发只确定 ABI 规则，不在实现验证前声称已冻结具体函数表。

## 2. 协议族和传输

| 协议 | 用途 | 持久性 |
| --- | --- | --- |
| `onyx.job.v1` | Prepare、Submit、事件、取消、输出 | 请求和终态记录可持久化 |
| `onyx.model.v1` | 模型 manifest、Semantic IR、recipe、组件和权重引用 | 长期保存 |
| `onyx.plan.v1` | 已解析 shape、物理布局、驻留、kernel 选择 | 可缓存、可删除、可重建 |
| `onyx.trace.v1` | 有序事件、实际执行身份、性能与输出 | 长期保存，支持 replay |
| `onyx.loop.v1` | Case、Candidate、Experiment、Evaluation 和注册决定 | 长期保存，只允许追加 |

规范表示采用 Protocol Buffers。每个消息以完整协议名和 major 标识，不以 runtime 的语义化版本代替协议版本。JSON 是调试和本地 HTTP 的映射，不是另一套语义。

传输与协议解耦：首发 CLI 可以进程内调用，本地服务可使用 Unix domain socket；未来 HTTP/gRPC 不改变消息语义。大张量和图像字节不放入 Protobuf，统一通过 Artifact 或 Buffer handle 引用。

基础单位固定为：字节、单调时钟纳秒、像素、无序 UUID/128-bit 标识和 SHA-256。任何新字段都不能复用旧字段而改变单位。

## 3. 公共 Session / Job 接口

以下是语言无关签名，不是提前冻结的 C++ 头文件：

```text
Engine.create_session(SessionConfig) -> Session
Session.prepare(ModelRef, PrepareOptions) -> PreparedModel
Session.submit(GenerateRequest) -> Job
Job.events() -> ordered stream<RunEvent>
Job.cancel(CancelReason) -> accepted | already_terminal
Session.release(handle) -> Status
Session.close(drain | cancel_jobs) -> Status
```

语义约束：

- `prepare` 解析制品、能力和计划，不表示所有权重已经驻留。
- `submit` 是非阻塞操作；实际 backend、plan、数值配置和模型 digest 在运行记录中解析并固定。
- Session 拥有其创建的 handle；跨 Session 使用、释放后使用或并发释放都是明确错误。
- 不暴露全局可变 singleton。缓存是 Session 或显式共享缓存的资源，不影响请求语义。
- `release` 只释放逻辑所有权；底层资源可在设备完成和引用归零后回收。
- 每个 Job 恰好有一个 terminal event，terminal 后不再有事件。

### 3.1 ModelRef

| 字段 | 语义 |
| --- | --- |
| `model_id` | 稳定逻辑名称，如 `minimax-h3-image` |
| `source_revision` | 上游不可变 revision；不可只用 `latest` |
| `manifest_digest` | 本地模型包的内容身份；replay 必需 |
| `recipe_id` | 模型声明的采样/预处理 recipe |

用户友好的标签可以在提交前解析，但 RunRecord 必须保存完全解析后的值。

### 3.2 GenerateRequest

```text
GenerateRequest {
  request_id: UUID
  model: ModelRef
  prompt: UTF-8 string
  size: { width: uint32, height: uint32 }  // 像素
  seed: optional uint64
  references: ordered list<ArtifactRef>
  image_count: uint32
  output: OutputSpec
  execution: ExecutionHints
  client_metadata: map<string, string>     // 不参与模型语义和哈希默认值
}
```

规则：

- 同一 Session 内重复 `request_id` 且请求摘要相同，返回原 Job；摘要不相同则返回 `IDEMPOTENCY_CONFLICT`，不能执行两次。
- `seed` 缺省时生成一次实际 seed 并写入 RunRecord。JSON 中 uint64 使用十进制字符串，避免 JavaScript 精度损失。
- `size` 是最终输出尺寸；是否要求 32 的倍数、有效面积和比例由模型 recipe 校验，不能由通用引擎猜测。
- `references` 的顺序属于语义。首发文生图要求为空；编辑支持后沿用同一字段，不复制一套隐式输入。
- `image_count` 首发可以为 1；不支持批量时必须明确拒绝，不能排队后伪装成支持。多样本请求的子 seed/noise 派生顺序由 recipe 和 RNG 规范定义，并逐样本记录，不能依赖循环执行顺序。
- `recipe_id` 选择模型声明的能力。通用协议不强行暴露 `guidance_scale`、`negative_prompt` 或任意 steps；H3 固定 6 步、无 CFG，非法覆盖返回 `UNSUPPORTED_PARAMETER`。
- `ExecutionHints` 可表达 backend、编译策略、缓存和 deadline 偏好，但不能暗中改变模型语义。任何数值近似都必须来自已声明的 numeric profile 并进入 RunRecord。
- 请求中的未知字段必须报错，防止调用者以为某项意图已生效。

Prompt expansion 不在该请求中隐式发生。上层若使用展开器，应将展开结果作为本请求的 `prompt`，并在自己的产品记录或 Loop Case 中同时保存原始输入和展开链。

## 4. Artifact 与内容身份

```text
ArtifactRef {
  artifact_id: content-addressed ID
  media_type: string
  byte_size: uint64
  sha256: bytes
}

ArtifactDescriptor {
  ref: ArtifactRef
  kind: image | tensor | weights | model | plan | trace | patch | other
  schema: optional protocol identifier
  local_locator: optional opaque locator
}
```

- 内容哈希对原始字节计算；`local_locator` 不是身份，不能进入 replay 语义。
- 派生制品保存 `derived_from` 和转换工具摘要。例如 INT4 权重不能覆盖原 BF16 权重的身份。
- 权重可继续采用分片 SafeTensors；manifest 记录每个分片的哈希和逻辑 tensor 对应关系。
- Artifact store 可以垃圾回收未被 manifest、Session 或 RunRecord 引用的数据，但不能原地修改已有 ID 的内容。
- 模型包只包含数据和声明，不在加载时执行包内代码。

模型 `semantic_digest` 与文件哈希分离：前者覆盖 Semantic IR、recipe 和权重内容身份，不包含 plan、路径、kernel 名称或物理布局。规范哈希输入使用 `onyx-canon-v1`（基于 RFC 8785 JCS 的逻辑 JSON 映射）；不依赖 Protobuf 实现的 map 顺序或偶然二进制布局。浮点属性、整数和 uint64 的 JSON 表示必须在 schema 中固定。

## 5. 作业事件和错误

```text
RunEvent {
  run_id: UUID
  request_id: UUID
  sequence: uint64
  monotonic_time_ns: uint64
  wall_time: optional timestamp
  stage: optional StageIdentity
  payload: oneof {
    accepted | stage_started | progress | metric |
    artifact_ready | completed | failed | cancelled
  }
}
```

- `sequence` 从 0 开始严格递增，用于消费去重和断点恢复；时间戳不能代替顺序。
- 耗时使用单调时钟，审计时间使用 UTC；两者不能混用。
- `progress` 使用阶段索引、step index 和已声明总量，不要求虚假的百分比精度。
- 完成事件必须引用最终 RunRecord；输出通过 `artifact_ready` 或完成事件中的 ArtifactRef 提供。
- 取消是协作式的，最多在已声明的阶段/步骤边界生效；已经完成的 GPU 工作不能假装回滚。

稳定错误结构：

```text
Error {
  code: ErrorCode
  retryability: no | after_change | transient
  stage: optional StageIdentity
  message: human-readable, non-contractual
  details: typed extension
}
```

首组错误码：`INVALID_REQUEST`、`UNSUPPORTED_PARAMETER`、`MODEL_NOT_FOUND`、`ARTIFACT_INTEGRITY`、`CAPABILITY_MISSING`、`PLAN_MISMATCH`、`RESOURCE_EXHAUSTED`、`NUMERIC_FAILURE`、`CANCELLED`、`DEADLINE_EXCEEDED`、`BACKEND_FAILURE`、`INTERNAL`、`IDEMPOTENCY_CONFLICT`。调用方只依赖 code，不匹配 message 文本。

## 6. `onyx.model.v1`：模型语义协议

逻辑模型包包含：

```text
ModelManifest {
  model identity and source provenance
  source_model_semantic_digest
  components: text encoder | transformer | VAE | tokenizer | auxiliary
  stages and allowed dependencies
  recipes and supported request capabilities
  numeric and quantization profile identity
  semantic_digest
}

SemanticGraph {
  values: ValueId -> LogicalTensorType
  nodes: versioned op, ordered inputs, typed attributes, outputs
  symbolic dimensions and constraints
}

Recipe {
  stage schedule and iterative regions
  per-step inputs and auxiliary artifacts
  RNG specification and noise layout
  numeric profile
  request constraints
}
```

约束：

- 图是纯语义数据流；阶段调用和采样循环由 recipe 显式描述。
- 所有 op 使用 `name@version`；遇到未知 op、dtype 或关键属性必须拒绝加载，不能忽略后执行。
- shape 可以是符号表达式，但 prepare 后必须完全求解，或对已声明的动态范围生成合法计划。
- 节点数值语义包含 accumulation、rounding、epsilon 和边界行为。禁止把全局 fast-math 开关当作隐式默认。
- RNG 由算法标识、版本、seed、noise shape/order 和变换共同定义。仅有 seed 不足以重放。
- H3 默认 recipe 若要与官方结果对齐，应复现其 CPU generator/noise 语义；未来 GPU counter-based RNG 必须使用新 RNG 标识，不能声称同 seed 必然得到同一 noise。
- PDD 每步输出头和 schedule 都是 H3 模型制品的一部分，不硬编码进 scheduler。
- 模型适配必须显式处理原始 H3 的 head 交错 QKV 布局 `[head][q/k/v][128]` 与执行时连续 Q/K/V 的差异；不能把融合矩阵的存储顺序默认为 `[q,k,v]`。原始 timestep 的 `flip_sin_to_cos=True` 同样要求 `[cos,sin]`，这些都有独立数值回归。

有损量化模型是派生模型包，不是原 BF16 包的隐藏执行选项。其 manifest 必须引用 `source_model_semantic_digest`，并产生新的 `semantic_digest`；调用方通过 ModelRef 明确选择它。

`QuantizationProfile` 至少声明：

- 算法与版本、校准集/评估集内容身份、量化工具与参数摘要。
- 每个 tensor 或 tensor class 的 bit width、signedness、per-tensor/per-channel/group-wise 粒度和 group size。
- scale/zero-point 的 dtype、布局、舍入、clipping、对称性与异常值处理。
- 保留 BF16/FP16/FP32 的例外层，以及激活、矩阵/注意力 accumulation 和 softmax/normalization 的数值模式。
- 打包后权重大小、每分片哈希和端到端质量证据引用。

Loop 可以改变这些参数来生成新候选，但不能在执行同一 ModelRef 时暗中更换量化格式。低 bit 模型与官方 BF16 之间只承诺已验证的质量等级，不承诺 bitwise 或逐阶段数值等价。

## 7. 张量、内存与同步契约

### 7.1 逻辑与物理描述分离

```text
LogicalTensorType {
  dtype: bf16 | fp16 | fp32 | fp64 | signed/unsigned integers | bool
  shape: list<literal or symbolic dimension>
}

PhysicalTensorDesc {
  logical_type
  storage_dtype and optional QuantizationDesc
  memory_space: host | shared | device
  byte_size, strides, alignment
  optional tiling/swizzle description
}
```

逻辑 shape 的轴顺序不由 kernel 自行解释。物理 stride、channel packing、swizzle 和量化 block scale 必须完整描述，不能依靠“Metal 通常如此”的隐式约定。

### 7.2 Buffer 与生命周期

`DeviceCapabilities` 必须直接报告 `physical_memory_bytes`、`recommended_working_set_bytes`、`max_buffer_bytes`、unified-memory 和支持的 tensor/quantization 能力。Planner 使用本次 Session 的实际预算，不能从 `M5 Max` 等名称推导。首发 M5 Max 36 GB 的实测推荐值为 28.08 GiB，但这只是设备输入，不是协议常量。

- 公共 Buffer 是 opaque handle，不暴露可长期保存的 GPU 虚拟地址。
- 每个 allocation 有明确 owner、byte size、memory space 和引用/作用域生命周期。
- host pointer 只在已映射且同步完成后有效；GPU 使用期间 host 不得读写，除非协议明确允许。
- 统一内存上的 zero-copy 是 plan 优化。Semantic IR 中的 transfer 表示逻辑依赖，backend 可在能力允许时消除它。
- arena 复用必须经过 liveness 和依赖证明；优化不能重用仍被异步 GPU 命令引用的存储。
- 内存不足在 prepare 或阶段边界显式报错。不得静默启用磁盘 swap，导致性能记录失真。

### 7.3 异步执行

backend 内部采用类似以下的三阶段接口：

```text
prepare(op/signature, physical descriptions, device capabilities)
  -> immutable compiled program + resource requirements

encode(context, inputs, outputs, dependency events)
  -> completion event

wait/event_stream
  -> completion, error, or diagnostic counters
```

`prepare` 可编译 shader、选择 tile 和建立资源；`encode` 只提交工作，不隐式等待 CPU。backend 的 Event 是依赖令牌，不是公共 Job event，也不能直接暴露为跨平台 ABI 的 Metal 类型。

## 8. 算子与 backend 的替换边界

长期稳定的是 OpDefinition，不是某个 C++ 基类：

```text
OpDefinition = op@version + logical types + attributes + numeric contract
KernelVariant = OpDefinition + physical signature + device capabilities
              + implementation identity + tuning parameters
```

一个 variant 必须声明：

- 支持的完整物理签名和 shape 范围，而非仅写 `supports_m5 = true`。
- accumulation/rounding、确定性和原子操作行为。
- 编译制品、源码和构建参数摘要。
- 所需临时内存、对齐和异步资源生命周期。
- reference 对照和 shape 覆盖证据。

首发 backend SPI 是仓内 C++ 源码接口，可随实现演进。计划、kernel registry 和 Metal command encoder 不进入公共 Job ABI。

未来 `onyx.abi.v1` 必须遵守：固定宽度类型、`struct_size`、显式 `abi_version`、opaque handle、函数表查询、能力协商、调用方/被调方所有权对称；禁止 C++ 异常、STL、RTTI、C++ 对象布局或 allocator 跨越动态库边界。旧函数表不能被原地重排，扩展通过新函数表或新 major。

## 9. `onyx.plan.v1`：可丢弃物理计划

计划至少记录：

- `semantic_digest`、解析后的 shape 和 recipe 身份。
- 设备能力指纹、OS、backend、compiler、metallib 和 KernelVariant 身份。
- numeric profile、量化/布局/fusion/tiling 选择。
- 阶段驻留集、临时内存上界、arena 复用和 transfer/dependency。
- 到原 Semantic IR 节点的来源映射。
- 完整 cache key 和计划构建摘要。

任何一个影响执行的输入变化都必须生成新 cache key。缓存加载后要校验完整身份；不匹配返回 `PLAN_MISMATCH` 并重建，而不是尝试“兼容执行”。计划不是 golden，也不是模型正确性的唯一来源。

## 10. `onyx.trace.v1`：可重放运行记录

RunRecord 至少包含：

- 实际请求摘要、完全解析的 ModelRef、recipe 和 request constraints。
- 实际 ExecutionPlan、KernelVariant、numeric/quantization profile、量化权重制品、RNG 和初始 noise 摘要。
- 设备能力、OS、runtime/backend 构建和编译参数。
- 有序阶段事件、取消/失败、输出 ArtifactRef 和逐阶段指标。
- 冷/热缓存状态、峰值内存、能耗与测量来源（若平台支持）。

默认运行日志可以只保存 prompt 哈希和 Artifact 引用，避免本地隐私数据无意扩散；用于 replay 的 Loop Case 必须经显式导出，包含重现所需的实际输入。没有实际输入的摘要记录只能审计，不能声称可完整重放。

高频 kernel 计时可进入独立 trace 分片，RunEvent 只承载低频控制事件。二者通过 run/stage/kernel 身份关联，不用字符串拼接猜测关系。

## 11. Schema 演进规则

同一 major 内允许：

- 新增 optional 字段，且缺省行为与旧版完全一致。
- 新增事件、能力、op variant、Artifact kind 和可选指标。
- 新增不影响旧请求的执行计划实现。

同一 major 内禁止：

- 修改字段编号、单位、默认值、枚举含义、张量轴语义、RNG 或错误分类。
- 将 optional 字段变为事实上的必需字段。
- 让旧的合法请求产生不同模型语义。
- 通过新增“优化”绕过 numeric profile 或 trace。

兼容性细则：

- 请求读取器拒绝未知字段；事件、trace 和元数据读取器应保留或忽略可安全忽略的未知 optional 字段。
- 未知 op、数值模式、量化格式和关键能力不能忽略。
- 删除字段时保留编号；函数表和 enum 的旧位置不能重用。
- writer 按 capability 协商降级；不能只比较 OS 版本字符串。
- 破坏语义时创建 `onyx.*.v2`，提供显式转换器；v1 读取器遇到 v2 必须明确拒绝。

在 M0 完成前，这些定义仍需经过真实 importer、job replay 和一个 Metal kernel 的纸面/原型验证。冻结之后，Loop 可以自动修改实现，但协议变更必须走独立设计评审，不能由优化 Loop 自行批准。

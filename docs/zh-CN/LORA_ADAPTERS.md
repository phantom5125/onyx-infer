# LoRA Adapter 支持：深度分析与设计草案

状态：**草案，未评审**。本文档是协议变更的前置设计输入（见 `docs/PROTOCOLS.md` 的变更规则），
不改变 v1 已冻结协议的任何声明。

## 0. 结论先行

1. 采纳**运行时低秩分支**（行业主路径：vLLM / llama.cpp / ComfyUI / diffusers 均如此），LoRA 永远旁路计算，
   不合并进 W4 量化权重；`fuse` 仅作为非量化模型的可选工具，v1 不做。
2. v1 支持矩阵：plain Linear LoRA + DoRA + rsLoRA，safetensors 输入，
   key 归一化覆盖 PEFT/diffusers、kohya、ComfyUI generic 三类约定；LyCORIS（LoHa/LoKr/OFT）显式拒载。
3. 性能代价可忽略：额外 flops 比例 = r/K（<0.25%），kernel launch 开销通过 epilogue 融合消除。
   与 simdgroup MMA Q4 GEMM 优化正交，新 kernel 直接内置 LoRA epilogue。
4. 关键新事实：**ComfyUI 上游已内置 MiniMax-H3 的 LoRA key 映射**
   （`comfy/lora.py` 的 MiniMax-H3 分支：裸键直映 + PDD 多头 final_layer 的 head-0 切片 patch）。
   社区 H3 LoRA 会以 "裸路径 kohya/generic" 约定流出，我们的归一化表必须首先覆盖它。

## 1. H3 目标模型的适配点盘点

| 组件 | 结构 | 适配点（均为 2D linear，均已 W4 group-64 量化） |
|---|---|---|
| DiT（50 层） | hidden 5376 | `attn.qkv_proj`（**fused**，21504 列）、`attn.out_proj`、`mlp.fc1`（→14336）、`mlp.fc2` |
| DiT 全局 | — | `time_embedder.proj_in/out`、各层 `adaln_proj`、`final_layer`、`video_patch_proj` |
| Text encoder（Qwen3-VL，64 层） | hidden 5120 | **split** `q_proj/k_proj/v_proj/o_proj`、`gate/up/down_proj`（intermediate 25600） |

关键事实：

- `src/onyx/quantize.py` 的 `LINEAR_SUFFIXES` 与 LoRA 典型 target 集合**恰好重合**；
  AdaLN/embed/norm/VAE 本来就在 `high_precision_exceptions` 中保持高精度。
  因此 LoRA 分支只需覆盖 W4 linear 的 epilogue：
  `out_fp16 = Q4GEMM(x) + Σᵢ scaleᵢ · (x·Aᵢ)·Bᵢ`，激活域 FP16/FP32，无量化交互。
- `runtime/native/dit.mm` 以唯一 string key 查表调用 linear（`session.linear(store, prefix + ".qkv_proj", x)`），
  天然存在适配 hook：module_path → adapter 绑定。
- `weight_store.mm` 为 pread + 可选常驻 cache；LoRA 权重极小（r=32 时单 linear 约 0.4 MB，
  全量 adapter 预估 <100 MB），强制 `MEMORY_SPACE_DEVICE` 常驻，复用现有
  `ResourcePhase.resident_buffers`，无需新驻留策略。
- DiT QKV 是 fused（21504 列），text encoder 是 split——**fused/split 桥接是必需能力**（见 §4.3）。

## 2. 性能分析

### 2.1 计算量

LoRA 分支两次小 GEMM 共约 `2·M·N·r`，与 base 的 `2·M·N·K` 之比为 **r/K**：

| Case | base（tile32 实测） | r=8 | r=32 |
|---|---|---|---|
| DiT qkv 1024×21504×5376 | 65.6 ms | ~0.05% flops | ~0.19% flops |
| text qproj 69×8192×5120 | 2.44 ms | ~0.06% | ~0.25% |

即使小 GEMM 有效算力只有 base 的 1/10，单层开销 1–3%，r=8 时 <0.5%。

### 2.2 launch 开销与融合策略

DiT 50 层 × 4 个适配点 = 200 个潜在分支；未融合时每层 +2 次 launch（各 5–20 µs）。
对策：`add` 融合进 base GEMM epilogue（`out = base_out + (x·A)·B·scale`），单 launch。
LoRA A/B 全模型共享 rank，可打包成 batched GEMM——与规划中的 skinny-GEMM variant 是同一批 kernel。

### 2.3 与 base kernel 优化的关系

base GEMM 当前距 MLX ~12×，但这是 base kernel 的问题；LoRA 分支在已有开销上叠加 <3%，
比例在 MMA 版 kernel 落地后不变。**排序：先 simdgroup MMA Q4 GEMM，LoRA epilogue 设计进新 kernel。**

## 3. 生态格式调研结论

### 3.1 主流引擎的共同架构

所有框架都是同一分层：**引擎只认一种内部表示，加载层做归一化**。

- vLLM：`PEFTHelper` 校验 → `LoRALayerWeights` → punica SGMV kernel；
  `optimize()` 在装载时把 `scaling = α/r`（rsLoRA 为 `α/√r`）**折进 lora_B**，运行期 kernel 无 scale 参数。
- llama.cpp：GGUF lora 仅接受 `*.lora_a/*.lora_b` + `adapter.lora.alpha`，
  其余后缀直接抛异常；运行期 `cur = base_out + (α/r)·lora_b@(lora_a@cur)`，LoRA 保持 f16/f32。
- ComfyUI：key 归一化表（`model_lora_keys_unet`）→ 格式注册表解析 → patch 列表顺序累加，
  `calculate_weight` 中间量固定 fp32。
- diffusers：读 safetensors metadata 的 `peft.lora_metadata`（内嵌 LoraConfig JSON）优先于推断；
  rank/alpha 用众数 + `rank_pattern/alpha_pattern`；kohya 无 alpha 键时约定 **α=r（scale=1）**。

### 3.2 格式与变体清单

**存储/键约定**：
- PEFT：`{module}.lora_A.weight [r, in]`、`.lora_B.weight [out, r]`、DoRA 另有
  `.lora_magnitude_vector.weight [out]`；目录配 `adapter_config.json`。
- kohya：`lora_down/lora_up` + `{module}.alpha`，前缀 `lora_unet_`（下划线化）/裸路径/`transformer.`。
- 其他：lyCORIS（LoHa `hada_w1_a…`、LoKr `lokr_w1…`、OFT `oft_blocks`）、GGUF lora、完全合并 checkpoint。

**算法变体**：plain LoRA、LoCon（作用于 linear 的同构 LoRA，可直接接受）、DoRA
（`W' = scale·(W+ΔW)/‖W+ΔW‖`，norm 需在合并权重上算——量化模型上每步旁路需读 base 权重做归一，
v1 不支持旁路 DoRA，见 §4.5）、rsLoRA（仅 scale 函数不同，免费支持）、LoHa/LoKr/OFT（拒载）。

**fused vs split QKV**：社区两种都训且互不兼容（FLUX 的 `img_attn.qkv` fused 单矩阵 vs
diffusers 的 `to_q/to_k/to_v` 分裂）。ComfyUI 的通用解法是**行切片 patch 原语** `(key, (offset, size))`：
fused `linear1_qkv` LoRA 映射到已融合权重的行区间，反之 split 键以三条 `(0, i*third, third)` 切片挂上。
MiniMax-H3 的 PDD 多头 final_layer 同理（head-0 基址切片）。

### 3.3 数值交互规范

1. LoRA 张量保持 fp16；shrink（`x·Aᵀ`）输出与 expand 累加用 **fp32** 缓冲（防大 α 时 fp16 溢出）；
2. base Q4 GEMM 本来就是 fp32 累加，两路在高精度激活域相加——无量化交互；
3. scale 折进 lora_B 在 fp16 上有轻微方差损失，vLLM 接受此损失换 kernel 简洁，我们跟随；
4. 先 fuse 后量化不可逆（diffusers/vLLM 均拒绝），我们不做 fuse。

## 4. 设计方案

### 4.1 分层架构（对齐现有 importer → package → plan → runtime 管线）

```
adapter 文件（safetensors，任意社区约定）
  → AdapterImporter（Python，src/onyx/adapter.py 新增）
      1. 格式探测：lora_A/lora_B（PEFT）| lora_down/lora_up（kohya/generic）
      2. key 归一化：config/lora_key_map/h3.json（见 §4.3）→ 内部 module_path
      3. 校验白名单（§4.5），违规显式报错
      4. scale 折算：strength × α/r（rsLoRA：α/√r）折进 lora_B
  → AdapterManifest（版本化 protobuf，扩展 model.proto；独立 artifact + sha256，不进 W4 打包）
  → adapter.onxpack（fp16 张量 + manifest；典型 <100 MB，全量常驻 DEVICE）
  → plan：AdapterBinding（module_path → adapter 引用列表）+ 新 KernelVariant
  → runtime：q4_gemm_lora_fused epilogue
```

复用现有管线模式：safetensors 读取（`src/onyx/safetensors.py`）、artifact sha256 校验
（`src/onyx/artifacts.py`）、manifest 结构（`src/onyx/w4_package.py`）。

### 4.2 schema 影响（走独立协议评审）

- `model.proto`：新增 `AdapterManifest`（adapter_id、base_model_digest、rank/alpha、scale_fn、
  tensor 描述、source_format）；manifest 的 `semantic_digest` 覆盖 adapter 集合。
- `plan.proto`：新增 `AdapterBinding`；`KernelVariant` 增加 `q4_gemm_lora_fused` 与
  `lora_skinny_gemm` variant；`MemorySpace` 复用现有枚举。
- 按 `docs/PROTOCOLS.md`，冻结后的协议变更必须独立设计评审，不由优化 Loop 批准。

### 4.3 key 归一化表

`config/lora_key_map/h3.json`：条目为 `{source_pattern → (module_path, offset, size)}`，
offset/size 支持行切片桥接（fused↔split、PDD 多头）。首版必须覆盖：

1. **ComfyUI generic / kohya 裸路径**（社区 H3 LoRA 的主流约定，ComfyUI 已内置 H3 映射，直接对齐）；
2. PEFT/diffusers（`transformer.` 前缀，text encoder 为 `text_encoder.` 前缀的 split 键）；
3. kohya 下划线风格 `lora_transformer_…`（OneTrainer 系）。

表可热更新，模型无关（新 DiT 模型加一张 map，不动 runtime）。

### 4.4 runtime 与 kernel

- graph 构建时对被绑定的 linear 节点改用 `q4_gemm_lora_fused` variant；
  多 LoRA 叠加 = 同 module 多个 adapter，shrink 可合批（Σ Aᵢ·x 一次算），expand 按 Bᵢ 逐个或合批。
- LoRA 权重 fp16 常驻 DEVICE；中间量 fp32。
- 换 adapter 不重编译 plan：按 `max_rank` 预分配 LoRA 缓冲，热切换只做拷贝
  （借 diffusers `enable_lora_hotswap` 的思路，对 Metal 预编译图尤其关键）。
- v1 不做多请求多 adapter（SGMV 分段），单 job 单 adapter 集合即可，v2 再议。

### 4.5 v1 支持矩阵

**接受**：
- plain Linear LoRA（含 LoCon-linear），单文件 safetensors；
- PEFT + kohya + generic 三类键约定；
- `α/r` 与 rsLoRA `α/√r`；per-module `rank_pattern/alpha_pattern`；
- per-adapter 运行时 strength（多 LoRA 顺序叠加）；
- 运行时热切换（按 max_rank 预分配）。
- DoRA：仅在导入期与 base **高精度副本**合并后随 W4 打包（即走 LoRA→merge→package 路径），
  不做运行时旁路（norm 需要未量化权重）。

**显式拒载（错误信息指名原因，仿 vLLM `PEFTHelper._validate_features`）**：
- 检测到 `hada_*` / `lokr_*` / `lokr_t2` / `oft_blocks` → "unsupported LyCORIS variant"；
- `lora_bias`、非空 `modules_to_save`、`layer_replication`、PiSSA/OLoRA/CorDA/LoftQ 类
  会篡改 base 权重的初始化；
- 融合权重的 LoKr / 量化模型 fuse 请求 → 明确报错；
- GGUF lora v1 不做（张量名面向 LLM）。

**数值红线**：LoRA 不量化；fp32 中间累加；base 未量化副本仅存在于导入期（DoRA merge 用）。

### 4.6 验收路径（Loop）

1. fixture：用真实 H3 tensor 构造 r=16 fp16 adapter fixture（PEFT 与 kohya 各一份），
   CPU FP32 参照值校验 epilogue 数值（对齐现有算子 bench 的 correctness_scope 方式）；
2. 单算子 bench：q4_gemm_lora_fused vs 裸 q4_gemm（tile32）在同形状下的开销百分比，
   记入 `benchmarks/results/`；
3. 端到端：1024 teapot + 某 H3 LoRA，对照 diffusers 参照输出做功能与数值检查；
4. 回放：RunRecord 含 adapter digest，输出哈希可复现。

## 5. 路线对照（记录取舍）

- 路线 A（离线合并 + 重打包）：保留为 DoRA 的载体和快速验证路径，不是主路径。
  代价：每 adapter 重打包几十 GB、无热切换/无运行时 strength/多 LoRA 需预合并。
- 路线 B（运行时分支）：主路径，理由见 §2 与 §3.1 的行业一致性。

## 6. 待评审问题

1. `rank_pattern/alpha_pattern` 的后缀匹配语义照抄 PEFT `get_pattern_key` 还是简化为精确匹配？
2. adapter 与 base 模型版本绑定：base 更新（重新打包）后旧 adapter 是否强制拒载？
   （建议：manifest 记录 `base_model_digest`，不匹配即拒载，给出重新导入提示。）
3. shrink 合批的收益需要 bench 决定 v1 是否做；
4. text encoder 适配点是否 v1 全开（64 层 × 7 个 linear，仍是 flops 可忽略，主要影响 plan 节点数）。

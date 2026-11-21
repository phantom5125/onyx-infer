# 文字渲染质量基线

来源：MiniMax-H3-Image 官方 showcase `instruction-following.md`（喷气发动机爆炸图、moka pot 蓝图）。
目的：量化迭代（W4 格式、例外层、校准方式）的文字质量回归。

## 协议

- 分辨率 1024×1024，seed 42，各路径默认采样配置。
- 验收：prompt 中给定的中文标签逐字比对（`text-baseline-jet-engine.json` 的 `expected_labels_zh`）。
- **prompt 一律使用平引号**：中文弯引号 `“ ”` 会使社区 `h3-image-mlx` W8A8 路径输出完全无关图像（2026-11-17 实测）。JSON 内 prompt 已按平引号书写；`text-baseline-jet-engine.json` 中的弯引号版本仅作存档，执行时需替换。
- 两条路径命令：
  - Onyx W4：`uv run onyx native-generate --models artifacts/models/h3-w4 --native-bin .build/onyx-native ...`（h3-w4 可由 `h3-image-source` 重新派生）
  - 社区 W8A8：`external/h3-image-mlx/.venv/bin/h3-image-mlx generate --models artifacts/models/community-int8 --backend w8a8 ...`

## 已测结果（2026-11-17）

| 标签 | Onyx W4 | 社区 INT8 ConvRot (W8A8) |
|---|---|---|
| 进气罩 | ✓ | ✓ |
| 压气机叶片盘 | ✓ | ✓ |
| 燃烧室 | 燃燳室 | 讹误 |
| 涡轮盘 | 涡轮蛊 | 涡轮蛊（逐字相同） |
| 中心主轴 | ✓ | ✓ |
| 尾喷管 | ✓ | ✓（另有重复） |
| 机匣外壳 | ✓（重复×2） | 讹误 |
| 电子控制单元 | 电子掽制单元 | 讹字 |
| 燃油泵 | 燃畝泵+正确 | ✓ |
| 附件齿轮箱 | 齿蕗箱 | 讹字 |
| 燃油管路 | 管蕗 | ✓ |
| **精确率** | **4/11** | **约 6/11** |
| MOKA POT 标题 | 逐字完美 | — |

## 归因（2026-11-18 三方对照后修订）

| 标签 | BF16 满血 | INT8 ConvRot | Onyx W4 |
|---|---|---|---|
| 精确率 | ~6.5/11 | ~6/11 | 4/11 |
| 特征 | 繁简混排（尾噴管）、凭空标签（嵌片\*\*壳） | 与 W4 逐字相同的"涡轮蛊" | 形近字讹误最多 |

- **主因是裸 t2i 链路本身**（无 prompt expansion、6 步无 CFG）：满血 BF16 也只有 ~6.5/11，
  出现繁简混排与凭空标签。官方 showcase 的逐字清晰依赖完整 pipeline。
- **量化是次级叠加退化**（6.5 → 6 → 4 单调递减）："涡轮盘→蛊" 在两条量化路径上一致、
  BF16 上消失，确认该类损伤来自量化共享环节；W4 有小幅额外损失。
- 行动优先级：补 prompt expansion 链路（收益最大，也可放在 ComfyUI 节点层）＞ W4 混合精度改进。
- BF16 对照运行：`--precision bf16 --dit-mode stream`，权重为 Comfy-Org pruned BF16
  （AdaLN 剪枝保留），`artifacts/generated/text-baseline-jet-engine-bf16-1024.png`。

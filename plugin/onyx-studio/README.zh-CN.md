# Onyx Studio（DSH 插件）

把本地推理引擎 onyx-infer 做成 DeepSeek Harness 插件：一面服务研发，一面出图，还带一个本地图文阅读器。

- **研发**：运行瓶颈（阶段瀑布、同用例跨 plan 对比）、效率（算子耗时与 MLX 对照、
  TFLOPS / 带宽）、模型结构（组件 / DiT 图 / 量化 / 精度分布）、硬件利用率
  （内存曲线、DiT 每步 Metal 显存、目标硬件规格）。
- **出图**：提示词驱动的完整原生链路生成，实时进度与可复现 RunRecord。
- **书架（图文阅读器）**：导入整本书（TXT / Markdown / EPUB），自动分章分页，
  阅读时一页文字、一页插图；插图由本地引擎按页生成，也可交给智能体批量配图。

本目录即 DSH bundle 包 `@local/onyx-studio`。安装方式：DeepSeek Harness →
Plugins → Add plugin → 选择本目录（或让智能体执行 install_bundle）。

在生成结果下方打开「精确文字标注」，可编辑文字框和引线。渲染后的 PNG 和排版文件会随该次运行归档。[排版格式和命令行示例](../../docs/IMAGE_TEXT.md)

## 文件

- `index.js`：Host 半边。注册 `/onyx-studio/` HTTP 路由与 `onyx_generate` /
  `onyx_analyze` / `onyx_book` 工具；生成任务通过引擎 venv 的 `python -m onyx.replay`
  执行，`sampler.py` 提供 100 ms 级进程采样（阶段探测 + RSS 时间序列）。
- `client.js`：浏览器半边。整页工作台（`main` / `conversation.view`）：生成 / 书架 /
  运行 / 模型 / 算子；左侧栏面板入口（`sidebar.panellist`）；右侧栏标签页
  （`sidebar.right.pane.tab`）。
- `book_import.py`：书籍导入管线（纯标准库）。GBK/UTF-8 双编码、章回切分、
  硬换行合并、广告行过滤、EPUB spine 解析、按字数分页、逐页插图提示词生成。
- 零第三方依赖：Host 侧只用 Node 内建模块，浏览器侧只依赖页面提供的 React。

## 书架与阅读器

- 存储：引擎根 `artifacts/studio/books/<bookId>/`，内含 `book.json`（元信息 +
  进度 + 排版设置）、`pages.json`（章节 + 分页 + 每页插图槽位）、`source.txt`、
  `images/p-XXXX.png`。目录位于 `/onyx-studio/files` 已允许的 artifacts 子树内，
  插图直接由既有文件路由提供。
- 导入：浏览器 FileReader 本地读取文件 → base64 → `POST /api/books/import`；
  Host 调引擎 venv 的 `book_import.py` 完成解析与分页。TXT 自动尝试 UTF-8/GB18030、
  合并硬换行、过滤常见站点水印行、按 `第X回/章/卷` 等正则切章；MD 按标题切；
  EPUB 按 spine 抽正文。每页默认 500–1400 字可调（默认 800），不拆段落。
- 插图：每页在导入时自动生成一份提示词（章节 + 启发式场景句 + 风格模板：
  古典水墨 / 写实电影感 / 插画绘本 / 极简图形）。阅读器在翻到插图页时可手动
  生成/重绘/改提示词，也可开启「自动预生成」——引擎空闲时按当前进度向后串行
  预生成（引擎同一时刻只跑一张，约 40s / 1024px）。生成完成后 PNG 拷入书目录
  并回写 `pages.json`，运行记录仍归档在 `artifacts/benchmarks/onyx-native-h3/`。
- 智能体工具 `onyx_book`：`list` / `import`（给路径即可）/ `page`（读分页）/
  `illustrate`（配单页插图，串行队列）/ `status`。可以直接说
  「给《西游记》第一回到第五回配插图」，由智能体逐页排队完成。
- 快捷键：`←` `→` 空格翻页，`Esc` 关闭目录/设置/提示词编辑。

## 本地检查

```bash
node --check index.js
node --check client.js          # client.js 顶部有 window.__ModuleLoader__，语法检查仍有效
python3 -m py_compile sampler.py book_import.py
python3 tests/run_book_import_test.py   # 导入器 fixture 测试（TXT/MD/EPUB/坏档）
node e2e.mjs                          # 需要构建引擎和 W4 模型包，创建一条真实出图记录
node smoke.mjs                        # 验证已有出图记录和书架路由
node e2e-book.mjs                     # 导入测试文本并为第一页生成插图
```

测试默认使用插件所在的引擎工作树。可设置 `ONYX_ENGINE_ROOT` 指向另一份已构建的工作树。

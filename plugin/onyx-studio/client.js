/**
 * Browser half of the Onyx Studio plugin.
 *
 * Registers a full-page conversation view, a sidebar panel entry, and a
 * right-sidebar tab. All data comes from the host half's same-origin HTTP
 * surface; rendering is plain React against the host theme tokens.
 */
window.__ModuleLoader__.load({
  id: '@local/onyx-studio',
  factory(require) {
    const React = require('react')
    const { useState, useEffect, useMemo, useRef, useCallback } = React
    const h = React.createElement

    const API = '/onyx-studio/api'
    const FILES = '/onyx-studio/files?path='
    const NS = 'onyxStudio'
    const TAB_ID = '@local/onyx-studio'

    /* ----------------------------------------------------------- locale */
    const dict = {
      zh: {
        'view.title': 'Onyx Studio',
        'view.subtitle': '本地图像生成引擎工作台',
        'tab.title': 'Onyx 出图',
        'tab.guide.title': 'Onyx 出图 · 本地引擎',
        'tab.guide.description': '生成图像，查看运行瓶颈、模型结构与硬件利用率。',
        'nav.generate': '生成',
        'nav.runs': '运行',
        'nav.model': '模型',
        'nav.operators': '算子',
        'common.retry': '重试',
        'common.refresh': '刷新',
        'common.loading': '加载中…',
        'common.error': '出错了',
        'common.empty': '暂无数据',
        'common.open': '打开原图',
        'generate.prompt': '提示词',
        'generate.promptPlaceholder': '描述要生成的图像…（英文提示词效果最好）',
        'generate.size': '尺寸',
        'generate.seed': '种子',
        'generate.start': '开始生成',
        'generate.cancel': '取消',
        'generate.running': '生成中',
        'generate.elapsed': '已用时',
        'generate.memory': '进程内存',
        'generate.log': '引擎输出',
        'generate.result': '生成结果',
        'generate.viewRun': '查看运行记录',
        'generate.recent': '最近的 Studio 运行',
        'generate.estimate': '预计总时长按近期运行中位数估计',
        'generate.waitJob': '引擎正忙：已有生成任务在运行',
        'generate.stylePresets': '风格',
        'generate.regenerateSame': '同种子重绘',
        'generate.regenerateNew': '换一种子',
        'generate.labels': '精确文字标注',
        'generate.labelLayout': '标注排版 JSON',
        'generate.labelExample': '标题',
        'generate.renderLabels': '渲染标注',
        'generate.renderingLabels': '渲染标注中…',
        'generate.openLabels': '打开标注图',
        'stage.text': '文本编码',
        'stage.dit': 'DiT 采样',
        'stage.vae': 'VAE 解码',
        'run.total': '总耗时',
        'runs.count': '{count} 次运行',
        'run.peakRss': '峰值内存',
        'run.swap': 'Swap 用量（前 → 后）',
        'run.tokens': '文本 token',
        'run.samples': '内存采样数',
        'run.waterfall': '阶段瀑布',
        'run.ditSteps': 'DiT 每步 Metal 显存',
        'run.build': '构建',
        'run.device': '设备',
        'run.output': '输出',
        'run.sameCase': '同用例运行对比',
        'run.memoryHint': '内存时间序列仅对 Studio 启动的运行实时记录',
        'model.target': '目标硬件',
        'model.pipeline': '流水线组件',
        'model.dit': 'DiT 结构',
        'model.dtypes': '权重精度分布',
        'model.quant': '量化规则',
        'model.recipe': '采样配方',
        'model.numeric': '数值规格',
        'model.graphNodes': '{count} 个节点',
        'model.blockNodes': '{count} 个变换块',
        'model.totalParams': '参数量约 {value}',
        'model.statParams': '参数量',
        'model.statTensors': '张量',
        'model.statSize': '权重体积',
        'ops.compare': 'Q4 算子对照（Onyx vs MLX）',
        'ops.raw': '原生基准（TFLOPS / 带宽）',
        'ops.case': '用例',
        'ops.shape': '形状',
        'ops.onyx': 'Onyx 墙钟',
        'ops.mlx': 'MLX 墙钟',
        'ops.ratio': '倍率',
        'ops.correct': '正确',
        'ops.tflops': '有效 TFLOPS',
        'ops.bandwidth': '有效带宽',
        'status.idle': '引擎空闲',
        'status.busy': '引擎工作中',
        'nav.books': '书架',
        'book.import': '导入书籍',
        'book.cancelImport': '取消',
        'book.count': '{count} 本',
        'book.importFile': '选择文件（TXT / Markdown / EPUB）',
        'book.importHint': '整本 TXT 自动识别章回与硬换行；论文与课本建议 Markdown 或 EPUB。文件在浏览器本地读取后交给本机引擎。',
        'book.title': '书名',
        'book.author': '作者',
        'book.style': '插图风格',
        'book.style.ink': '古典水墨',
        'book.style.cinematic': '写实电影感',
        'book.style.storybook': '插画绘本',
        'book.style.minimal': '极简图形',
        'book.styleDesc.ink': '留白水墨，适合四大名著',
        'book.styleDesc.cinematic': '写实光影，适合历史与冒险',
        'book.styleDesc.storybook': '柔和手绘，适合轻松读物',
        'book.styleDesc.minimal': '几何图形，适合教材笔记',
        'book.charsPerPage': '每页字数',
        'book.importing': '导入中…',
        'book.empty': '书架还是空的，导入一本开始阅读',
        'book.open': '继续阅读',
        'book.readFromStart': '从头阅读',
        'book.progress': '阅读进度',
        'book.illustrated': '已配图',
        'book.pages': '页',
        'book.delete': '删除',
        'book.deleteConfirm': '删除这本书？正文分页、插图与阅读进度都会一并删除。',
        'book.toc': '目录',
        'book.settings': '排版设置',
        'book.theme': '主题',
        'book.theme.paper': '纸',
        'book.theme.night': '夜',
        'book.theme.ink': '墨',
        'book.font': '字体',
        'book.font.song': '宋体',
        'book.font.kai': '楷体',
        'book.fontSize': '字号',
        'book.lineHeight': '行距',
        'book.autoIllustrate': '自动预生成插图',
        'book.aheadCount': '预生成页数',
        'book.illustration': '插图',
        'book.generateIllust': '生成本页插图',
        'book.regenerate': '重新绘制',
        'book.editPrompt': '编辑提示词',
        'book.generating': '插图生成中…',
        'book.queued': '引擎忙，稍后自动重试',
        'book.promptLabel': '插图提示词',
        'book.applyPrompt': '保存并重绘',
        'book.pageOf': '{a} / {b}',
        'book.chapter': '章节',
        'book.prev': '上一页',
        'book.next': '下一页',
        'book.tooLarge': '文件超过 48 MB：请先拆分、转 EPUB 或压缩图片后再导入',
        'book.pagerHint': '← → 或空格翻页',
        'book.illustFailed': '插图生成失败',
        'book.backToShelf': '返回书架',
        'book.position': '第 {page} 页 · {chapter}',
        'book.text': '文字',
        'book.seed': '种子',
      },
      en: {
        'view.title': 'Onyx Studio',
        'view.subtitle': 'Local image-generation engine workbench',
        'tab.title': 'Onyx generate',
        'tab.guide.title': 'Onyx generate · local engine',
        'tab.guide.description': 'Generate images and inspect run bottlenecks, model structure, and hardware utilization.',
        'nav.generate': 'Generate',
        'nav.runs': 'Runs',
        'nav.model': 'Model',
        'nav.operators': 'Operators',
        'common.retry': 'Retry',
        'common.refresh': 'Refresh',
        'common.loading': 'Loading…',
        'common.error': 'Something went wrong',
        'common.empty': 'No data yet',
        'common.open': 'Open image',
        'generate.prompt': 'Prompt',
        'generate.promptPlaceholder': 'Describe the image to generate… (English prompts work best)',
        'generate.size': 'Size',
        'generate.seed': 'Seed',
        'generate.start': 'Generate',
        'generate.cancel': 'Cancel',
        'generate.running': 'Generating',
        'generate.elapsed': 'Elapsed',
        'generate.memory': 'Process memory',
        'generate.log': 'Engine output',
        'generate.result': 'Result',
        'generate.viewRun': 'View run record',
        'generate.recent': 'Recent Studio runs',
        'generate.estimate': 'Total estimate uses the median of recent runs',
        'generate.waitJob': 'Engine busy: another generation is running',
        'generate.stylePresets': 'Style',
        'generate.regenerateSame': 'Redraw (same seed)',
        'generate.regenerateNew': 'New seed',
        'generate.labels': 'Exact text labels',
        'generate.labelLayout': 'Label layout JSON',
        'generate.labelExample': 'Title',
        'generate.renderLabels': 'Render labels',
        'generate.renderingLabels': 'Rendering labels…',
        'generate.openLabels': 'Open labeled image',
        'stage.text': 'Text encode',
        'stage.dit': 'DiT sampling',
        'stage.vae': 'VAE decode',
        'run.total': 'Total',
        'runs.count': '{count} runs',
        'run.peakRss': 'Peak RSS',
        'run.swap': 'Swap used (before → after)',
        'run.tokens': 'Text tokens',
        'run.samples': 'Memory samples',
        'run.waterfall': 'Stage waterfall',
        'run.ditSteps': 'Metal allocation per DiT step',
        'run.build': 'Build',
        'run.device': 'Device',
        'run.output': 'Output',
        'run.sameCase': 'Same case, across runs',
        'run.memoryHint': 'Live memory series are recorded only for Studio-started runs',
        'model.target': 'Target hardware',
        'model.pipeline': 'Pipeline components',
        'model.dit': 'DiT structure',
        'model.dtypes': 'Weight precision mix',
        'model.quant': 'Quantization rules',
        'model.recipe': 'Sampling recipe',
        'model.numeric': 'Numeric profile',
        'model.graphNodes': '{count} nodes',
        'model.blockNodes': '{count} transform blocks',
        'model.totalParams': '≈ {value} parameters',
        'model.statParams': 'Parameters',
        'model.statTensors': 'Tensors',
        'model.statSize': 'Weight size',
        'ops.compare': 'Q4 operator comparison (Onyx vs MLX)',
        'ops.raw': 'Native benchmarks (TFLOPS / bandwidth)',
        'ops.case': 'Case',
        'ops.shape': 'Shape',
        'ops.onyx': 'Onyx wall',
        'ops.mlx': 'MLX wall',
        'ops.ratio': 'Ratio',
        'ops.correct': 'Correct',
        'ops.tflops': 'Effective TFLOPS',
        'ops.bandwidth': 'Effective bandwidth',
        'status.idle': 'Engine idle',
        'status.busy': 'Engine busy',
        'nav.books': 'Library',
        'book.import': 'Import book',
        'book.cancelImport': 'Cancel',
        'book.count': '{count} books',
        'book.importFile': 'Choose a file (TXT / Markdown / EPUB)',
        'book.importHint': 'Full-length TXT files get chapter detection and hard-wrap merging; papers and textbooks work best as Markdown or EPUB. The file is read locally in the browser, then handled by the local engine.',
        'book.title': 'Title',
        'book.author': 'Author',
        'book.style': 'Illustration style',
        'book.style.ink': 'Classical ink wash',
        'book.style.cinematic': 'Cinematic realism',
        'book.style.storybook': 'Storybook illustration',
        'book.style.minimal': 'Minimal graphic',
        'book.styleDesc.ink': 'Negative space ink wash; great for the classics',
        'book.styleDesc.cinematic': 'Realistic lighting for history and adventure',
        'book.styleDesc.storybook': 'Soft hand-drawn, for lighter reads',
        'book.styleDesc.minimal': 'Clean geometry for notes and textbooks',
        'book.charsPerPage': 'Characters per page',
        'book.importing': 'Importing…',
        'book.empty': 'The shelf is empty — import a book to start reading',
        'book.open': 'Continue reading',
        'book.readFromStart': 'Read from the start',
        'book.progress': 'Progress',
        'book.illustrated': 'Illustrated',
        'book.pages': 'pages',
        'book.delete': 'Delete',
        'book.deleteConfirm': 'Delete this book? Pages, illustrations, and reading progress are removed together.',
        'book.toc': 'Contents',
        'book.settings': 'Typography',
        'book.theme': 'Theme',
        'book.theme.paper': 'Paper',
        'book.theme.night': 'Night',
        'book.theme.ink': 'Ink',
        'book.font': 'Font',
        'book.font.song': 'Song',
        'book.font.kai': 'Kai',
        'book.fontSize': 'Size',
        'book.lineHeight': 'Line height',
        'book.autoIllustrate': 'Auto-generate illustrations',
        'book.aheadCount': 'Pages to pre-generate',
        'book.illustration': 'Illustration',
        'book.generateIllust': 'Illustrate this page',
        'book.regenerate': 'Redraw',
        'book.editPrompt': 'Edit prompt',
        'book.generating': 'Rendering illustration…',
        'book.queued': 'Engine busy; will retry',
        'book.promptLabel': 'Illustration prompt',
        'book.applyPrompt': 'Save and redraw',
        'book.pageOf': '{a} / {b}',
        'book.chapter': 'Chapter',
        'book.prev': 'Previous',
        'book.next': 'Next',
        'book.tooLarge': 'File exceeds 48 MB — split it, convert to EPUB, or compress images first',
        'book.pagerHint': '← → or space to turn pages',
        'book.illustFailed': 'Illustration failed',
        'book.backToShelf': 'Back to shelf',
        'book.position': 'Page {page} · {chapter}',
        'book.text': 'Text',
        'book.seed': 'Seed',
      },
    }

    /* ---------------------------------------------------------- helpers */
    async function api(path, init) {
      const response = await fetch(`${API}${path}`, {
        credentials: 'same-origin',
        signal: AbortSignal.timeout(20_000),
        ...init,
      })
      let value = null
      try { value = await response.json() } catch { /* non-json error */ }
      if (!response.ok) throw new Error(value?.error ?? `HTTP ${response.status}`)
      return value
    }

    const fmt = {
      gb(bytes) { return bytes > 0 ? `${(bytes / 2 ** 30).toFixed(2)} GB` : '—' },
      mb(bytes) { return bytes > 0 ? `${(bytes / 2 ** 20).toFixed(0)} MB` : '—' },
      sec(seconds) {
        if (seconds == null || !Number.isFinite(seconds)) return '—'
        return seconds >= 90 ? `${Math.floor(seconds / 60)}m${String(Math.round(seconds % 60)).padStart(2, '0')}s` : `${seconds.toFixed(1)}s`
      },
      ms(milliseconds) { return milliseconds == null ? '—' : `${Number(milliseconds).toFixed(1)}ms` },
      num(value, digits = 2) { return value == null ? '—' : Number(value).toFixed(digits) },
      short(text, size = 10) { return text == null ? '—' : String(text).slice(0, size) },
      time(ms) {
        try { return new Date(ms).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) }
        catch { return '—' }
      },
    }

    function StageName({ stage, t }) {
      const key = stage != null && dict.en[`stage.${stage}`] !== undefined ? `stage.${stage}` : null
      return h('span', null, key === null ? (stage ?? '—') : t(key))
    }

    /* ------------------------------------------------------------ hooks */
    function useInterval(callback, intervalMs, enabled) {
      const ref = useRef(callback)
      ref.current = callback
      useEffect(() => {
        if (!enabled) return undefined
        const timer = setInterval(() => ref.current(), intervalMs)
        return () => clearInterval(timer)
      }, [intervalMs, enabled])
    }

    function useApi(path, { intervalMs = 0 } = {}) {
      const [data, setData] = useState(null)
      const [error, setError] = useState(null)
      const [loading, setLoading] = useState(path != null)
      const [nonce, setNonce] = useState(0)
      const lastPath = useRef(path)
      useEffect(() => {
        if (path == null) { setData(null); setError(null); setLoading(false); return undefined }
        let live = true
        // A different path is a different query: drop the previous answer so
        // the view never shows one run's numbers under another run's title.
        if (lastPath.current !== path) { lastPath.current = path; setData(null); setError(null); setLoading(true) }
        api(path).then(
          (value) => { if (live) { setData(value); setError(null); setLoading(false) } },
          (reason) => { if (live) { setError(reason); setLoading(false) } },
        )
        return () => { live = false }
      }, [path, nonce])
      useInterval(() => {
        if (path == null) return
        api(path).then(
          (value) => { setData(value); setError(null) },
          (reason) => setError(reason),
        )
      }, intervalMs, intervalMs > 0)
      const reload = useCallback(() => setNonce((n) => n + 1), [])
      return { data, error, loading, reload }
    }

    /* ------------------------------------------------------- primitives */
    /**
     * Load an image through the page's fetch (same-origin credentials, or the
     * Desktop app's IPC fetch bridge) and expose it as a blob URL. Raw <img
     * src> requests from surfaces without this origin's session cookie are
     * rejected by the Host/Origin fence, so images always travel via fetch.
     */
    function useBlobUrl(url) {
      const [objectUrl, setObjectUrl] = useState(null)
      const [error, setError] = useState(null)
      useEffect(() => {
        if (url == null) return undefined
        let live = true
        let created = null
        fetch(url, { credentials: 'same-origin', signal: AbortSignal.timeout(60_000) })
          .then(async (response) => {
            if (!response.ok) {
              const text = await response.text().catch(() => '')
              throw new Error(`HTTP ${response.status} ${text.slice(0, 160)}`)
            }
            return response.blob()
          })
          .then((blob) => {
            if (!live) return
            created = URL.createObjectURL(blob)
            setObjectUrl(created)
          })
          .catch((reason) => { if (live) setError(reason instanceof Error ? reason : new Error(String(reason))) })
        return () => { live = false ; setObjectUrl(null) ; setError(null) ; if (created !== null) URL.revokeObjectURL(created) }
      }, [url])
      return { objectUrl, error }
    }

    /** Image that loads via fetch and shows the failure reason in place. */
    function FetchImage({ src, alt, style, t }) {
      const { objectUrl, error } = useBlobUrl(src)
      if (error !== null) {
        return h('div', { className: 'onyx-warn', role: 'alert', style: { fontSize: 11 } },
          `${t != null ? t('common.error') : 'error'}: ${error.message}`)
      }
      if (objectUrl === null) return h('div', { className: 'onyx-empty' }, t != null ? t('common.loading') : '…')
      return h('img', { className: 'onyx-img', src: objectUrl, alt, style })
    }

    /** Open the fetched blob (new tab / download) instead of a fenced raw URL. */
    function OpenImageLink({ src, label }) {
      const { objectUrl } = useBlobUrl(src)
      return h('a', {
        className: 'onyx-btn',
        href: objectUrl ?? undefined,
        target: '_blank', rel: 'noreferrer',
        'aria-disabled': objectUrl === null,
        style: objectUrl === null ? { opacity: 0.5, pointerEvents: 'none' } : undefined,
      }, label)
    }

    const CSS = `
.onyx-root{--ox-radius:12px;--ox-radius-s:8px;--ox-shadow:0 1px 2px rgb(0 0 0/8%),0 2px 10px rgb(0 0 0/5%);--ox-accent:var(--dsw-alias-brand-primary);--ox-accent2:#8a5cf6;--ox-accent-soft:rgba(79,143,214,.12);display:flex;flex-direction:column;width:100%;height:100%;min-height:0;background:var(--dsw-alias-bg-base);color:var(--dsw-alias-label-primary);font-size:13px}
.onyx-head{display:flex;align-items:center;gap:12px;padding:12px 16px;border-bottom:1px solid var(--dsw-alias-border-l1);flex:none;flex-wrap:wrap}
.onyx-brand{display:flex;align-items:center;gap:10px;min-width:0}
.onyx-logoChip{width:32px;height:32px;border-radius:9px;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,var(--ox-accent),var(--ox-accent2));box-shadow:0 2px 8px rgb(0 0 0/18%);flex:none}
.onyx-title{font-weight:700;font-size:15px;letter-spacing:.01em}
.onyx-sub{color:var(--dsw-alias-label-secondary);font-size:11px;margin-top:1px}
.onyx-status{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--dsw-alias-border-l1);border-radius:999px;padding:3px 10px;font-size:11px;color:var(--dsw-alias-label-secondary);background:var(--dsw-alias-bg-layer-1)}
.onyx-dot{width:7px;height:7px;border-radius:999px;background:var(--dsw-alias-state-idle-primary,#8a8f98);flex:none}
.onyx-dot.is-running{background:var(--dsw-alias-state-warn-primary,#d6a35c);animation:onyx-breathe 1.6s ease-in-out infinite}
.onyx-dot.is-ok{background:var(--dsw-alias-state-success-primary,#5faf6e)}
.onyx-nav{display:flex;gap:2px;margin-left:auto;background:var(--dsw-alias-bg-layer-1);border:1px solid var(--dsw-alias-border-l1);border-radius:10px;padding:3px}
.onyx-navBtn{border:0;background:transparent;color:var(--dsw-alias-label-secondary);padding:5px 12px;border-radius:7px;cursor:pointer;font-size:12px;transition:color .15s,background .15s}
.onyx-navBtn:hover{color:var(--dsw-alias-label-primary)}
.onyx-navBtn[aria-selected=true]{background:var(--dsw-alias-bg-base);color:var(--dsw-alias-label-primary);font-weight:600;box-shadow:0 1px 3px rgb(0 0 0/14%)}
.onyx-body{flex:1;min-height:0;overflow:auto;padding:16px 18px;scroll-behavior:smooth}
.onyx-body.is-compact{padding:10px}
.onyx-view{animation:onyx-fade .18s ease}
.onyx-section{margin-bottom:18px}
.onyx-sectionTitle{font-size:11px;font-weight:600;color:var(--dsw-alias-label-secondary);margin:0 0 8px;text-transform:uppercase;letter-spacing:.06em;display:flex;align-items:center;gap:8px}
.onyx-sectionTitle::before{content:'';width:3px;height:12px;border-radius:2px;background:linear-gradient(180deg,var(--ox-accent),var(--ox-accent2))}
.onyx-card{background:var(--dsw-alias-bg-layer-1);border:1px solid var(--dsw-alias-border-l1);border-radius:var(--ox-radius);padding:12px 14px}
.onyx-card.is-hover{transition:transform .15s,box-shadow .15s}
.onyx-card.is-hover:hover{transform:translateY(-1px);box-shadow:var(--ox-shadow)}
.onyx-grid{display:grid;gap:8px}
.onyx-stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px;margin-bottom:12px}
.onyx-stat{background:var(--dsw-alias-bg-layer-1);border:1px solid var(--dsw-alias-border-l1);border-radius:var(--ox-radius);padding:10px 12px}
.onyx-statValue{font-size:19px;font-weight:650;font-variant-numeric:tabular-nums;letter-spacing:-.01em}
.onyx-statLabel{font-size:11px;color:var(--dsw-alias-label-secondary);margin-top:2px}
.onyx-kv{display:grid;grid-template-columns:auto 1fr;gap:2px 12px;font-size:12px}
.onyx-kv dt{color:var(--dsw-alias-label-secondary)}
.onyx-kv dd{margin:0;font-variant-numeric:tabular-nums;word-break:break-all}
.onyx-row{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.onyx-spacer{flex:1}
.onyx-btn{border:1px solid var(--dsw-alias-border-l2);background:var(--dsw-alias-bg-layer-2);color:var(--dsw-alias-label-primary);border-radius:9px;padding:6px 14px;font-size:12px;cursor:pointer;transition:filter .15s,transform .05s}
.onyx-btn:hover:not(:disabled){filter:brightness(1.08)}
.onyx-btn:active:not(:disabled){transform:translateY(1px)}
.onyx-btn:disabled{opacity:.5;cursor:not-allowed}
.onyx-btn:focus-visible,.onyx-input:focus-visible,.onyx-select:focus-visible,.onyx-textarea:focus-visible,.onyx-navBtn:focus-visible{outline:2px solid var(--ox-accent);outline-offset:1px}
.onyx-btnPrimary{border-color:transparent;background:linear-gradient(135deg,var(--ox-accent),var(--ox-accent2));color:#fff;font-weight:600}
.onyx-btnGhost{border-color:transparent;background:transparent;color:var(--dsw-alias-label-secondary)}
.onyx-btnGhost:hover{color:var(--dsw-alias-label-primary)}
.onyx-input,.onyx-select,.onyx-textarea{width:100%;border:1px solid var(--dsw-alias-border-l2);background:var(--dsw-alias-bg-base);color:var(--dsw-alias-label-primary);border-radius:9px;padding:7px 10px;font-size:12px;font-family:inherit;box-sizing:border-box;transition:border-color .15s}
.onyx-input:hover,.onyx-select:hover,.onyx-textarea:hover{border-color:var(--dsw-alias-border-l2)}
.onyx-textarea{min-height:88px;resize:vertical;line-height:1.6}
.onyx-label{font-size:11px;color:var(--dsw-alias-label-secondary);margin-bottom:4px;display:block}
.onyx-chip{display:inline-flex;align-items:center;gap:4px;border:1px solid var(--dsw-alias-border-l1);border-radius:999px;padding:2px 9px;font-size:11px;color:var(--dsw-alias-label-secondary)}
.onyx-chip.is-active{border-color:var(--ox-accent);color:var(--ox-accent);background:var(--ox-accent-soft)}
.onyx-chipBtn{cursor:pointer;background:transparent;transition:border-color .15s,color .15s}
.onyx-chipBtn:hover{border-color:var(--ox-accent);color:var(--ox-accent)}
.onyx-badge{display:inline-flex;align-items:center;border-radius:6px;padding:1px 7px;font-size:11px;font-weight:600;font-variant-numeric:tabular-nums}
.onyx-badge.is-ok{color:var(--dsw-alias-state-success-primary);background:rgba(95,175,110,.12)}
.onyx-badge.is-warn{color:var(--dsw-alias-state-warn-primary);background:rgba(214,163,92,.12)}
.onyx-badge.is-bad{color:var(--dsw-alias-state-error-primary);background:rgba(214,92,92,.12)}
.onyx-list{display:flex;flex-direction:column;gap:2px;max-height:100%;overflow:auto}
.onyx-item{border:1px solid transparent;border-radius:9px;padding:7px 9px;cursor:pointer;text-align:left;background:transparent;width:100%;display:flex;gap:10px;align-items:center}
.onyx-item:hover{background:var(--dsw-alias-bg-layer-1)}
.onyx-item.is-active{border-color:var(--ox-accent);background:var(--ox-accent-soft)}
.onyx-itemThumb{width:52px;height:52px;border-radius:7px;object-fit:cover;flex:none;border:1px solid var(--dsw-alias-border-l1);background:var(--dsw-alias-bg-layer-2)}
.onyx-itemBody{min-width:0;flex:1}
.onyx-itemTitle{font-size:12px;font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.onyx-itemMeta{font-size:11px;color:var(--dsw-alias-label-secondary);display:flex;gap:8px;margin-top:1px}
.onyx-table{width:100%;border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums}
.onyx-table th{color:var(--dsw-alias-label-secondary);font-weight:500;text-align:left;padding:5px 8px;border-bottom:1px solid var(--dsw-alias-border-l1);white-space:nowrap}
.onyx-table td{padding:5px 8px;border-bottom:1px solid var(--dsw-alias-border-l1);white-space:nowrap}
.onyx-table tr:hover td{background:var(--dsw-alias-bg-layer-1)}
.onyx-table td.wrap{white-space:normal;word-break:break-all}
.onyx-miniBar{height:6px;border-radius:4px;background:linear-gradient(90deg,var(--ox-accent),var(--ox-accent2));min-width:2px}
.onyx-pre{margin:0;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;line-height:1.5;color:var(--dsw-alias-label-secondary);white-space:pre-wrap;word-break:break-all;max-height:120px;overflow:auto;background:var(--dsw-alias-bg-layer-2);border-radius:8px;padding:8px}
.onyx-bars{display:flex;flex-direction:column;gap:3px}
.onyx-barRow{display:grid;grid-template-columns:64px 1fr 72px;align-items:center;gap:8px;font-size:11px}
.onyx-barTrack{background:var(--dsw-alias-bg-layer-2);border-radius:4px;height:10px;overflow:hidden}
.onyx-barFill{height:100%;background:linear-gradient(90deg,var(--ox-accent),var(--ox-accent2));border-radius:4px;transition:width .5s ease}
.onyx-error{border:1px solid var(--dsw-alias-state-error-primary);color:var(--dsw-alias-state-error-primary);border-radius:9px;padding:8px 10px;font-size:12px}
.onyx-warn{border:1px solid var(--dsw-alias-state-warn-primary);color:var(--dsw-alias-state-warn-primary);border-radius:9px;padding:8px 10px;font-size:12px}
.onyx-img{max-width:100%;border-radius:10px;border:1px solid var(--dsw-alias-border-l1);display:block}
.onyx-blocks{display:flex;flex-wrap:wrap;gap:2px}
.onyx-blockCell{width:10px;height:18px;border-radius:2px;background:linear-gradient(180deg,var(--ox-accent),var(--ox-accent2));opacity:.8}
.onyx-empty{color:var(--dsw-alias-label-secondary);font-size:12px;padding:12px 0}
.onyx-split{display:grid;grid-template-columns:minmax(200px,260px) 1fr;gap:14px;min-height:0}
.onyx-split.is-compact{grid-template-columns:1fr}
.onyx-skeleton{border-radius:9px;background:linear-gradient(100deg,var(--dsw-alias-bg-layer-1) 30%,var(--dsw-alias-bg-layer-2) 50%,var(--dsw-alias-bg-layer-1) 70%);background-size:200% 100%;animation:onyx-shimmer 1.2s linear infinite}
.onyx-steps{display:flex;gap:6px;align-items:center;margin:8px 0}
.onyx-stepChip{display:inline-flex;align-items:center;gap:6px;font-size:11px;color:var(--dsw-alias-label-secondary);border:1px solid var(--dsw-alias-border-l1);border-radius:999px;padding:3px 10px}
.onyx-stepChip.is-active{color:var(--ox-accent);border-color:var(--ox-accent);background:var(--ox-accent-soft)}
.onyx-stepChip.is-done{color:var(--dsw-alias-state-success-primary)}
.onyx-thumbCard{border:1px solid var(--dsw-alias-border-l1);border-radius:10px;overflow:hidden;background:var(--dsw-alias-bg-layer-1);cursor:pointer;transition:transform .15s,box-shadow .15s;padding:0;text-align:left}
.onyx-thumbCard:hover{transform:translateY(-2px);box-shadow:var(--ox-shadow)}
.onyx-aspect{display:inline-flex;flex-direction:column;align-items:center;gap:4px;border:1px solid var(--dsw-alias-border-l1);border-radius:9px;padding:7px 10px;cursor:pointer;background:transparent;transition:border-color .15s}
.onyx-aspect:hover{border-color:var(--ox-accent)}
.onyx-aspect.is-active{border-color:var(--ox-accent);background:var(--ox-accent-soft)}
.onyx-aspectBox{border:1.5px solid currentColor;border-radius:3px;opacity:.75}
@keyframes onyx-shimmer{from{background-position:200% 0}to{background-position:-200% 0}}
@keyframes onyx-fade{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
@keyframes onyx-breathe{0%,100%{opacity:.45}50%{opacity:1}}
@media (max-width:560px){.onyx-split{grid-template-columns:1fr}}
@media (prefers-reduced-motion:reduce){.onyx-view,.onyx-thumbCard,.onyx-card.is-hover{animation:none;transition:none}.onyx-dot.is-running{animation:none}}
/* ---- book shelf ---- */
.bk-shelfHead{display:flex;align-items:center;gap:10px;margin-bottom:14px}
.bk-shelf{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:16px}
.bk-coverCard{border:1px solid var(--dsw-alias-border-l1);border-radius:12px;overflow:hidden;background:var(--dsw-alias-bg-layer-1);transition:transform .15s,box-shadow .15s;padding:0;text-align:left;cursor:pointer}
.bk-coverCard:hover{transform:translateY(-3px);box-shadow:var(--ox-shadow)}
.bk-cover{aspect-ratio:3/4;width:100%;display:flex;align-items:center;justify-content:center;position:relative;background:linear-gradient(160deg,rgba(79,143,214,.16),rgba(138,92,246,.16));overflow:hidden}
.bk-coverGlyph{font-size:44px;font-weight:700;color:var(--ox-accent);opacity:.55}
.bk-coverTitle{position:absolute;left:10px;top:10px;right:10px;font-size:13px;font-weight:600;line-height:1.35;color:var(--dsw-alias-label-primary);text-shadow:0 1px 4px rgba(0,0,0,.25);word-break:break-all;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden}
.bk-coverMeta{padding:8px 10px;font-size:11px;color:var(--dsw-alias-label-secondary);display:flex;flex-direction:column;gap:6px}
.bk-progressTrack{height:4px;border-radius:3px;background:var(--dsw-alias-bg-layer-2);overflow:hidden}
.bk-progressFill{height:100%;background:linear-gradient(90deg,var(--ox-accent),var(--ox-accent2))}
.bk-empty{border:1px dashed var(--dsw-alias-border-l2);border-radius:12px;padding:48px 24px;text-align:center;color:var(--dsw-alias-label-secondary);font-size:13px}
.onyx-modalOverlay{position:fixed;inset:0;background:rgba(0,0,0,.42);display:flex;align-items:center;justify-content:center;z-index:60;animation:onyx-fade .15s ease}
.onyx-modal{width:min(460px,92vw);max-height:88vh;overflow:auto;background:var(--dsw-alias-bg-layer-1);border:1px solid var(--dsw-alias-border-l1);border-radius:14px;padding:18px;box-shadow:0 12px 40px rgba(0,0,0,.3)}
.onyx-modalTitle{font-size:15px;font-weight:700;margin:0 0 12px}
.bk-fileDrop{border:1.5px dashed var(--dsw-alias-border-l2);border-radius:10px;padding:18px;text-align:center;font-size:12px;color:var(--dsw-alias-label-secondary);cursor:pointer;transition:border-color .15s,background .15s}
.bk-fileDrop:hover,.bk-fileDrop.is-over{border-color:var(--ox-accent);background:var(--ox-accent-soft)}
.bk-fileName{margin-top:6px;color:var(--dsw-alias-label-primary);font-weight:500}
/* ---- reader ---- */
.bk-reader{--bk-bg:#f6efe2;--bk-fg:#3a3226;--bk-muted:#6e6350;--bk-frame:rgba(58,50,38,.16);--bk-accent:#a0642e;display:flex;flex-direction:column;height:100%;min-height:0;background:var(--bk-bg);color:var(--bk-fg);border-radius:10px}
.bk-reader[data-theme=paper]{--bk-bg:#f6efe2;--bk-fg:#3a3226;--bk-muted:#6e6350;--bk-frame:rgba(58,50,38,.16);--bk-accent:#a0642e}
.bk-reader[data-theme=night]{--bk-bg:#191a1e;--bk-fg:#c9c5bb;--bk-muted:#9a958a;--bk-frame:rgba(201,197,187,.16);--bk-accent:#b08a5a}
.bk-reader[data-theme=ink]{--bk-bg:#232529;--bk-fg:#d5d2c8;--bk-muted:#a09d94;--bk-frame:rgba(213,210,200,.16);--bk-accent:#9db3cf}
.bk-reader[data-font=song]{font-family:"Songti SC","STSong","Noto Serif CJK SC","Source Han Serif SC","SimSun",serif}
.bk-reader[data-font=kai]{font-family:"Kaiti SC","STKaiti","KaiTi","AR PL UKai CN",serif}
.bk-topbar{display:flex;align-items:center;gap:10px;padding:10px 16px;border-bottom:1px solid var(--bk-frame);flex:none;flex-wrap:wrap}
.bk-topTitle{font-size:13px;font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bk-topChapter{font-size:11px;color:var(--bk-muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bk-btn{border:1px solid var(--bk-frame);background:transparent;color:var(--bk-fg);border-radius:8px;padding:4px 11px;font-size:12px;cursor:pointer;font-family:inherit;transition:background .15s}
.bk-btn:hover{background:var(--bk-frame)}
.bk-progress{display:flex;align-items:center;gap:10px;padding:6px 16px;border-bottom:1px solid var(--bk-frame);flex:none;font-size:11px;color:var(--bk-muted);font-variant-numeric:tabular-nums}
.bk-slider{flex:1;-webkit-appearance:none;appearance:none;height:4px;border-radius:3px;background:var(--bk-frame);outline:none;cursor:pointer}
.bk-slider::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:13px;height:13px;border-radius:999px;background:var(--bk-accent);border:none}
.bk-slider::-moz-range-thumb{width:13px;height:13px;border-radius:999px;background:var(--bk-accent);border:none}
.bk-stage{flex:1;min-height:0;overflow:auto;position:relative}
.bk-pageText{max-width:680px;margin:0 auto;padding:36px 28px 48px;font-size:var(--bk-font-size,19px);line-height:var(--bk-line-height,1.9);text-align:justify}
.bk-chapterMark{text-align:center;color:var(--bk-muted);font-size:.78em;letter-spacing:.18em;margin:0 0 28px}
.bk-para{margin:0;text-indent:2em}
.bk-para + .bk-para{margin-top:.35em}
.bk-pageIllus{max-width:820px;margin:0 auto;padding:28px}
.bk-illusFrame{border:1px solid var(--bk-frame);border-radius:12px;overflow:hidden;background:rgba(0,0,0,.03);box-shadow:0 4px 24px rgba(0,0,0,.1)}
.bk-illusFrame img{width:100%;display:block}
.bk-illusEmpty{aspect-ratio:4/3;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;color:var(--bk-muted);font-size:12px;padding:20px;text-align:center}
.bk-caption{display:flex;align-items:flex-start;gap:8px;margin-top:10px;font-size:11.5px;color:var(--bk-muted);line-height:1.5}
.bk-captionText{flex:1;min-width:0;overflow:hidden;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical}
.bk-zone{position:absolute;top:0;bottom:0;width:20%;border:0;background:transparent;cursor:pointer;padding:0;z-index:2}
.bk-zone.is-left{left:0}.bk-zone.is-right{right:0}
.bk-zone:focus-visible{outline:2px solid var(--bk-accent);outline-offset:-2px}
.bk-pager{display:flex;align-items:center;justify-content:center;gap:14px;padding:10px 16px 14px;border-top:1px solid var(--bk-frame);flex:none;font-size:12px;color:var(--bk-muted);font-variant-numeric:tabular-nums}
.bk-pager .bk-btn{min-width:64px}
.bk-notice{position:absolute;left:50%;transform:translateX(-50%);bottom:18px;background:var(--dsw-alias-bg-layer-2);color:var(--dsw-alias-label-primary);border:1px solid var(--dsw-alias-border-l1);border-radius:999px;padding:6px 14px;font-size:12px;box-shadow:var(--ox-shadow);z-index:5;animation:onyx-fade .15s ease;max-width:80%;text-align:center}
.bk-drawerOverlay{position:fixed;inset:0;background:rgba(0,0,0,.35);z-index:70;animation:onyx-fade .15s ease}
.bk-drawer{position:fixed;top:0;bottom:0;left:0;width:min(320px,85vw);background:var(--dsw-alias-bg-layer-1);color:var(--dsw-alias-label-primary);border-right:1px solid var(--dsw-alias-border-l1);z-index:71;display:flex;flex-direction:column;box-shadow:8px 0 32px rgba(0,0,0,.25);animation:bk-slide .18s ease}
@keyframes bk-slide{from{transform:translateX(-24px);opacity:0}to{transform:none;opacity:1}}
.bk-drawerHead{display:flex;align-items:center;justify-content:space-between;padding:14px 16px;border-bottom:1px solid var(--dsw-alias-border-l1);font-weight:600;font-size:13px}
.bk-tocList{flex:1;overflow:auto;padding:6px}
.bk-tocItem{display:block;width:100%;text-align:left;border:0;background:transparent;color:var(--dsw-alias-label-primary);border-radius:8px;padding:8px 12px;font-size:12.5px;cursor:pointer;line-height:1.4}
.bk-tocItem:hover{background:var(--dsw-alias-bg-layer-2)}
.bk-tocItem.is-active{background:var(--ox-accent-soft);color:var(--ox-accent);font-weight:600}
.bk-tocNum{color:var(--dsw-alias-label-secondary);font-size:11px;margin-right:8px;font-variant-numeric:tabular-nums}
.bk-popover{position:fixed;z-index:71;width:min(300px,90vw);background:var(--dsw-alias-bg-layer-1);color:var(--dsw-alias-label-primary);border:1px solid var(--dsw-alias-border-l1);border-radius:12px;padding:14px;box-shadow:0 12px 40px rgba(0,0,0,.3);animation:onyx-fade .15s ease;max-height:80vh;overflow:auto}
.bk-settingRow{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:12px;font-size:12px}
.bk-settingRow:last-child{margin-bottom:0}
.bk-segChoice{display:inline-flex;gap:2px;background:var(--dsw-alias-bg-layer-2);border-radius:8px;padding:2px}
.bk-segChoice button{border:0;background:transparent;color:var(--dsw-alias-label-secondary);border-radius:6px;padding:3px 10px;font-size:11.5px;cursor:pointer;font-family:inherit}
.bk-segChoice button.is-active{background:var(--dsw-alias-bg-base);color:var(--dsw-alias-label-primary);font-weight:600}
.bk-promptBox{width:100%;min-height:84px;font-size:11.5px;line-height:1.5;border:1px solid var(--dsw-alias-border-l2);border-radius:8px;background:var(--dsw-alias-bg-base);color:var(--dsw-alias-label-primary);padding:8px;font-family:inherit;resize:vertical;box-sizing:border-box}
.bk-toggle{display:inline-flex;align-items:center;gap:8px;cursor:pointer;font-size:12px}
`

    function Section({ title, children, t }) {
      return h('section', { className: 'onyx-section' },
        title === undefined ? null : h('h3', { className: 'onyx-sectionTitle' }, title),
        children)
    }

    function Field({ label, children }) {
      // The control sits inside its label element, so screen readers name it.
      return h('label', { className: 'onyx-label', style: { display: 'block' } }, label, children)
    }

    function Sparkline({ points, height = 40, fill = true, label }) {
      const values = points.filter((v) => Number.isFinite(v))
      if (values.length < 2) return h('div', { className: 'onyx-empty' }, '—')
      const min = Math.min(...values)
      const max = Math.max(...values)
      const span = max - min || 1
      const width = 200
      const step = width / (values.length - 1)
      const path = values.map((v, i) => `${i === 0 ? 'M' : 'L'}${(i * step).toFixed(1)},${(height - ((v - min) / span) * (height - 4) - 2).toFixed(1)}`).join('')
      const area = `${path}L${width},${height}L0,${height}Z`
      return h('svg', {
        viewBox: `0 0 ${width} ${height}`, preserveAspectRatio: 'none',
        style: { width: '100%', height }, role: 'img',
        'aria-label': label ?? 'sparkline',
      },
        fill ? h('path', { d: area, fill: 'var(--dsw-alias-brand-primary)', opacity: 0.14 }) : null,
        h('path', { d: path, fill: 'none', stroke: 'var(--dsw-alias-brand-primary)', strokeWidth: 1.5, vectorEffect: 'non-scaling-stroke' }))
    }

    function Waterfall({ stages, t }) {
      const total = stages.reduce((sum, stage) => sum + stage.seconds, 0) || 1
      const colors = { text: '#7ba7d9', dit: 'var(--dsw-alias-brand-primary)', vae: '#9a86c8' }
      return h('div', null,
        h('div', { style: { display: 'flex', height: 16, borderRadius: 6, overflow: 'hidden', border: '1px solid var(--dsw-alias-border-l1)' } },
          stages.map((stage, index) => h('div', {
            key: stage.id + index,
            title: `${stage.id} ${fmt.sec(stage.seconds)}`,
            style: { width: `${(stage.seconds / total) * 100}%`, background: colors[stage.id] ?? 'var(--dsw-alias-state-idle-primary)' },
          }))),
        h('div', { className: 'onyx-row', style: { marginTop: 6 } },
          stages.map((stage, index) => h('span', { key: stage.id + index, className: 'onyx-chip' },
            h(StageName, { stage: stage.id, t }), ' · ', fmt.sec(stage.seconds), ' · ', `${((stage.seconds / total) * 100).toFixed(0)}%`))))
    }

    function ErrorNote({ error, onRetry, t }) {
      if (error == null) return null
      return h('div', { className: 'onyx-error', role: 'alert', style: { marginBottom: 10 } },
        `${t('common.error')}: ${error.message}`,
        onRetry === undefined ? null : h('button', { className: 'onyx-btn', style: { marginLeft: 10 }, onClick: onRetry }, t('common.retry')))
    }

    const GEN_STYLES = [
      ['photography', 'photorealistic photography, natural light, 50mm, shallow depth of field, highly detailed, no text'],
      ['ink', 'traditional Chinese ink-wash painting, expressive brush strokes, misty negative space, rice paper texture, no text'],
      ['storybook', 'warm storybook illustration, soft painterly texture, gentle palette, cozy lighting, no text'],
      ['cinematic', 'cinematic film still, dramatic volumetric lighting, rich color grading, 35mm, no text'],
    ]
    const SIZE_PRESETS = [
      ['1024', '1024'], ['1024', '1536'], ['1536', '1024'], ['2048', '2048'], ['4096', '4096'],
    ]

    function RecentStudio({ runs, onOpenRun, t }) {
      if (runs.length === 0) return null
      return h(Section, { title: t('generate.recent'), t },
        h('div', { style: { display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))', gap: 12 } },
          runs.map((run) => h('button', { key: run.dir, className: 'onyx-thumbCard', onClick: () => onOpenRun(run.dir) },
            run.outputExists && run.outputRel != null
              ? h(FetchImage, { src: `${FILES}${encodeURIComponent(run.outputRel)}`, alt: run.caseId, t, style: { width: '100%', aspectRatio: '4/3', objectFit: 'cover', borderRadius: 0, border: 'none' } })
              : h('div', { style: { width: '100%', aspectRatio: '4/3', background: 'var(--dsw-alias-bg-layer-2)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--dsw-alias-label-secondary)', fontSize: 11 } }, '—'),
            h('div', { style: { padding: '8px 10px' } },
              h('div', { className: 'onyx-itemTitle' }, run.caseId),
              h('div', { className: 'onyx-itemMeta' },
                h('span', null, fmt.sec(run.totalSeconds)),
                h('span', null, fmt.short(run.sha16, 8)),
                h('span', null, fmt.time(run.mtimeMs))))))))
    }

    /* ------------------------------------------------------ generate view */
    function LabelEditor({ runDir, t }) {
      const example = () => JSON.stringify({
        labels: [{ text: t('generate.labelExample'), box: [0.1, 0.06, 0.8, 0.09], background: '#293e52' }],
      }, null, 2)
      const [source, setSource] = useState(example)
      const [imageRel, setImageRel] = useState(null)
      const [error, setError] = useState(null)
      const [rendering, setRendering] = useState(false)
      const render = async () => {
        setRendering(true)
        setError(null)
        try {
          const result = await api('/generate/labels', {
            method: 'POST',
            headers: { 'content-type': 'application/json' },
            body: JSON.stringify({ runDir, layout: JSON.parse(source) }),
          })
          setImageRel(result.imageRel)
        } catch (failure) {
          setError(failure.message)
        } finally {
          setRendering(false)
        }
      }
      return h('details', { style: { marginTop: 12 } },
        h('summary', { className: 'onyx-label' }, t('generate.labels')),
        h(Field, { label: t('generate.labelLayout') },
          h('textarea', {
            className: 'onyx-textarea', value: source, spellCheck: false, disabled: rendering,
            'aria-label': t('generate.labelLayout'), onChange: (event) => setSource(event.target.value),
          })),
        error != null ? h('div', { className: 'onyx-error', role: 'alert' }, error) : null,
        h('div', { className: 'onyx-row', style: { marginTop: 8 } },
          h('button', { className: 'onyx-btn', onClick: render, disabled: rendering || source.trim() === '' },
            t(rendering ? 'generate.renderingLabels' : 'generate.renderLabels')),
          imageRel != null ? h(OpenImageLink, { src: `${FILES}${encodeURIComponent(imageRel)}`, label: t('generate.openLabels') }) : null),
        imageRel != null ? h(FetchImage, {
          src: `${FILES}${encodeURIComponent(imageRel)}`, alt: t('generate.openLabels'), t,
          style: { maxHeight: 400, objectFit: 'contain', margin: '10px auto 0' },
        }) : null)
    }

    function GenerateView({ t, onOpenRun }) {
      const [prompt, setPrompt] = useState('A close-up product photograph of one bright yellow enamel teapot with a black arched handle, standing on a blue wooden table. Soft morning window light, realistic enamel, no text.')
      const [width, setWidth] = useState(1024)
      const [height, setHeight] = useState(1024)
      const [seed, setSeed] = useState(42)
      const [startError, setStartError] = useState(null)
      const status = useApi('/generate/status', { intervalMs: 1200 })
      const runs = useApi('/runs?limit=40', { intervalMs: 15000 })
      const snapshot = status.data?.state === 'idle' ? null : status.data
      const running = snapshot != null && (snapshot.state === 'running' || snapshot.state === 'starting')
      // Expected total from runs of a comparable pixel area; falls back globally, then to 40s.
      const median = useMemo(() => {
        const all = (runs.data?.runs ?? runs.data ?? []).filter((run) => Number.isFinite(run.totalSeconds) && run.totalSeconds > 0)
        if (all.length === 0) return 40
        const area = width * height
        const bucket = all.filter((run) => {
          const match = /(\d+)x(\d+)/.exec(run.caseId ?? '')
          return match !== null && Math.abs(Number(match[1]) * Number(match[2]) - area) / area < 0.35
        })
        const pool = bucket.length > 0 ? bucket : all
        const sorted = pool.slice(0, 24).map((run) => run.totalSeconds).sort((a, b) => a - b)
        return sorted[Math.floor(sorted.length / 2)] ?? 40
      }, [runs.data, width, height])
      const weights = { text: 0.1, dit: 0.78, vae: 0.12 }
      const order = ['text', 'dit', 'vae']
      const stageIndex = order.indexOf(snapshot?.stage)
      const samples = snapshot?.samples ?? []
      const progress = useMemo(() => {
        if (!running || snapshot == null) return null
        let fraction = 0
        for (let i = 0; i < order.length; i += 1) {
          if (i < stageIndex) fraction += weights[order[i]]
          else if (i === stageIndex) {
            const expected = median * weights[order[i]] * 1000
            fraction += weights[order[i]] * Math.min(0.96, (snapshot.stageElapsedMs ?? 0) / expected)
          }
        }
        return Math.min(0.97, Math.max(0.02, stageIndex < 0 ? 0.02 : fraction))
      }, [running, snapshot, stageIndex, median])
      const studioRuns = useMemo(() => {
        const list = runs.data?.runs ?? runs.data ?? []
        return list.filter((run) => (run.caseId ?? '').startsWith('studio-')).slice(0, 8)
      }, [runs.data])
      const result = snapshot?.result ?? null

      const start = async (override) => {
        setStartError(null)
        const body = override ?? { prompt, width, height, seed }
        try {
          await api('/generate', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) })
          status.reload()
        } catch (error) {
          // The engine-busy conflict gets its own localized wording.
          setStartError(error.message?.includes?.('already running') ? { message: t('generate.waitJob') } : error)
        }
      }
      const cancel = async () => {
        try { await api('/generate/cancel', { method: 'POST' }) ; status.reload() } catch { status.reload() }
      }
      const retry = () => { setStartError(null); status.reload() }
      const appendStyle = (suffix) => {
        setPrompt((prev) => (prev.includes(suffix) ? prev : `${prev.trimEnd().replace(/[,\s]+$/, '')}, ${suffix}`))
      }
      const regenerate = (newSeed) => {
        if (snapshot?.request == null) return
        const base = snapshot.request
        start({ prompt: base.prompt, width: base.width, height: base.height, seed: newSeed ? Math.floor(Math.random() * 2 ** 31) : base.seed })
      }

      return h('div', { className: 'onyx-view' },
        h(ErrorNote, { error: status.error ?? startError, onRetry: retry, t }),
        h(Section, { title: t('generate.prompt'), t },
          h('textarea', { className: 'onyx-textarea', value: prompt, onChange: (e) => setPrompt(e.target.value), placeholder: t('generate.promptPlaceholder'), disabled: running }),
          h('div', { className: 'onyx-row', style: { marginTop: 8 } },
            h('span', { className: 'onyx-label', style: { margin: 0 } }, t('generate.stylePresets')),
            GEN_STYLES.map(([label, suffix]) =>
              h('button', { key: label, className: 'onyx-chip onyx-chipBtn', onClick: () => appendStyle(suffix) }, label)),
            h('span', { className: 'onyx-spacer' }),
            h('span', { className: 'onyx-sub' }, `${prompt.length}`)),
          h('div', { className: 'onyx-row', style: { marginTop: 10, alignItems: 'flex-end' } },
            h('div', null,
              h('div', { className: 'onyx-label' }, t('generate.size')),
              h('div', { className: 'onyx-row', style: { flexWrap: 'nowrap' } },
                SIZE_PRESETS.map(([w, hh]) => {
                  const ratio = Math.min(2, Math.max(0.5, Number(w) / Number(hh)))
                  return h('button', {
                    key: `${w}x${hh}`, className: `onyx-aspect${width === Number(w) && height === Number(hh) ? ' is-active' : ''}`,
                    disabled: running, onClick: () => { setWidth(Number(w)); setHeight(Number(hh)) },
                  },
                    h('span', { className: 'onyx-aspectBox', style: { width: Math.round(22 * ratio), height: Math.round(22 / ratio) } }),
                    h('span', { style: { fontSize: 10.5, fontVariantNumeric: 'tabular-nums' } }, `${w}×${hh}`))
                }))),
            h('div', { style: { width: 110 } },
              h(Field, { label: t('generate.seed') },
                h('div', { className: 'onyx-row', style: { flexWrap: 'nowrap' } },
                  h('input', { className: 'onyx-input', type: 'number', min: 0, value: seed, disabled: running, onChange: (e) => setSeed(Number(e.target.value)) }),
                  h('button', { className: 'onyx-btn', title: `${t('generate.seed')} 🎲`, disabled: running, onClick: () => setSeed(Math.floor(Math.random() * 2 ** 31)) }, '🎲')))),
            h('div', { className: 'onyx-spacer' }),
            running
              ? h('button', { className: 'onyx-btn', onClick: cancel }, t('generate.cancel'))
              : h('button', { className: 'onyx-btn onyx-btnPrimary', style: { padding: '9px 26px', fontSize: 13 }, onClick: () => start(), disabled: prompt.trim() === '' }, t('generate.start')))),

        running && snapshot != null ? h(Section, { title: `${t('generate.running')} · ${t('generate.elapsed')} ${fmt.sec((snapshot.elapsedMs ?? 0) / 1000)}`, t },
          h('div', { className: 'onyx-card' },
            h('div', { className: 'onyx-steps' }, order.map((stage, index) =>
              h('span', { key: stage, className: `onyx-stepChip${snapshot.stage === stage ? ' is-active' : ''}`, style: index < stageIndex ? { color: 'var(--dsw-alias-state-success-primary)' } : undefined },
                index < stageIndex ? '✓ ' : '', h(StageName, { stage, t })))),
            h('div', { className: 'onyx-barTrack', style: { height: 8 } },
              progress == null ? null : h('div', { className: 'onyx-barFill', style: { width: `${progress * 100}%` } })),
            h('div', { className: 'onyx-row', style: { marginTop: 10, alignItems: 'stretch' } },
              samples.length > 2 ? h('div', { style: { flex: 1, minWidth: 180 } },
                h('div', { className: 'onyx-label' }, `${t('generate.memory')} · ${fmt.mb(samples[samples.length - 1].rss)}`),
                h(Sparkline, { points: samples.map((s) => s.rss / 2 ** 20), label: t('generate.memory') })) : null,
              snapshot.logTail.length > 0 ? h('details', { style: { flex: 1, minWidth: 180 } },
                h('summary', { className: 'onyx-label', style: { cursor: 'pointer' } }, t('generate.log')),
                h('pre', { className: 'onyx-pre' }, snapshot.logTail.slice(-8).join('\n'))) : null),
            h('div', { className: 'onyx-sub', style: { marginTop: 6 } }, t('generate.estimate')))) : null,

        snapshot?.state === 'failed' ? h(Section, { t }, h('div', { className: 'onyx-error', role: 'alert' }, snapshot.error)) : null,
        snapshot?.state === 'cancelled' ? h(Section, { t }, h('div', { className: 'onyx-warn' }, snapshot.error ?? snapshot.state)) : null,

        result != null && result.runDir != null ? h(Section, { title: t('generate.result'), t },
          h('div', { className: 'onyx-card' },
            result.outputRel != null
              ? h(FetchImage, { src: `${FILES}${encodeURIComponent(result.outputRel)}`, alt: result.caseId, style: { maxHeight: 400, objectFit: 'contain', margin: '0 auto', borderRadius: 12 }, t })
              : null,
            h('div', { className: 'onyx-row', style: { marginTop: 10 } },
              h('span', { className: 'onyx-chip' }, `${t('run.total')} ${fmt.sec(result.totalSeconds)}`),
              (result.stages ?? []).map((stage) => h('span', { key: stage.id, className: 'onyx-chip' },
                h(StageName, { stage: stage.id, t }), ` ${fmt.sec(stage.seconds)}`)),
              h('span', { className: 'onyx-chip' }, `${t('run.peakRss')} ${fmt.gb(result.peakMemoryBytes)}`)),
            h('div', { className: 'onyx-row', style: { marginTop: 10 } },
              h('button', { className: 'onyx-btn', disabled: running, onClick: () => regenerate(false) }, t('generate.regenerateSame')),
              h('button', { className: 'onyx-btn', disabled: running, onClick: () => regenerate(true) }, t('generate.regenerateNew')),
              h('button', { className: 'onyx-btn onyx-btnGhost', onClick: () => onOpenRun(result.runDir) }, t('generate.viewRun')),
              result.outputRel != null ? h(OpenImageLink, { src: `${FILES}${encodeURIComponent(result.outputRel)}`, label: t('common.open') }) : null),
            result.outputRel != null ? h(LabelEditor, { key: result.runDir, runDir: result.runDir, t }) : null)) : null,

        h(RecentStudio, { runs: studioRuns, onOpenRun, t }))
    }

    /* ---------------------------------------------------------- runs view */
    function RunListItem({ run, active, onSelect }) {
      return h('button', { key: run.dir, className: `onyx-item${active ? ' is-active' : ''}`, onClick: () => onSelect(run.dir) },
        run.outputExists && run.outputRel != null
          ? h(FetchImage, { src: `${FILES}${encodeURIComponent(run.outputRel)}`, alt: '', style: { width: 44, height: 44, objectFit: 'cover', borderRadius: 7, flex: 'none' } })
          : h('div', { className: 'onyx-itemThumb' }),
        h('div', { className: 'onyx-itemBody' },
          h('div', { className: 'onyx-itemTitle' }, run.caseId),
          h('div', { className: 'onyx-itemMeta' },
            h('span', null, fmt.sec(run.totalSeconds)),
            h('span', null, fmt.gb(run.peakMemoryBytes)),
            h('span', null, fmt.time(run.mtimeMs)))))
    }

    function DitSteps({ steps, t }) {
      const scale = Math.max(...steps.map((step) => step.metalGiB), 0.1) * 1.05
      return h(Section, { title: `${t('run.ditSteps')} · max ${fmt.num(scale, 1)} GiB`, t },
        h('div', { className: 'onyx-card onyx-bars' }, steps.map((step) =>
          h('div', { key: step.step, className: 'onyx-barRow' },
            h('span', null, `step ${step.step}/${step.steps}`),
            h('div', { className: 'onyx-barTrack' },
              h('div', { className: 'onyx-barFill', style: { width: `${Math.min(100, (step.metalGiB / scale) * 100)}%` } })),
            h('span', null, `${fmt.num(step.metalGiB, 2)} GiB`))))
      )
    }

    function SameCaseTable({ runs, t }) {
      const shown = runs.slice(0, 8)
      const base = shown[shown.length - 1]?.totalSeconds || 1
      return h(Section, { title: t('run.sameCase'), t },
        h('table', { className: 'onyx-table' },
          h('thead', null, h('tr', null, h('th', null, 'plan'), h('th', null, t('run.total')), h('th', null, 'Δ'), h('th', null, 'sha'))),
          h('tbody', null, shown.map((run, index) => {
            const deltaPct = index === shown.length - 1 ? null : (run.totalSeconds / base - 1) * 100
            return h('tr', { key: run.dir }, [
              h('td', { key: 'p', className: 'wrap' }, fmt.short(run.planId, 26)),
              h('td', { key: 't' }, fmt.sec(run.totalSeconds)),
              h('td', { key: 'd' }, deltaPct === null ? '—' : h('span', { className: `onyx-badge ${deltaPct <= 0 ? 'is-ok' : 'is-bad'}` },
                `${deltaPct >= 0 ? '+' : ''}${deltaPct.toFixed(1)}%`)),
              h('td', { key: 's' }, fmt.short(run.sha16, 8)),
            ])
          }))))
    }

    function RunDetail({ detail, sameCase, t }) {
      const data = detail.data
      const metric = (name) => data.metrics?.[name]
      const output = data.output
      const outputUrl = output?.relPath != null && output.exists ? `${FILES}${encodeURIComponent(output.relPath)}` : null
      return h('div', { className: 'onyx-view' },
        h(Section, { title: `${data.caseId} · ${fmt.sec(data.totalSeconds)}`, t },
          h('div', { className: 'onyx-stats' },
            h('div', { className: 'onyx-stat' },
              h('div', { className: 'onyx-statValue' }, fmt.sec(data.totalSeconds)),
              h('div', { className: 'onyx-statLabel' }, t('run.total'))),
            h('div', { className: 'onyx-stat' },
              h('div', { className: 'onyx-statValue' }, fmt.gb(data.peakMemoryBytes)),
              h('div', { className: 'onyx-statLabel' }, t('run.peakRss'))),
            h('div', { className: 'onyx-stat' },
              h('div', { className: 'onyx-statValue' }, `${fmt.gb(metric('swap_used_before'))} → ${fmt.gb(metric('swap_used_after'))}`),
              h('div', { className: 'onyx-statLabel' }, t('run.swap'))),
            h('div', { className: 'onyx-stat' },
              h('div', { className: 'onyx-statValue' }, String(metric('text_tokens') ?? '—')),
              h('div', { className: 'onyx-statLabel' }, t('run.tokens')))),
          h('div', { className: 'onyx-card' },
            h('div', { className: 'onyx-label' }, t('run.waterfall')),
            h(Waterfall, { stages: data.stages, t }),
            h('dl', { className: 'onyx-kv', style: { marginTop: 8 } },
              h('dt', null, t('run.samples')), h('dd', null, String(metric('memory_samples') ?? '—')),
              h('dt', null, 'plan'), h('dd', null, data.planId ?? '—'))),
          h('div', { className: 'onyx-sub', style: { marginTop: 4 } }, t('run.memoryHint'))),
        data.process?.ditSteps?.length > 0 ? h(DitSteps, { steps: data.process.ditSteps, t }) : null,
        h(Section, { title: t('run.output'), t },
          outputUrl != null
            ? h(FetchImage, { src: outputUrl, alt: data.runId, style: { maxHeight: 300, objectFit: 'contain', borderRadius: 12 }, t })
            : h('div', { className: 'onyx-empty' }, t('common.empty')),
          h('div', { className: 'onyx-sub', style: { marginTop: 4 } }, `sha256:${output?.sha16 ?? '—'} · ${fmt.gb(output?.bytes)}`)),
        h(Section, { title: `${t('run.build')} · ${t('run.device')}`, t },
          h('dl', { className: 'onyx-kv' },
            h('dt', null, 'revision'), h('dd', null, `${fmt.short(data.build?.sourceRevision, 12)}${data.build?.dirty ? ' (dirty)' : ''}`),
            h('dt', null, 'runtime'), h('dd', null, String(data.build?.runtimeVersion ?? '—')),
            h('dt', null, 'backend'), h('dd', null, String(data.build?.backendVersion ?? '—')),
            h('dt', null, 'chip'), h('dd', null, String(data.device?.chip ?? data.device?.modelIdentifier ?? '—')),
            h('dt', null, 'gpu cores'), h('dd', null, String(data.device?.gpuCores ?? '—')))),
        sameCase.length > 1 ? h(SameCaseTable, { runs: sameCase, t }) : null)
    }

    function RunsView({ t, selectedDir, onSelect, compact = false }) {
      const [query, setQuery] = useState('')
      const runs = useApi('/runs?limit=120', { intervalMs: 20000 })
      const detail = useApi(selectedDir == null ? null : `/run?dir=${encodeURIComponent(selectedDir)}`, {})
      const engineError = runs.data?.error != null ? new Error(runs.data.error) : null
      const all = Array.isArray(runs.data?.runs) ? runs.data.runs : Array.isArray(runs.data) ? runs.data : []
      const list = query.trim() === '' ? all : all.filter((run) => (run.caseId ?? '').toLowerCase().includes(query.trim().toLowerCase()))
      const sameCase = all.filter((run) => selectedDir != null && run.caseId === selectedDir.split('/')[0])
      return h('div', { className: `onyx-split${compact ? ' is-compact' : ''}` },
        h('div', { style: { minHeight: 0 } },
          h(ErrorNote, { error: runs.error ?? engineError, onRetry: runs.reload, t }),
          h('div', { className: 'onyx-row', style: { marginBottom: 8 } },
            h('input', { className: 'onyx-input', style: { flex: 1, minWidth: 120 }, value: query, placeholder: '⌕ case id', onChange: (e) => setQuery(e.target.value) }),
            h('button', { className: 'onyx-btn', onClick: runs.reload }, t('common.refresh'))),
          h('div', { className: 'onyx-sub', style: { marginBottom: 6 } }, t('runs.count', { count: list.length })),
          h('div', { className: 'onyx-list', style: { maxHeight: compact ? 220 : 480 } },
            list.map((run) => h(RunListItem, { key: run.dir, run, active: run.dir === selectedDir, onSelect })))),
        h('div', { style: { minHeight: 0 } },
          detail.loading && selectedDir != null ? h('div', { className: 'onyx-skeleton', style: { height: 120 } }) : null,
          detail.error != null && selectedDir != null ? h(ErrorNote, { error: detail.error, onRetry: detail.reload, t }) : null,
          detail.data == null
            ? (selectedDir == null ? h('div', { className: 'onyx-empty' }, t('common.empty')) : null)
            : h(RunDetail, { detail, sameCase, t })))
    }

    /* --------------------------------------------------------- model view */
    function TargetSection({ target, t }) {
      return h(Section, { title: t('model.target'), t },
        h('dl', { className: 'onyx-card onyx-kv', style: { display: 'grid' } }, [
          h('dt', { key: 'chip-dt' }, 'chip'), h('dd', { key: 'chip-dd' }, String(target.chip ?? '—')),
          h('dt', { key: 'cg-dt' }, 'cpu / gpu'), h('dd', { key: 'cg-dd' }, `${target.cpu_cores ?? '—'} cores / ${target.gpu_cores ?? '—'} cores`),
          h('dt', { key: 'mem-dt' }, 'memory'), h('dd', { key: 'mem-dd' }, `${fmt.gb(target.physical_memory_bytes)} unified`),
          h('dt', { key: 'mtl-dt' }, 'metal'), h('dd', { key: 'mtl-dd' }, `${target.metal ?? '—'} · macOS ${target.macos_version ?? '—'}`),
          h('dt', { key: 'mb-dt' }, 'max metal buffer'), h('dd', { key: 'mb-dd' }, fmt.gb(target.max_metal_buffer_bytes)),
          h('dt', { key: 'rw-dt' }, 'recommended working set'), h('dd', { key: 'rw-dd' }, fmt.gb(target.recommended_gpu_working_set_bytes)),
        ]))
    }

    function PipelineSection({ manifest, totalBytes, t }) {
      return h(Section, { title: `${t('model.pipeline')} · ${fmt.gb(totalBytes)}`, t },
        h('table', { className: 'onyx-table' }, [
          h('thead', { key: 'head' }, h('tr', null, [
            h('th', { key: 'role' }, 'role'), h('th', { key: 'size' }, 'size'), h('th', { key: 'files' }, 'files'),
          ])),
          h('tbody', { key: 'body' }, manifest.components.map((component) =>
            h('tr', { key: component.role }, [
              h('td', { key: 'c' }, component.role),
              h('td', { key: 'b' }, fmt.gb(component.bytes)),
              h('td', { key: 'f' }, String(component.files.length)),
            ]))),
        ]))
    }

    function DitSection({ graph, t }) {
      return h(Section, { title: `${t('model.dit')} · ${t('model.graphNodes', { count: graph.nodes })}`, t },
        h('div', { className: 'onyx-card' }, [
          h('div', { key: 'sub', className: 'onyx-sub', style: { marginBottom: 6 } }, t('model.blockNodes', { count: graph.blockNodes })),
          h('div', { key: 'blocks', className: 'onyx-blocks' },
            Array.from({ length: Math.min(graph.blockNodes, 60) }, (_, i) =>
              h('span', { key: i, className: 'onyx-blockCell', title: `block ${i}` }))),
          h('details', { key: 'attrs', style: { marginTop: 8 } }, [
            h('summary', { key: 'sum', className: 'onyx-sub', style: { cursor: 'pointer' } }, 'block attributes'),
            h('dl', { key: 'dl', className: 'onyx-kv', style: { marginTop: 6 } },
              Object.entries(graph.representativeAttributes ?? {}).slice(0, 14).map(([key, value]) =>
                h(React.Fragment, { key }, [
                  h('dt', null, key),
                  h('dd', null, typeof value === 'object' ? JSON.stringify(value) : String(value)),
                ]))),
          ]),
        ]))
    }

    function PackSection({ pack, t }) {
      const total = pack.byDtype.reduce((sum, d) => sum + d.bytes, 0) || 1
      const palette = ['#4f8fd6', '#8fb85c', '#b087c8', '#d6a35c', '#7a8aa0']
      return h(Section, { title: `${t('model.dtypes')} · ${pack.component}`, t },
        h('div', { className: 'onyx-card' }, [
          h('div', { key: 'bar', style: { display: 'flex', height: 14, borderRadius: 6, overflow: 'hidden', border: '1px solid var(--dsw-alias-border-l1)' } },
            pack.byDtype.map((d, i) =>
              h('div', { key: d.dtype, title: `${d.dtype} ${fmt.gb(d.bytes)}`, style: { width: `${(d.bytes / total) * 100}%`, background: palette[i % 5] } }))),
          h('div', { key: 'legend', className: 'onyx-row', style: { marginTop: 6 } },
            pack.byDtype.map((d) => h('span', { key: d.dtype, className: 'onyx-chip' }, `${d.dtype} · ${d.tensors} tensors · ${fmt.gb(d.bytes)}`))),
          h('div', { key: 'total', className: 'onyx-sub', style: { marginTop: 6 } },
            `${t('model.totalParams', { value: `${(pack.totalElements / 1e9).toFixed(2)}B` })} · ${pack.tensors} tensors`),
        ]))
    }

    function QuantSection({ quant, rules, t }) {
      return h(Section, { title: `${t('model.quant')} · ${quant.profileId ?? quant.profile_id ?? ''}`, t },
        h('div', { className: 'onyx-card' }, [
          h('div', { key: 'algo', className: 'onyx-sub', style: { marginBottom: 6 } }, String(quant.algorithm ?? '')),
          h('table', { key: 'table', className: 'onyx-table' }, [
            h('thead', { key: 'head' }, h('tr', null, [
              h('th', { key: 'p' }, 'tensor pattern'), h('th', { key: 'b' }, 'bits'), h('th', { key: 'g' }, 'group'),
            ])),
            h('tbody', { key: 'body' }, rules.map((rule, i) =>
              h('tr', { key: i }, [
                h('td', { key: 'p', className: 'wrap' }, String(rule.tensorPattern ?? rule.tensor_pattern ?? '')),
                h('td', { key: 'b' }, String(rule.bits ?? '—')),
                h('td', { key: 'g' }, String(rule.groupSize ?? rule.group_size ?? '—')),
              ]))),
          ]),
        ]))
    }

    function RecipeSection({ recipe, t }) {
      return h(Section, { title: `${t('model.recipe')} · ${recipe.recipeId ?? recipe.recipe_id ?? ''}`, t },
        h('div', { className: 'onyx-card' }, [
          h('div', { key: 'label', className: 'onyx-label' }, 'sigmas'),
          h(Sparkline, { key: 'spark', points: recipe.sigmas ?? [], t }),
          h('div', { key: 'calls', className: 'onyx-row', style: { marginTop: 6 } },
            (recipe.calls ?? []).map((call, i) =>
              h('span', { key: i, className: 'onyx-chip' }, `${call.stage ?? call.stageId ?? ''} ×${call.count ?? call.times ?? 1}`))),
        ]))
    }

    function ModelView({ t }) {
      const model = useApi('/model', {})
      const data = model.data
      const engineError = data?.error != null ? new Error(data.error) : null
      const target = data?.target
      const manifest = data?.manifest
      const ditGraph = (manifest?.graphs ?? []).reduce((best, graph) => (best == null || graph.nodes > best.nodes ? graph : best), null)
      const quant = manifest?.quantization ?? {}
      const rules = quant.tensorRules ?? quant.tensor_rules ?? []
      const recipe = (manifest?.recipes ?? [])[0]
      const numeric = (manifest?.numericProfiles ?? [])[0]
      const totalBytes = (manifest?.components ?? []).reduce((sum, c) => sum + c.bytes, 0)
      const paramB = (data?.packs ?? []).reduce((sum, pack) => sum + pack.totalElements, 0)
      const tensorCount = (data?.packs ?? []).reduce((sum, pack) => sum + pack.tensors, 0)
      const modelMissing = data?.manifest == null
      return h('div', { className: 'onyx-view' }, [
        h(ErrorNote, { key: 'err', error: model.error ?? engineError, onRetry: model.reload, t }),
        engineError == null && data?.problems?.length > 0 ? h('div', { key: 'warn', className: 'onyx-warn', style: { marginBottom: 10 } }, data.problems.join('；')) : null,
        modelMissing ? null : h('div', { key: 'hero', className: 'onyx-stats' }, [
          h('div', { key: 'p', className: 'onyx-stat' },
            h('div', { className: 'onyx-statValue' }, paramB > 0 ? `${(paramB / 1e9).toFixed(2)}B` : '—'),
            h('div', { className: 'onyx-statLabel' }, t('model.statParams'))),
          h('div', { key: 't', className: 'onyx-stat' },
            h('div', { className: 'onyx-statValue' }, tensorCount > 0 ? String(tensorCount) : '—'),
            h('div', { className: 'onyx-statLabel' }, t('model.statTensors'))),
          h('div', { key: 'b', className: 'onyx-stat' },
            h('div', { className: 'onyx-statValue' }, fmt.gb(totalBytes)),
            h('div', { className: 'onyx-statLabel' }, t('model.statSize'))),
          h('div', { key: 'g', className: 'onyx-stat' },
            h('div', { className: 'onyx-statValue' }, String((manifest?.graphs ?? []).length)),
            h('div', { className: 'onyx-statLabel' }, 'DiT graphs')),
        ]),
        target == null ? null : h(TargetSection, { key: 'target', target, t }),
        manifest == null ? null : h(PipelineSection, { key: 'pipeline', manifest, totalBytes, t }),
        ditGraph == null ? null : h(DitSection, { key: 'dit', graph: ditGraph, t }),
        (data?.packs ?? []).map((pack) => h(PackSection, { key: pack.component, pack, t })),
        rules.length === 0 ? null : h(QuantSection, { key: 'quant', quant, rules, t }),
        recipe == null ? null : h(RecipeSection, { key: 'recipe', recipe, t }),
        numeric == null ? null : h(Section, { key: 'numeric', title: t('model.numeric'), t },
          h('dl', { className: 'onyx-card onyx-kv', style: { display: 'grid' } },
            Object.entries(numeric).slice(0, 12).map(([key, value]) =>
              h(React.Fragment, { key }, h('dt', null, key), h('dd', null, typeof value === 'object' ? JSON.stringify(value) : String(value)))))),
      ])
    }

    /* ----------------------------------------------------- operators view */
    function CompareTable({ rows, t }) {
      const maxRatio = Math.max(...rows.map((row) => row.ratio ?? 0), 1)
      return h(Section, { title: t('ops.compare'), t },
        h('table', { className: 'onyx-table' }, [
          h('thead', { key: 'head' }, h('tr', null, [
            h('th', { key: 'case' }, t('ops.case')),
            h('th', { key: 'shape' }, t('ops.shape')),
            h('th', { key: 'onyx' }, t('ops.onyx')),
            h('th', { key: 'mlx' }, t('ops.mlx')),
            h('th', { key: 'ratio' }, t('ops.ratio')),
            h('th', { key: 'correct' }, t('ops.correct')),
          ])),
          h('tbody', { key: 'body' }, rows.map((row) => h('tr', { key: row.caseId }, [
            h('td', { key: 'case', className: 'wrap' }, row.caseId),
            h('td', { key: 'shape' }, `${row.shape.m ?? '?'}×${row.shape.n ?? '?'}×${row.shape.k ?? '?'}`),
            h('td', { key: 'onyx' }, fmt.ms(row.onyx.wallMedianMs)),
            h('td', { key: 'mlx' }, fmt.ms(row.mlx.wallMedianMs)),
            h('td', { key: 'ratio' }, h('div', { className: 'onyx-row', style: { gap: 6, flexWrap: 'nowrap' } },
              h('span', { className: `onyx-badge ${row.ratio > 10 ? 'is-bad' : row.ratio > 3 ? 'is-warn' : 'is-ok'}` }, `${fmt.num(row.ratio, 1)}×`),
              h('div', { style: { width: 60, height: 6, borderRadius: 4, background: 'var(--dsw-alias-bg-layer-2)', overflow: 'hidden' } },
                h('div', { className: 'onyx-miniBar', style: { width: `${Math.min(100, ((row.ratio ?? 0) / maxRatio) * 100)}%` } })))),
            h('td', { key: 'correct' }, row.onyx.correct === true ? '✓' : row.onyx.correct === false ? '✗' : '—'),
          ]))),
        ]))
    }

    function RawTable({ rows, t }) {
      return h(Section, { title: t('ops.raw'), t },
        h('table', { className: 'onyx-table' }, [
          h('thead', { key: 'head' }, h('tr', null, [
            h('th', { key: 'case' }, 'case'),
            h('th', { key: 'tf' }, t('ops.tflops')),
            h('th', { key: 'bw' }, t('ops.bandwidth')),
            h('th', { key: 'metal' }, 'metal'),
          ])),
          h('tbody', { key: 'body' }, rows.map((row) =>
            h('tr', { key: row.file }, [
              h('td', { key: 'case', className: 'wrap' }, row.file),
              h('td', { key: 'tf' }, fmt.num(row.effectiveTflops)),
              h('td', { key: 'bw' }, `${fmt.num(row.effectiveBandwidthGbS, 1)} GB/s`),
              h('td', { key: 'metal' }, fmt.gb(row.metalAllocatedBytes)),
            ]))),
        ]))
    }

    function OperatorsView({ t }) {
      const operators = useApi('/operators', {})
      const data = operators.data
      const engineError = data?.error != null ? new Error(data.error) : null
      if (data == null || engineError != null) {
        return h('div', null, [
          h(ErrorNote, { key: 'err', error: operators.error ?? engineError, onRetry: operators.reload, t }),
          engineError == null ? h('div', { key: 'empty', className: 'onyx-empty' }, operators.loading ? t('common.loading') : t('common.empty')) : null,
        ])
      }
      return h('div', null, [
        h(CompareTable, { key: 'compare', rows: data.comparisons ?? [], t }),
        h(RawTable, { key: 'raw', rows: data.raw ?? [], t }),
      ])
    }

    /* ------------------------------------------------------------ books */
    /**
     * The reading model: one data page holds text plus an illustration slot;
     * the reader shows them as two views (text at 2n, image at 2n+1), so a
     * book reads as one page of text followed by one page of image.
     */
    const BOOK_STYLE_IDS = ['ink', 'cinematic', 'storybook', 'minimal']
    const BOOK_CHARS_OPTIONS = [500, 800, 1100, 1400]

    function safePageView(meta) {
      return Math.min(Math.max(0, Math.round(safeNumber(meta?.progress?.lastPage, 0))), Math.max(0, safeNumber(meta?.pageCount, 1) - 1))
    }
    function bytesToBase64(buffer) {
      const bytes = new Uint8Array(buffer)
      const chunk = 0x8000
      let binary = ''
      for (let offset = 0; offset < bytes.length; offset += chunk) {
        binary += String.fromCharCode(...bytes.subarray(offset, offset + chunk))
      }
      return btoa(binary)
    }

    function BookCover({ book, height }) {
      if (book.cover != null) {
        return h(FetchImage, { src: `${FILES}${encodeURIComponent(book.cover)}`, alt: book.title, style: { position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover', borderRadius: 0, border: 'none' } })
      }
      return h('span', { className: 'bk-coverGlyph' }, (book.title ?? 'B').slice(0, 1))
    }

    function BookCard({ book, t, onOpen, onDelete }) {
      const progress = Math.min(100, Math.round(((safeNumber(book.progress?.lastPage, 0) + 1) / Math.max(1, book.pageCount)) * 100))
      return h('div', {
        className: 'bk-coverCard', role: 'button', tabIndex: 0, title: book.title,
        onClick: () => onOpen(book),
        onKeyDown: (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onOpen(book) } },
      },
        h('div', { className: 'bk-cover' },
          h(BookCover, { book }),
          h('span', { className: 'bk-coverTitle' }, book.title)),
        h('div', { className: 'bk-coverMeta' },
          h('div', { className: 'onyx-row', style: { justifyContent: 'space-between', gap: 4 } },
            h('span', { style: { overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' } }, book.author || book.format.toUpperCase()),
            h('button', {
              className: 'onyx-btn onyx-btnGhost', style: { padding: '1px 7px', fontSize: 11 },
              onClick: (event) => { event.stopPropagation(); if (window.confirm(t('book.deleteConfirm'))) onDelete(book) },
            }, '🗑')),
          h('div', { className: 'bk-progressTrack' },
            h('div', { className: 'bk-progressFill', style: { width: `${progress}%` } })),
          h('div', { className: 'onyx-row', style: { justifyContent: 'space-between', gap: 4 } },
            h('span', null, `${t('book.progress')} ${progress}%`),
            h('span', null, `${book.illustrated ?? 0}/${book.pageCount} ${t('book.illustrated')}`))))
    }

    function ImportDialog({ t, onClose, onImported }) {
      const [file, setFile] = useState(null)
      const [title, setTitle] = useState('')
      const [author, setAuthor] = useState('')
      const [style, setStyle] = useState('ink')
      const [charsPerPage, setCharsPerPage] = useState(800)
      const [busy, setBusy] = useState(false)
      const [error, setError] = useState(null)
      const [dragOver, setDragOver] = useState(false)
      const inputRef = useRef(null)

      useEffect(() => {
        const onKey = (event) => { if (event.key === 'Escape' && !busy) onClose() }
        window.addEventListener('keydown', onKey)
        return () => window.removeEventListener('keydown', onKey)
      }, [busy, onClose])

      const pickFile = (picked) => {
        if (picked == null || picked.length === 0) return
        const chosen = picked[0]
        const ext = (chosen.name.split('.').pop() ?? '').toLowerCase()
        if (!['txt', 'md', 'epub', 'markdown'].includes(ext)) {
          setError(new Error(`${chosen.name}: txt / md / epub only`))
          return
        }
        // Server cap is 64MB of JSON body; base64 inflates by ~4/3.
        if (chosen.size > 48 * 1024 * 1024) {
          setError(new Error(t('book.tooLarge')))
          return
        }
        setFile(chosen)
        setTitle(chosen.name.replace(/\.[^.]+$/, '').slice(0, 60))
        setError(null)
      }

      const submit = async () => {
        if (file == null || busy) return
        setBusy(true)
        setError(null)
        try {
          const buffer = await file.arrayBuffer()
          const ext = (file.name.split('.').pop() ?? '').toLowerCase()
          const format = ext === 'markdown' ? 'md' : ext
          const meta = await api('/books/import', {
            method: 'POST',
            headers: { 'content-type': 'application/json' },
            body: JSON.stringify({ format, title: title || file.name, author, style, charsPerPage, payloadB64: bytesToBase64(buffer) }),
            signal: AbortSignal.timeout(180_000),
          })
          onImported(meta)
        } catch (reason) {
          setError(reason instanceof Error ? reason : new Error(String(reason)))
        } finally {
          setBusy(false)
        }
      }

      return h('div', { className: 'onyx-modalOverlay', onClick: (event) => { if (event.target === event.currentTarget && !busy) onClose() } },
        h('div', { className: 'onyx-modal', role: 'dialog', 'aria-modal': 'true', 'aria-label': t('book.import') },
          h('h3', { className: 'onyx-modalTitle' }, t('book.import')),
          h('div', {
            className: `bk-fileDrop${dragOver ? ' is-over' : ''}`, role: 'button', tabIndex: 0,
            onClick: () => inputRef.current?.click(),
            onKeyDown: (event) => { if (event.key === 'Enter' || event.key === ' ') inputRef.current?.click() },
            onDragOver: (event) => { event.preventDefault(); setDragOver(true) },
            onDragLeave: () => setDragOver(false),
            onDrop: (event) => { event.preventDefault(); setDragOver(false); pickFile(event.dataTransfer.files) },
          },
            h('input', {
              ref: inputRef, type: 'file', accept: '.txt,.md,.markdown,.epub', style: { display: 'none' },
              onChange: (event) => pickFile(event.target.files),
            }),
            file == null ? t('book.importFile') : h('span', { className: 'bk-fileName' }, `📄 ${file.name} (${fmt.mb(file.size)})`)),
          h('div', { className: 'onyx-sub', style: { margin: '6px 0 12px', lineHeight: 1.5 } }, t('book.importHint')),
          h('div', { className: 'onyx-grid', style: { gridTemplateColumns: '1fr 1fr' } },
            h(Field, { label: t('book.title') }, h('input', { className: 'onyx-input', value: title, onChange: (e) => setTitle(e.target.value) })),
            h(Field, { label: t('book.author') }, h('input', { className: 'onyx-input', value: author, onChange: (e) => setAuthor(e.target.value) }))),
          h(Field, { label: t('book.style') },
            h('div', { className: 'onyx-row' }, BOOK_STYLE_IDS.map((id) =>
              h('button', { key: id, className: `onyx-chip onyx-chipBtn${style === id ? ' is-active' : ''}`, title: t(`book.styleDesc.${id}`), onClick: () => setStyle(id) }, t(`book.style.${id}`))))),
          h(Field, { label: t('book.charsPerPage') },
            h('div', { className: 'onyx-row' }, BOOK_CHARS_OPTIONS.map((count) =>
              h('button', { key: count, className: `onyx-chip onyx-chipBtn${charsPerPage === count ? ' is-active' : ''}`, onClick: () => setCharsPerPage(count) }, String(count))))),
          error != null ? h('div', { className: 'onyx-error', role: 'alert', style: { marginTop: 10 } }, error.message) : null,
          h('div', { className: 'onyx-row', style: { marginTop: 14, justifyContent: 'flex-end' } },
            h('button', { className: 'onyx-btn', onClick: onClose, disabled: busy }, t('book.cancelImport')),
            h('button', { className: 'onyx-btn onyx-btnPrimary', onClick: submit, disabled: file == null || busy },
              busy ? t('book.importing') : t('book.import')))))
    }

    function TocDrawer({ t, book, chapters, currentPage, onJump, onClose }) {
      return h('div', null,
        h('div', { className: 'bk-drawerOverlay', onClick: onClose }),
        h('aside', { className: 'bk-drawer', role: 'dialog', 'aria-label': t('book.toc') },
          h('div', { className: 'bk-drawerHead' },
            h('span', null, t('book.toc')),
            h('button', { className: 'onyx-btn onyx-btnGhost', onClick: onClose }, '✕')),
          h('div', { className: 'bk-tocList' },
            (chapters ?? []).map((chapter) => {
              const active = currentPage >= chapter.firstPage
                && (chapters.indexOf(chapter) === chapters.length - 1 || currentPage < (chapters[chapters.indexOf(chapter) + 1]?.firstPage ?? Infinity))
              return h('button', {
                key: chapter.index, className: `bk-tocItem${active ? ' is-active' : ''}`,
                onClick: () => onJump(chapter.firstPage),
              },
                h('span', { className: 'bk-tocNum' }, String(chapter.index + 1)),
                chapter.title)
            }))))
    }

    function SettingsPopover({ t, settings, onChange, onClose, anchor }) {
      const rect = anchor?.getBoundingClientRect?.()
      const styleProps = rect == null ? { top: 90, right: 16 } : { top: Math.min(rect.bottom + 8, window.innerHeight - 340), right: Math.max(12, window.innerWidth - rect.right - 10) }
      const set = (patch) => onChange({ ...settings, ...patch })
      return h('div', null,
        h('div', { className: 'bk-drawerOverlay', style: { zIndex: 70 }, onClick: onClose }),
        h('div', { className: 'bk-popover', style: styleProps, role: 'dialog', 'aria-label': t('book.settings') },
          h('div', { className: 'bk-settingRow' },
            h('span', null, t('book.theme')),
            h('div', { className: 'bk-segChoice' }, ['paper', 'night', 'ink'].map((theme) =>
              h('button', { key: theme, className: settings.theme === theme ? 'is-active' : '', onClick: () => set({ theme }) }, t(`book.theme.${theme}`))))),
          h('div', { className: 'bk-settingRow' },
            h('span', null, t('book.font')),
            h('div', { className: 'bk-segChoice' }, ['song', 'kai'].map((font) =>
              h('button', { key: font, className: settings.serif === font ? 'is-active' : '', onClick: () => set({ serif: font }) }, t(`book.font.${font}`))))),
          h('div', { className: 'bk-settingRow' },
            h('span', null, `${t('book.fontSize')} ${settings.fontSize ?? 19}`),
            h('input', { type: 'range', min: 15, max: 26, step: 1, value: settings.fontSize ?? 19, className: 'bk-slider', style: { width: 120 }, 'aria-label': t('book.fontSize'), onChange: (e) => set({ fontSize: Number(e.target.value) }) })),
          h('div', { className: 'bk-settingRow' },
            h('span', null, `${t('book.lineHeight')} ${Number(settings.lineHeight ?? 1.9).toFixed(1)}`),
            h('input', { type: 'range', min: 1.5, max: 2.4, step: 0.1, value: settings.lineHeight ?? 1.9, className: 'bk-slider', style: { width: 120 }, 'aria-label': t('book.lineHeight'), onChange: (e) => set({ lineHeight: Number(e.target.value) }) })),
          h('div', { className: 'bk-settingRow' },
            h('label', { className: 'bk-toggle' },
              h('input', { type: 'checkbox', checked: settings.autoIllustrate !== false, onChange: (e) => set({ autoIllustrate: e.target.checked }) }),
              t('book.autoIllustrate'))),
          settings.autoIllustrate === false ? null : h('div', { className: 'bk-settingRow' },
            h('span', null, t('book.aheadCount')),
            h('div', { className: 'bk-segChoice' }, [1, 2, 3].map((count) =>
              h('button', { key: count, className: Number(settings.aheadCount ?? 2) === count ? 'is-active' : '', onClick: () => set({ aheadCount: count }) }, String(count)))))))
    }

    function PromptEditor({ t, initial, busy, onApply, onClose }) {
      const [value, setValue] = useState(initial ?? '')
      useEffect(() => {
        const onKey = (event) => { if (event.key === 'Escape' && !busy) onClose() }
        window.addEventListener('keydown', onKey)
        return () => window.removeEventListener('keydown', onKey)
      }, [busy, onClose])
      return h('div', { className: 'onyx-modalOverlay', onClick: (event) => { if (event.target === event.currentTarget) onClose() } },
        h('div', { className: 'onyx-modal', role: 'dialog', 'aria-modal': 'true', 'aria-label': t('book.promptLabel') },
          h('h3', { className: 'onyx-modalTitle' }, t('book.promptLabel')),
          h('textarea', { className: 'bk-promptBox', value, onChange: (event) => setValue(event.target.value) }),
          h('div', { className: 'onyx-row', style: { marginTop: 12, justifyContent: 'flex-end' } },
            h('button', { className: 'onyx-btn', onClick: onClose }, t('book.cancelImport')),
            h('button', { className: 'onyx-btn onyx-btnPrimary', disabled: busy || value.trim() === '', onClick: () => onApply(value.trim()) }, t('book.applyPrompt')))))
    }

    function IllustrationPane({ t, bookId, page, requested, failure, jobSnapshot, onGenerate, onEditPrompt, onRegenerate }) {
      const image = page.image
      const busyForThisPage = jobSnapshot != null
        && (jobSnapshot.state === 'running' || jobSnapshot.state === 'starting')
        && jobSnapshot.purpose?.bookId === bookId && jobSnapshot.purpose?.page === page.index
      let body
      if (busyForThisPage) {
        body = h('div', { className: 'bk-illusEmpty' },
          h('div', { className: 'onyx-dot is-running', style: { width: 12, height: 12 } }),
          h('div', null, t('book.generating')),
          h('div', { className: 'onyx-sub' }, `${fmt.sec((jobSnapshot.elapsedMs ?? 0) / 1000)} · ${jobSnapshot.stage ?? ''}`))
      } else if (image != null) {
        body = h(FetchImage, { src: `${FILES}${encodeURIComponent(image.file)}`, alt: page.prompt, style: { width: '100%' } })
      } else {
        body = h('div', { className: 'bk-illusEmpty' },
          h('div', { style: { fontSize: 26, opacity: 0.5 } }, '🖼'),
          h('div', { style: { maxWidth: 420, overflow: 'hidden', display: '-webkit-box', WebkitLineClamp: 3, WebkitBoxOrient: 'vertical' } }, page.prompt),
          failure != null ? h('div', { className: 'onyx-error', role: 'alert', style: { maxWidth: 420, fontSize: 11, wordBreak: 'break-all' } },
            `${t('book.illustFailed')}: ${failure}`) : null,
          requested && failure == null ? h('div', null, t('book.queued')) : null,
          h('button', { className: 'bk-btn', onClick: () => onGenerate(page.index) }, failure != null ? t('book.regenerate') : t('book.generateIllust')),
          h('button', { className: 'bk-btn', style: { opacity: 0.75 }, onClick: () => onEditPrompt(page.index) }, t('book.editPrompt')))
      }
      return h('div', { className: 'bk-pageIllus' },
        h('div', { className: 'bk-illusFrame' }, body),
        h('div', { className: 'bk-caption' },
          h('span', null, '✦'),
          h('span', { className: 'bk-captionText' }, image?.prompt ?? page.prompt),
          image != null ? h('button', { className: 'bk-btn', style: { flex: 'none', padding: '2px 9px', fontSize: 11 }, disabled: busyForThisPage, onClick: () => onRegenerate(page.index) }, t('book.regenerate')) : null),
        image != null ? h('div', { className: 'bk-caption', style: { opacity: 0.7 } },
          h('span', null, '⟳'),
          h('span', { className: 'bk-captionText' }, `${t('book.seed')} ${image.seed ?? '—'} · ${image.width ?? '?'}×${image.height ?? '?'} · ${fmt.time(image.createdAtMs)}`)) : null)
    }

    function ReaderView({ t, meta, onBack }) {
      const bookId = meta.id
      const total = safeNumber(meta.pageCount, 0)
      const [pos, setPos] = useState(() => safePageView(meta) * 2)
      const [settings, setSettings] = useState(() => ({ ...(meta.settings ?? {}) }))
      const [tocOpen, setTocOpen] = useState(false)
      const [settingsOpen, setSettingsOpen] = useState(false)
      const [promptEdit, setPromptEdit] = useState(null) // { page, initial }
      const [manualBusy, setManualBusy] = useState(false)
      const settingsAnchorRef = useRef(null)
      const requestedRef = useRef(new Set())
      const stageRef = useRef(null)

      const currentPage = Math.floor(pos / 2)
      const isImage = pos % 2 === 1
      const windowStart = Math.max(0, currentPage - 1)
      const windowEnd = Math.min(total, currentPage + 4)
      const pages = useApi(total > 0 ? `/pages?id=${encodeURIComponent(bookId)}&from=${windowStart}&to=${windowEnd}` : null, {})
      const status = useApi('/generate/status', { intervalMs: 1500 })
      const pageMap = useMemo(() => {
        const map = new Map()
        for (const page of pages.data?.pages ?? []) map.set(page.index, page)
        return map
      }, [pages.data])
      const page = pageMap.get(currentPage)
      const chapters = pages.data?.chapters ?? []
      const jobSnapshot = status.data?.state === 'idle' ? null : status.data
      const jobState = jobSnapshot?.state
      const jobBookId = jobSnapshot?.purpose?.bookId
      const jobPage = jobSnapshot?.purpose?.page
      const jobAttachError = jobSnapshot?.attachError ?? null
      const jobResultImage = jobSnapshot?.result?.image ?? null
      const [notice, setNotice] = useState(null)
      const noticeTimerRef = useRef(null)
      const showNotice = useCallback((message) => {
        setNotice(message)
        if (noticeTimerRef.current != null) clearTimeout(noticeTimerRef.current)
        noticeTimerRef.current = setTimeout(() => setNotice(null), 4000)
      }, [])

      // Persist reading progress (debounced, flushed on unmount). Only the
      // data page matters, so text↔image moves inside one page save nothing.
      const posRef = useRef(pos)
      posRef.current = pos
      const savedPageRef = useRef(Math.floor(pos / 2))
      const saveProgress = useCallback(() => {
        const current = Math.floor(posRef.current / 2)
        if (savedPageRef.current === current) return
        savedPageRef.current = current
        api('/book', { method: 'PATCH', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ id: bookId, progress: { lastPage: current } }) }).catch(() => {})
      }, [bookId])
      const saveRef = useRef(saveProgress)
      saveRef.current = saveProgress
      useEffect(() => {
        const timer = setTimeout(() => saveRef.current(), 800)
        return () => clearTimeout(timer)
      }, [pos])
      useEffect(() => () => { saveRef.current() }, []) // flush on unmount

      // Debounced settings persistence (slider drags fire many changes).
      const settingsRef = useRef(settings)
      settingsRef.current = settings
      const settingsTimerRef = useRef(null)
      useEffect(() => () => {
        if (settingsTimerRef.current != null) {
          clearTimeout(settingsTimerRef.current)
          api('/book', { method: 'PATCH', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ id: bookId, settings: settingsRef.current }) }).catch(() => {})
        }
      }, [bookId])

      // Refresh the page window when a book illustration finishes; the server
      // only reports 'succeeded' after the attach has been written.
      useEffect(() => {
        if (jobState === 'succeeded' && jobBookId === bookId) pages.reload()
      }, [jobState, jobBookId, bookId]) // eslint-disable-line react-hooks/exhaustive-deps

      // Release the reserved slot when a job for this book ends without a
      // usable image, so the auto loop and the manual button recover.
      useEffect(() => {
        if (jobBookId !== bookId || jobPage == null) return
        const failed = jobState === 'failed' || jobState === 'cancelled'
          || jobAttachError != null || (jobState === 'succeeded' && jobResultImage == null)
        if (failed) requestedRef.current.delete(`${bookId}:${jobPage}`)
      }, [jobState, jobBookId, jobPage, jobAttachError, jobResultImage, bookId])

      // Auto pre-generation: keep the next few pages illustrated while reading.
      const ahead = Math.max(0, Math.round(safeNumber(settings.aheadCount, 2)))
      const autoOn = settings.autoIllustrate !== false
      useEffect(() => {
        if (!autoOn || total === 0 || pages.loading) return
        if (jobState === 'running' || jobState === 'starting') return
        for (let offset = 0; offset <= ahead && currentPage + offset < total; offset += 1) {
          const candidate = pageMap.get(currentPage + offset)
          if (candidate == null) return // window not loaded yet
          if (candidate.image != null) continue
          const key = `${bookId}:${candidate.index}`
          if (requestedRef.current.has(key)) return // in flight
          requestedRef.current.add(key)
          api('/book/illustrate', {
            method: 'POST', headers: { 'content-type': 'application/json' },
            body: JSON.stringify({ id: bookId, page: candidate.index }),
          }).catch(() => { requestedRef.current.delete(key) })
          return
        }
      }, [autoOn, ahead, currentPage, total, bookId, pageMap, pages.loading, jobState])

      const go = useCallback((delta) => {
        setPos((prev) => Math.min(total * 2 - 1, Math.max(0, prev + delta)))
      }, [total])

      // Turn the page without losing the previous scroll position.
      useEffect(() => {
        if (stageRef.current != null) stageRef.current.scrollTop = 0
      }, [pos])

      useEffect(() => {
        const onKey = (event) => {
          const target = event.target
          const tag = target != null && typeof target.tagName === 'string' ? target.tagName : ''
          if (event.key === 'Escape') { setTocOpen(false); setSettingsOpen(false); setPromptEdit(null); return }
          // Space activates a focused button natively; never hijack it.
          if (['INPUT', 'TEXTAREA', 'SELECT', 'BUTTON'].includes(tag)) return
          if (tocOpen || settingsOpen || promptEdit != null) return
          if (event.key === 'ArrowRight' || event.key === ' ') { event.preventDefault(); go(1) }
          else if (event.key === 'ArrowLeft') { event.preventDefault(); go(-1) }
        }
        window.addEventListener('keydown', onKey)
        return () => window.removeEventListener('keydown', onKey)
      }, [go, tocOpen, settingsOpen, promptEdit])

      const applySettings = (patch) => {
        setSettings(patch)
        if (settingsTimerRef.current != null) clearTimeout(settingsTimerRef.current)
        settingsTimerRef.current = setTimeout(() => {
          settingsTimerRef.current = null
          api('/book', { method: 'PATCH', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ id: bookId, settings: patch }) }).catch(() => {})
        }, 500)
      }
      const generate = async (pageIndex, promptOverride, seedOverride) => {
        setManualBusy(true)
        // Reserve the slot immediately so the auto loop never double-fires
        // for this page while our request is in flight.
        requestedRef.current.add(`${bookId}:${pageIndex}`)
        try {
          await api('/book/illustrate', {
            method: 'POST', headers: { 'content-type': 'application/json' },
            body: JSON.stringify({ id: bookId, page: pageIndex, prompt: promptOverride, seed: seedOverride }),
          })
          status.reload()
        } catch (error) {
          requestedRef.current.delete(`${bookId}:${pageIndex}`)
          // 409 = engine busy with another job: make the wait explicit.
          if (error?.message?.includes?.('already running')) showNotice(t('book.queued'))
          else showNotice(`${t('book.illustFailed')}: ${error?.message ?? ''}`.slice(0, 160))
        }
        finally { setManualBusy(false) }
      }
      const regenerate = (pageIndex) => {
        generate(pageIndex, undefined, Math.floor(Math.random() * 2 ** 31))
      }

      const chapterTitle = page?.chapterTitle ?? ''
      // A terminal job for this page that produced no usable image = failure.
      const pageFailure = page != null && jobBookId === bookId && jobPage === page.index
        && (jobState === 'failed' || jobState === 'cancelled' || jobAttachError != null
          || (jobState === 'succeeded' && jobResultImage == null))
        ? String(jobAttachError ?? jobSnapshot?.error ?? (jobState === 'succeeded' ? 'no image attached' : jobState)).slice(0, 200)
        : null
      return h('div', { className: 'bk-reader', 'data-theme': settings.theme ?? 'paper', 'data-font': settings.serif ?? 'song' },
        h('style', null, `.bk-reader{--bk-font-size:${settings.fontSize ?? 19}px;--bk-line-height:${settings.lineHeight ?? 1.9}}`),
        h('div', { className: 'bk-topbar' },
          h('button', { className: 'bk-btn', onClick: onBack }, '‹ ' + t('book.backToShelf')),
          h('div', { style: { minWidth: 0, flex: 1 } },
            h('div', { className: 'bk-topTitle' }, meta.title),
            h('div', { className: 'bk-topChapter' }, chapterTitle)),
          h('button', { className: 'bk-btn', onClick: () => setTocOpen(true) }, '☰ ' + t('book.toc')),
          h('button', { className: 'bk-btn', ref: settingsAnchorRef, onClick: () => setSettingsOpen((open) => !open) }, 'Aa ' + t('book.settings'))),
        h('div', { className: 'bk-progress' },
          h('span', null, `${t('book.position', { page: currentPage + 1, chapter: chapterTitle.slice(0, 12) || '—' })}`),
          h('input', {
            type: 'range', className: 'bk-slider', min: 0, max: Math.max(0, total * 2 - 1), value: pos,
            'aria-label': t('book.progress'), 'aria-valuetext': `${currentPage + 1}/${total}`,
            onChange: (event) => setPos(Number(event.target.value)),
          }),
          h('span', null, t('book.pageOf', { a: currentPage + 1, b: total }))),
        pages.error != null && page == null && total > 0
          ? h('div', { style: { padding: '16px 20px' } }, h(ErrorNote, { error: pages.error, onRetry: pages.reload, t }))
          : null,
        h('div', { className: 'bk-stage', ref: stageRef },
          total === 0 ? h('div', { className: 'bk-pageText' }, t('common.empty')) : null,
          page == null && total > 0 ? h('div', { className: 'bk-pageText' }, h('div', { className: 'onyx-skeleton', style: { height: 200, opacity: 0.4 } })) : null,
          page != null && !isImage ? h('div', { key: `t${page.index}`, className: 'bk-pageText' },
            h('p', { className: 'bk-chapterMark' }, chapterTitle),
            (page.text ?? '').split('\n').map((paragraph, index) =>
              h('p', { key: index, className: 'bk-para' }, paragraph))) : null,
          page != null && isImage ? h(IllustrationPane, {
            key: `i${page.index}`, t, bookId, page,
            requested: requestedRef.current.has(`${bookId}:${page.index}`) || manualBusy,
            failure: pageFailure,
            jobSnapshot, onGenerate: (index) => generate(index), onEditPrompt: (index) => setPromptEdit({ page: index, initial: page.prompt }),
            onRegenerate: regenerate,
          }) : null,
          h('button', { className: 'bk-zone is-left', 'aria-label': t('book.prev'), onClick: () => go(-1) }),
          h('button', { className: 'bk-zone is-right', 'aria-label': t('book.next'), onClick: () => go(1) }),
          notice != null ? h('div', { className: 'bk-notice', role: 'status' }, notice) : null),
        h('div', { className: 'bk-pager' },
          h('button', { className: 'bk-btn', 'aria-label': t('book.prev'), onClick: () => go(-1), disabled: pos === 0 }, '‹'),
          h('span', null, `${isImage ? '🖼' : '¶'} ${t('book.pageOf', { a: pos + 1, b: total * 2 })}`),
          h('button', { className: 'bk-btn', 'aria-label': t('book.next'), onClick: () => go(1), disabled: pos >= total * 2 - 1 }, '›'),
          h('span', { style: { opacity: 0.6 } }, t('book.pagerHint'))),
        tocOpen ? h(TocDrawer, { t, book: meta, chapters, currentPage, onJump: (target) => { setPos(target * 2); setTocOpen(false) }, onClose: () => setTocOpen(false) }) : null,
        settingsOpen ? h(SettingsPopover, { t, settings, onChange: applySettings, onClose: () => setSettingsOpen(false), anchor: settingsAnchorRef.current }) : null,
        promptEdit != null ? h(PromptEditor, {
          t, initial: promptEdit.initial, busy: manualBusy,
          onApply: (value) => { generate(promptEdit.page, value); setPromptEdit(null) },
          onClose: () => setPromptEdit(null),
        }) : null)
    }

    function ShelfView({ t, onOpen }) {
      const books = useApi('/books', {})
      const [importOpen, setImportOpen] = useState(false)
      const list = Array.isArray(books.data) ? books.data : (books.data?.books ?? [])
      const remove = async (book) => {
        try { await api(`/book?id=${encodeURIComponent(book.id)}`, { method: 'DELETE' }); books.reload() }
        catch { books.reload() }
      }
      return h('div', { className: 'onyx-view' },
        h('div', { className: 'bk-shelfHead' },
          h('span', { className: 'onyx-sub' }, t('book.count', { count: list.length })),
          h('div', { className: 'onyx-spacer' }),
          h('button', { className: 'onyx-btn onyx-btnPrimary', onClick: () => setImportOpen(true) }, '+ ' + t('book.import'))),
        books.loading && list.length === 0 ? h('div', { className: 'bk-shelf' }, [0, 1, 2].map((i) => h('div', { key: i, className: 'onyx-skeleton', style: { height: 220, borderRadius: 12 } }))) : null,
        books.error != null ? h(ErrorNote, { error: books.error, onRetry: books.reload, t }) : null,
        !books.loading && list.length === 0 && books.error == null
          ? h('div', { className: 'bk-empty' }, t('book.empty'))
          : h('div', { className: 'bk-shelf' },
            list.map((book) => h(BookCard, { key: book.id, book, t, onOpen, onDelete: remove }))),
        importOpen ? h(ImportDialog, {
          t,
          onClose: () => setImportOpen(false),
          onImported: (meta) => { setImportOpen(false); books.reload(); onOpen(meta) },
        }) : null)
    }

    function BooksView({ t, compact = false }) {
      const [openBook, setOpenBook] = useState(null)
      const open = (bookOrMeta) => setOpenBook(typeof bookOrMeta === 'string' ? { id: bookOrMeta } : bookOrMeta)
      if (openBook != null) {
        return h(BookMetaLoader, { t, bookId: openBook.id, onBack: () => setOpenBook(null) })
      }
      return h(ShelfView, { t, onOpen: open, compact })
    }

    /** The shelf hands over a cached meta; the reader refetches fresh state. */
    function BookMetaLoader({ t, bookId, onBack }) {
      const meta = useApi(`/book?id=${encodeURIComponent(bookId)}`, {})
      if (meta.error != null) return h('div', null, h(ErrorNote, { error: meta.error, onRetry: meta.reload, t }))
      if (meta.data == null) return h('div', { className: 'onyx-skeleton', style: { height: 300 } })
      if (meta.data.error != null) return h('div', { className: 'onyx-error' }, meta.data.error)
      return h(ReaderView, { t, meta: meta.data, onBack })
    }

    /* --------------------------------------------------------------- app */
    function App({ compact = false, initialView = 'generate', t }) {
      const [view, setView] = useState(initialView)
      const [selectedDir, setSelectedDir] = useState(null)
      const onOpenRun = useCallback((dir) => { setSelectedDir(dir); setView('runs') }, [])
      // A light status probe only on the generate tab; the reader polls its own.
      const status = useApi(view === 'generate' ? '/generate/status' : null, { intervalMs: 3000 })
      const running = status.data != null && (status.data.state === 'running' || status.data.state === 'starting')
      const tabs = [
        ['generate', t('nav.generate')],
        ['books', t('nav.books')],
        ['runs', t('nav.runs')],
        ['model', t('nav.model')],
        ['operators', t('nav.operators')],
      ]
      // WAI-ARIA tabs: roving tabindex with arrow-key movement.
      const onKeyDown = (event) => {
        const index = tabs.findIndex(([id]) => id === view)
        let next = null
        if (event.key === 'ArrowRight') next = (index + 1) % tabs.length
        else if (event.key === 'ArrowLeft') next = (index - 1 + tabs.length) % tabs.length
        else if (event.key === 'Home') next = 0
        else if (event.key === 'End') next = tabs.length - 1
        if (next === null) return
        event.preventDefault()
        setView(tabs[next][0])
      }
      return h('div', { className: `onyx-root${compact ? ' is-compact' : ''}` },
        h('style', null, CSS),
        h('header', { className: 'onyx-head' },
          h('div', { className: 'onyx-brand' },
            h('span', { className: 'onyx-logoChip' }, h(StudioIcon, { size: 18 })),
            h('div', null,
              h('div', { className: 'onyx-title' }, t('view.title')),
              h('div', { className: 'onyx-sub' }, t('view.subtitle')))),
          running && status.data?.purpose?.kind === 'book'
            ? h('span', { className: 'onyx-status' },
              h('span', { className: 'onyx-dot is-running' }),
              `${t('book.illustration')} · p${status.data.purpose.page + 1}`)
            : h('span', { className: 'onyx-status' },
              h('span', { className: `onyx-dot${running ? ' is-running' : ' is-ok'}` }),
              running ? t('status.busy') : t('status.idle')),
          h('nav', { className: 'onyx-nav', role: 'tablist', onKeyDown }, tabs.map(([id, label]) =>
            h('button', {
              key: id, role: 'tab', 'aria-selected': view === id, tabIndex: view === id ? 0 : -1,
              className: 'onyx-navBtn', onClick: () => setView(id),
            }, label)))),
        h('main', { className: `onyx-body${compact ? ' is-compact' : ''}` },
          view === 'generate' ? h('div', { className: 'onyx-view' }, h(GenerateView, { t, onOpenRun })) : null,
          view === 'books' ? h(BooksView, { t, compact }) : null,
          view === 'runs' ? h('div', { className: 'onyx-view' }, h(RunsView, { t, selectedDir, onSelect: setSelectedDir, compact })) : null,
          view === 'model' ? h(ModelView, { t }) : null,
          view === 'operators' ? h(OperatorsView, { t }) : null))
    }

    function StudioIcon({ size = 18 }) {
      return h('svg', { width: size, height: size, viewBox: '0 0 36 36', fill: 'none', 'aria-hidden': true },
        h('circle', { cx: 18, cy: 18, r: 15, fill: 'rgba(255,255,255,.16)' }),
        h('path', { d: 'M18 6l4.2 8.4L30 18l-7.8 3.6L18 30l-4.2-8.4L6 18l7.8-3.6z', fill: '#fff', opacity: 0.92 }),
        h('circle', { cx: 18, cy: 18, r: 3.1, fill: '#4f8fd6' }))
    }

    /* -------------------------------------------------------------- mount */
    return {
      inject: ['locale', 'slots'],
      apply(ctx) {
        let t = (key, params) => {
          let text = dict.en[key] ?? key
          if (params != null) for (const [name, value] of Object.entries(params)) text = text.replace(`{${name}}`, String(value))
          return text
        }
        const disposers = []
        try {
          const releaseLocale = ctx.locale.register(NS, dict)
          disposers.push(() => { try { releaseLocale() } catch { /* gone */ } })
          const bound = ctx.locale.bind(NS)
          t = (key, params) => bound(key, params)
        } catch { /* The English fallback handles hosts without the locale service. */ }

        const registerView = () => ctx.slots.register({ name: 'main', key: 'onyx-studio' }, () => h(App, { t }))
        const registerConversation = () => ctx.slots.register(
          { name: 'conversation.view', id: 'onyx-studio', order: 45, label: () => t('view.title') },
          () => h(App, { t }))
        const registerPanelIcon = () => ctx.slots.register(
          { name: 'sidebar.panellist', id: 'onyx-studio', order: 45, label: () => t('view.title') },
          (props) => h(StudioIcon, { size: typeof props?.size === 'number' ? props.size : 18 }))

        disposers.push(ctx.slots.inject('main', registerView))
        disposers.push(ctx.slots.inject('conversation.view', registerConversation))
        disposers.push(ctx.slots.inject('sidebar.panellist', registerPanelIcon))

        const releaseTabs = ctx.inject(['sidebarRightTabs'], (child) => {
          const inner = []
          inner.push(child.effect(() => child.sidebarRightTabs.register({
            id: TAB_ID,
            kind: 'onyx-studio',
            title: () => t('tab.title'),
            guide: [{ id: 'onyx-studio', order: 50, title: () => t('tab.guide.title'), description: () => t('tab.guide.description'), icon: StudioIcon }],
          }), 'onyx-studio: sidebar tab'))
          inner.push(child.slots.inject('sidebar.right.pane.tab', () => child.slots.register(
            { name: 'sidebar.right.pane.tab', key: TAB_ID }, () => h(App, { compact: true, initialView: 'generate', t }))))
          inner.push(child.slots.inject('sidebar.right.pane.tab.title', () => child.slots.register(
            { name: 'sidebar.right.pane.tab.title', key: TAB_ID }, () => h('span', null, t('tab.title')))))
          return () => { for (const dispose of inner.reverse()) { try { dispose() } catch { /* gone */ } } }
        })
        disposers.push(() => { try { releaseTabs() } catch { /* gone */ } })

        return () => { for (const dispose of disposers.reverse()) { try { dispose() } catch { /* gone */ } } }
      },
    }
  },
})

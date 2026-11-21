/**
 * Host half of the Onyx Studio plugin.
 *
 * Bridges the onyx-infer local inference engine into DeepSeek Harness:
 * - serves a JSON/asset HTTP surface under /onyx-studio/ for the Web panel,
 * - runs image generation through `onyx.replay` + `onyx.native` with a live
 *   process sampler (stage detection + RSS time series),
 * - exposes the same operations to agents as tools.
 *
 * No @deepseek-ai imports: the plugin runs from a plain checkout with only
 * Node built-ins, so it installs without a dependency step.
 */
import { spawn } from 'node:child_process'
import fs from 'node:fs/promises'
import { existsSync } from 'node:fs'
import path from 'node:path'
import readline from 'node:readline'
import { fileURLToPath } from 'node:url'

const PLUGIN_DIR = path.dirname(fileURLToPath(import.meta.url))
const ROUTE_BASE = '/onyx-studio'
const STUDIO_CASES_DIR = 'artifacts/studio/cases'
const RUNS_ROOT = 'artifacts/benchmarks/onyx-native-h3'
const BOOKS_ROOT = 'artifacts/studio/books'
const BOOK_ID_RE = /^b-[0-9a-z]+-[0-9a-z]{2,8}$/
const BOOK_FORMATS = ['txt', 'md', 'epub']
const BOOK_STYLES = ['ink', 'cinematic', 'storybook', 'minimal']
const MAX_BODY_BYTES = 1_000_000
const MAX_BOOK_BODY_BYTES = 64_000_000
const MAX_RUNS_LISTED = 240
const STAGE_WEIGHTS = { text: 0.1, dit: 0.78, vae: 0.12 }

/* ------------------------------------------------------------------ utils */

function nowMs() { return Date.now() }

function sendJson(res, code, value) {
  const body = JSON.stringify(value)
  res.writeHead(code, {
    'content-type': 'application/json; charset=utf-8',
    'cache-control': 'no-store',
  })
  res.end(body)
}

function readJsonBody(req, limit = MAX_BODY_BYTES) {
  return new Promise((resolve, reject) => {
    let size = 0
    let tooLarge = false
    const chunks = []
    req.on('data', (chunk) => {
      size += chunk.length
      if (size > limit) {
        // Drain instead of destroying so the client gets a real 413 JSON.
        tooLarge = true
        chunks.length = 0
        return
      }
      if (!tooLarge) chunks.push(chunk)
    })
    req.on('end', () => {
      if (tooLarge) return reject(new Error(`request body too large (limit ${limit} bytes)`))
      if (chunks.length === 0) return resolve({})
      try { resolve(JSON.parse(Buffer.concat(chunks).toString('utf8'))) }
      catch (error) { reject(new Error(`invalid JSON body: ${error.message}`)) }
    })
    req.on('error', reject)
  })
}

async function readJsonFile(file) {
  return JSON.parse(await fs.readFile(file, 'utf8'))
}

function safeNumber(value, fallback) {
  const n = Number(value)
  return Number.isFinite(n) ? n : fallback
}

function shortSha(artifactRef) {
  const sha = artifactRef?.sha256
  if (typeof sha !== 'string' || sha.length === 0) return undefined
  try { return Buffer.from(sha, 'base64').toString('hex').slice(0, 16) }
  catch { return sha.slice(0, 16) }
}

function gb(bytes) { return bytes / 1024 ** 3 }

/* --------------------------------------------------------- engine resolve */

function runCapture(ctx, command, args, options = {}) {
  return new Promise((resolve) => {
    let child
    try { child = spawn(command, args, { stdio: ['ignore', 'pipe', 'pipe'], ...options }) }
    catch (error) { resolve({ code: -1, stdout: '', stderr: String(error) }); return }
    let stdout = ''
    let stderr = ''
    let timedOut = false
    const killer = options.timeoutMs !== undefined
      ? setTimeout(() => { timedOut = true; try { child.kill('SIGKILL') } catch { /* gone */ } }, options.timeoutMs)
      : null
    if (killer !== null) killer.unref?.()
    child.stdout.on('data', (c) => { stdout += c })
    child.stderr.on('data', (c) => { stderr += c })
    child.on('error', (error) => resolve({ code: -1, stdout, stderr: stderr + String(error) }))
    child.on('close', (code) => {
      if (killer !== null) clearTimeout(killer)
      resolve(timedOut
        ? { code: code ?? -1, stdout, stderr: `${stderr}\nprocess killed after ${options.timeoutMs}ms` }
        : { code: code ?? -1, stdout, stderr })
    })
  })
}

async function resolveEngine(ctx, config) {
  const problems = []
  const candidates = []
  if (typeof config.engineRoot === 'string' && config.engineRoot.trim() !== '') {
    candidates.push(path.resolve(config.engineRoot))
  } else {
    // The plugin ships inside an onyx-infer worktree; the runnable engine is
    // the repository's main worktree, where .venv, .build and artifacts live.
    const git = await runCapture(ctx, 'git', ['-C', PLUGIN_DIR, 'rev-parse', '--path-format=absolute', '--git-common-dir'])
    const common = git.stdout.trim()
    if (common !== '' && existsSync(common)) candidates.push(path.dirname(common))
    candidates.push(path.resolve(PLUGIN_DIR, '..', '..', '..'))
  }
  let root = null
  for (const candidate of candidates) {
    if (!existsSync(path.join(candidate, 'pyproject.toml'))) continue
    if (!existsSync(path.join(candidate, 'src/onyx'))) continue
    root = candidate
    break
  }
  if (root === null) return { root: null, problems: ['no onyx-infer checkout found; set engineRoot in the plugin config'] }
  const venvPython = path.join(root, '.venv/bin/python')
  const nativeBin = path.join(root, '.build/onyx-native')
  const modelsDir = path.join(root, 'artifacts/models/h3-w4')
  if (!existsSync(venvPython)) problems.push(`missing ${path.relative(root, venvPython)}; run: uv sync`)
  if (!existsSync(nativeBin)) problems.push(`missing ${path.relative(root, nativeBin)}; build the native runtime first`)
  if (!existsSync(path.join(modelsDir, 'manifest.json'))) problems.push(`missing ${path.relative(root, modelsDir)}; package the W4 model first`)
  return { root, problems, venvPython, nativeBin, modelsDir }
}

/* ------------------------------------------------------------- data reads */

async function readManifestSummary(engine) {
  const manifest = await readJsonFile(path.join(engine.modelsDir, 'manifest.json'))
  const components = (manifest.components ?? []).map((component) => {
    const artifacts = component.artifacts ?? []
    return {
      role: component.role ?? 'unknown',
      bytes: artifacts.reduce((sum, a) => sum + safeNumber(a.byteSize ?? a.byte_size, 0), 0),
      files: artifacts.map((a) => ({
        kind: a.kind ?? '', bytes: safeNumber(a.byteSize ?? a.byte_size, 0),
        locator: a.localLocator ?? a.local_locator ?? '',
      })),
    }
  })
  const graphs = (manifest.graphs ?? []).map((graph) => {
    const nodes = graph.nodes ?? []
    const blockish = nodes.filter((n) => /block|layer/i.test(n.op ?? n.operator ?? ''))
    const representative = blockish.length > 0 ? blockish[Math.floor(blockish.length / 2)] : nodes[0]
    return {
      id: graph.graphId ?? graph.graph_id ?? '',
      nodes: nodes.length,
      blockNodes: blockish.length,
      representativeAttributes: representative?.attributes ?? representative?.parameters ?? {},
    }
  })
  return {
    modelId: manifest.modelId ?? manifest.model_id ?? 'unknown',
    sourceRevision: (manifest.sourceRevision ?? manifest.source_revision ?? '').slice(0, 12),
    semanticDigest: manifest.semanticDigest ?? manifest.semantic_digest ?? '',
    components,
    graphs,
    stages: manifest.stages ?? [],
    recipes: manifest.recipes ?? [],
    quantization: manifest.quantization ?? {},
    numericProfiles: manifest.numericProfiles ?? manifest.numeric_profiles ?? [],
  }
}

function tensorBytes(entry) {
  const shape = entry.shape ?? []
  let elements = 1
  for (const dim of shape) elements *= safeNumber(dim, 1)
  const dtype = String(entry.dtype ?? '').toUpperCase()
  // Q4S64: one 4-bit nibble per element plus one BF16 scale per group of 64.
  const bytesPerElement = { Q4S64: 0.5 + 2 / 64, BF16: 2, F16: 2, F32: 4, FP32: 4, UINT8: 1 }[dtype] ?? 2
  return { elements, bytes: elements * bytesPerElement }
}

async function readPackReports(engine) {
  const out = []
  for (const name of ['transformer.onxpack.report.json', 'text_encoder.onxpack.report.json']) {
    const file = path.join(engine.modelsDir, name)
    if (!existsSync(file)) continue
    const report = await readJsonFile(file)
    const tensors = report.tensors ?? []
    const byDtype = new Map()
    const byBlock = new Map()
    let totalElements = 0
    for (const entry of tensors) {
      const { elements, bytes } = tensorBytes(entry)
      totalElements += elements
      const dtype = String(entry.dtype ?? 'unknown')
      const slot = byDtype.get(dtype) ?? { dtype, tensors: 0, bytes: 0, elements: 0 }
      slot.tensors += 1; slot.bytes += bytes; slot.elements += elements
      byDtype.set(dtype, slot)
      const match = /(?:blocks|layers)[.](\d+)[.]/.exec(entry.name ?? '')
      const key = match === null ? 'other' : `block ${Number(match[1])}`
      const block = byBlock.get(key) ?? { block: key, bytes: 0, tensors: 0 }
      block.bytes += bytes; block.tensors += 1
      byBlock.set(key, block)
    }
    out.push({
      component: name.startsWith('transformer') ? 'transformer' : 'text_encoder',
      tensors: tensors.length,
      totalElements,
      byDtype: [...byDtype.values()].sort((a, b) => b.bytes - a.bytes),
      byBlock: [...byBlock.values()].sort((a, b) => a.block.localeCompare(b.block, undefined, { numeric: true })),
    })
  }
  return out
}

async function readTarget(engine) {
  const dir = path.join(engine.root, 'config/targets')
  if (!existsSync(dir)) return null
  const first = (await fs.readdir(dir)).find((f) => f.endsWith('.json'))
  return first === undefined ? null : await readJsonFile(path.join(dir, first))
}

const DIT_STEP = /native DiT step (\d+)\/(\d+), Metal allocated ([\d.]+) GiB/
const TEXT_LAYER = /native text layer (\d+)\/(\d+)/

function parseProcessLog(text) {
  const ditSteps = []
  const textLayers = []
  let backend = null
  const consume = (line) => {
    let match = DIT_STEP.exec(line)
    if (match !== null) {
      ditSteps.push({ step: Number(match[1]), steps: Number(match[2]), metalGiB: Number(match[3]) })
      return
    }
    match = TEXT_LAYER.exec(line)
    if (match !== null) textLayers.push({ layer: Number(match[1]), layers: Number(match[2]) })
  }
  for (const line of text.split('\n')) {
    if (!line.startsWith('{')) { consume(line); continue }
    try {
      const value = JSON.parse(line)
      if (value?.stage_seconds !== undefined) {
        backend = value
        // The native pipeline's stderr progress is embedded in stage_logs.
        for (const log of Object.values(value.stage_logs ?? {})) {
          for (const embedded of String(log?.stderr ?? '').split('\n')) consume(embedded)
        }
      }
    } catch { /* not json */ }
  }
  return { backend, ditSteps, textLayers }
}

async function readRunDetail(engine, relDir) {
  const dir = path.join(engine.root, RUNS_ROOT, relDir)
  const runFile = path.join(dir, 'run.json')
  if (!existsSync(runFile)) return null
  const run = await readJsonFile(runFile)
  let process = null
  try { process = parseProcessLog(await fs.readFile(path.join(dir, 'process.log'), 'utf8')) }
  catch { /* optional */ }
  const stages = (run.stages ?? []).map((stage) => ({
    id: stage.stage?.stageId ?? stage.stage?.stage_id ?? '?',
    seconds: safeNumber(stage.wallTimeNs ?? stage.wall_time_ns, 0) / 1e9,
  }))
  const totalSeconds = safeNumber(run.totalWallTimeNs ?? run.total_wall_time_ns, 0) / 1e9
  const metrics = {}
  for (const metric of run.metrics ?? []) {
    metrics[metric.name ?? metric.metricId ?? '?'] = safeNumber(metric.value ?? metric.valueInt, metric.value)
  }
  const output = (run.outputs ?? [])[0]
  const outputRel = typeof output?.localLocator === 'string' || typeof output?.local_locator === 'string'
    ? path.relative(engine.root, output?.localLocator ?? output?.local_locator)
    : undefined
  return {
    dir: relDir,
    runId: run.runId ?? run.run_id,
    requestId: run.requestId ?? run.request_id,
    caseId: relDir.split(path.sep)[0] ?? '',
    planId: run.planId ?? run.plan_id,
    totalSeconds,
    stages,
    peakMemoryBytes: safeNumber(run.peakMemoryBytes ?? run.peak_memory_bytes, 0),
    metrics,
    output: output == null ? null : {
      bytes: safeNumber(output.byteSize ?? output.byte_size, 0),
      sha16: shortSha(output),
      relPath: outputRel,
      exists: outputRel !== undefined && existsSync(path.join(engine.root, outputRel)),
    },
    build: run.build ?? null,
    device: run.device ?? null,
    replayRequest: run.replayRequest ?? run.replay_request ?? null,
    events: (run.events ?? []).map((event) => ({
      type: event.eventType ?? event.type ?? '',
      monotonicNs: safeNumber(event.monotonicTimeNs ?? event.monotonic_time_ns, 0),
    })),
    process,
  }
}

async function listRuns(engine, limit = MAX_RUNS_LISTED) {
  const root = path.join(engine.root, RUNS_ROOT)
  if (!existsSync(root)) return []
  const entries = []
  for (const caseDir of await fs.readdir(root)) {
    const casePath = path.join(root, caseDir)
    let caseStat
    try { caseStat = await fs.stat(casePath) } catch { continue }
    if (!caseStat.isDirectory()) continue
    for (const runDir of await fs.readdir(casePath)) {
      const runFile = path.join(casePath, runDir, 'run.json')
      try {
        const stat = await fs.stat(runFile)
        entries.push({ relDir: `${caseDir}/${runDir}`, mtimeMs: stat.mtimeMs })
      } catch { /* no run.json */ }
    }
  }
  entries.sort((a, b) => b.mtimeMs - a.mtimeMs)
  const runs = []
  for (const entry of entries.slice(0, limit)) {
    try {
      const run = await readJsonFile(path.join(engine.root, RUNS_ROOT, entry.relDir, 'run.json'))
      const output = (run.outputs ?? [])[0]
      const locator = output?.localLocator ?? output?.local_locator
      const outputRel = typeof locator === 'string' && locator !== '' ? path.relative(engine.root, locator).replaceAll(path.sep, '/') : undefined
      runs.push({
        dir: entry.relDir,
        caseId: entry.relDir.split('/')[0],
        runId: run.runId ?? run.run_id,
        planId: run.planId ?? run.plan_id,
        totalSeconds: safeNumber(run.totalWallTimeNs ?? run.total_wall_time_ns, 0) / 1e9,
        peakMemoryBytes: safeNumber(run.peakMemoryBytes ?? run.peak_memory_bytes, 0),
        sha16: shortSha(output),
        revision: String(run.build?.sourceRevision ?? run.build?.source_revision ?? '').slice(0, 12),
        mtimeMs: entry.mtimeMs,
        outputRel,
        outputExists: outputRel !== undefined && existsSync(path.join(engine.root, outputRel)),
      })
    } catch { /* unreadable run */ }
  }
  return runs
}

async function readOperators(engine) {
  const out = { comparisons: [], raw: [] }
  const resultsDir = path.join(engine.root, 'benchmarks/results')
  if (existsSync(resultsDir)) {
    for (const file of await fs.readdir(resultsDir)) {
      if (!file.startsWith('m5-max-36gb-q4-operator') || !file.endsWith('.json')) continue
      try {
        const doc = await readJsonFile(path.join(resultsDir, file))
        for (const comparison of doc.comparisons ?? []) {
          out.comparisons.push({
            file,
            caseId: comparison.caseId ?? comparison.case_id,
            shape: comparison.shape ?? {},
            onyx: {
              implementation: comparison.onyx?.implementation,
              wallMedianMs: comparison.onyx?.wallMedianMs ?? comparison.onyx?.wall_median_ms,
              gpuMedianMs: comparison.onyx?.gpuMedianMs ?? comparison.onyx?.gpu_median_ms,
              correct: comparison.onyx?.correct,
            },
            mlx: {
              wallMedianMs: comparison.mlx?.wallMedianMs ?? comparison.mlx?.wall_median_ms,
            },
            ratio: comparison.onyxOverMlxTimeRatio ?? comparison.onyx_over_mlx_time_ratio,
          })
        }
      } catch { /* skip malformed */ }
    }
  }
  const rawDir = path.join(engine.root, 'artifacts/benchmarks/onyx')
  if (existsSync(rawDir)) {
    for (const file of await fs.readdir(rawDir)) {
      if (!file.endsWith('.json') || file.includes('record')) continue
      try {
        const doc = await readJsonFile(path.join(rawDir, file))
        if (doc.effective_tflops === undefined && doc.gpu_median_ms === undefined) continue
        out.raw.push({
          file,
          case: doc.case ?? file.replace('.json', ''),
          gpuMedianMs: doc.gpu_median_ms,
          wallMedianMs: doc.wall_median_ms,
          effectiveTflops: doc.effective_tflops,
          effectiveBandwidthGbS: doc.effective_bandwidth_gb_per_s,
          metalAllocatedBytes: doc.metal_allocated_bytes,
        })
      } catch { /* skip malformed */ }
    }
  }
  return out
}

async function readOverview(engine) {
  const runs = await listRuns(engine, 60)
  const native = runs.filter((run) => run.totalSeconds > 0)
  const recent = native.slice(0, 5)
  const git = await runCapture(null, 'git', ['-C', engine.root, 'log', '-1', '--format=%h %cd', '--date=short'])
  return {
    engineRoot: engine.root,
    git: git.code === 0 ? git.stdout.trim() : 'unavailable',
    runCount: runs.length,
    lastTotalSeconds: recent[0]?.totalSeconds ?? null,
    medianTotalSeconds: native.length === 0 ? null
      : native.slice(0, 20).map((r) => r.totalSeconds).sort((a, b) => a - b)[Math.floor(Math.min(native.length, 20) / 2)],
    recentRuns: recent,
  }
}

/* ------------------------------------------------------------- book store */

/**
 * The reader keeps every book inside the engine's artifacts tree, so generated
 * illustrations are servable through the existing /onyx-studio/files route
 * (which already constrains paths to the artifacts subtree).
 */
function writeJsonFile(file, value) {
  return fs.writeFile(file, JSON.stringify(value), 'utf8')
}

/** Serialize per-book read-modify-write cycles (book.json / pages.json). */
const BOOK_LOCKS = new Map()

function withBookLock(key, task) {
  const previous = BOOK_LOCKS.get(key) ?? Promise.resolve()
  const next = previous.then(task, task)
  BOOK_LOCKS.set(key, next)
  void next.catch(() => {}).then(() => {
    if (BOOK_LOCKS.get(key) === next) BOOK_LOCKS.delete(key)
  })
  return next
}

function bookDir(engine, id) {
  if (!BOOK_ID_RE.test(id ?? '')) throw new Error('invalid book id')
  const root = path.resolve(engine.root, BOOKS_ROOT)
  const dir = path.resolve(root, id)
  if (!dir.startsWith(root + path.sep)) throw new Error('invalid book id')
  return dir
}

async function readBookMeta(engine, id) {
  return readJsonFile(path.join(bookDir(engine, id), 'book.json'))
}

async function readPagesDoc(engine, id) {
  return readJsonFile(path.join(bookDir(engine, id), 'pages.json'))
}

async function listBooks(engine) {
  const root = path.resolve(engine.root, BOOKS_ROOT)
  if (!existsSync(root)) return []
  const books = []
  for (const entry of await fs.readdir(root)) {
    try { books.push(await readBookMeta(engine, entry)) }
    catch { /* incomplete import or foreign directory */ }
  }
  books.sort((a, b) => safeNumber(b.updatedAtMs, 0) - safeNumber(a.updatedAtMs, 0))
  return books
}

async function importBook(engine, request) {
  if (!BOOK_FORMATS.includes(request.format)) throw new Error('format must be txt, md or epub')
  const payload = Buffer.from(String(request.payloadB64 ?? ''), 'base64')
  if (payload.length === 0) throw new Error('payloadB64 is required')
  if (!engine.venvPython || !existsSync(engine.venvPython)) {
    throw new Error('engine venv missing; run `uv sync` in the onyx-infer checkout first')
  }
  const booksRoot = path.resolve(engine.root, BOOKS_ROOT)
  await fs.mkdir(booksRoot, { recursive: true })
  const bookId = `b-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`
  const outDir = path.join(booksRoot, bookId)
  const payloadFile = path.join(booksRoot, `.import-${bookId}.payload`)
  await fs.writeFile(payloadFile, payload)
  const title = String(request.title ?? '').trim().slice(0, 80) || 'Untitled'
  const style = BOOK_STYLES.includes(request.style) ? request.style : 'ink'
  const charsPerPage = Math.min(2000, Math.max(200, Math.round(safeNumber(request.charsPerPage, 800))))
  const args = [
    path.join(PLUGIN_DIR, 'book_import.py'),
    '--format', request.format,
    '--payload-file', payloadFile,
    '--book-id', bookId,
    '--title', title,
    '--author', String(request.author ?? '').trim().slice(0, 60),
    '--style', style,
    '--chars-per-page', String(charsPerPage),
    '--out-dir', outDir,
  ]
  try {
    const result = await runCapture(null, engine.venvPython, args, { timeoutMs: 180_000 })
    let summary = null
    try { summary = JSON.parse(result.stdout.trim().split('\n').filter(Boolean).pop()) } catch { /* fallthrough */ }
    if (result.code !== 0 || summary?.ok !== true) {
      await fs.rm(outDir, { recursive: true, force: true })
      throw new Error(summary?.error ?? `book import failed: ${result.stderr.slice(-300) || `exit ${result.code}`}`)
    }
  } finally {
    await fs.rm(payloadFile, { force: true }).catch(() => {})
  }
  return withBookLock(bookId, () => readBookMeta(engine, bookId))
}

async function patchBook(engine, request) {
  const meta = await readBookMeta(engine, request.id)
  if (request.progress != null) {
    const lastPage = Math.round(safeNumber(request.progress.lastPage, safeNumber(meta.progress?.lastPage, 0)))
    meta.progress = { lastPage: Math.min(Math.max(0, lastPage), Math.max(0, safeNumber(meta.pageCount, 1) - 1)) }
  }
  if (request.settings != null && typeof request.settings === 'object') {
    const settings = { ...(meta.settings ?? {}) }
    for (const key of ['theme', 'fontSize', 'lineHeight', 'serif', 'autoIllustrate', 'aheadCount']) {
      if (request.settings[key] === undefined) continue
      const value = request.settings[key]
      if (key === 'theme') settings.theme = ['paper', 'night', 'ink'].includes(value) ? value : 'paper'
      else if (key === 'serif') settings.serif = ['song', 'kai'].includes(value) ? value : 'song'
      else if (key === 'fontSize') settings.fontSize = Math.min(32, Math.max(12, Math.round(safeNumber(value, 19))))
      else if (key === 'lineHeight') settings.lineHeight = Math.min(3, Math.max(1.2, safeNumber(value, 1.9)))
      else if (key === 'aheadCount') settings.aheadCount = Math.min(3, Math.max(0, Math.round(safeNumber(value, 2))))
      else if (key === 'autoIllustrate') settings.autoIllustrate = value === true
    }
    meta.settings = settings
  }
  meta.updatedAtMs = nowMs()
  await writeJsonFile(path.join(bookDir(engine, meta.id), 'book.json'), meta)
  return meta
}

async function deleteBook(engine, id) {
  const dir = bookDir(engine, id)
  await fs.rm(dir, { recursive: true, force: true })
  return { deleted: id }
}

async function bookPages(engine, id, from, to) {
  const pagesDoc = await readPagesDoc(engine, id)
  const total = pagesDoc.pages.length
  const start = Math.min(Math.max(0, Math.round(safeNumber(from, 0))), Math.max(0, total - 1))
  const end = Math.min(total, Math.max(start + 1, Math.round(safeNumber(to, start + 8))))
  return {
    total,
    chapters: pagesDoc.chapters,
    pages: pagesDoc.pages.slice(start, end).map((page, offset) => ({ index: start + offset, ...page })),
  }
}

/** Copy a finished run's PNG into the book and index it on the page. */
async function attachIllustration(engine, context, job) {
  return withBookLock(context.bookId, async () => {
    const result = job.result
    if (result?.outputRel == null) throw new Error(result?.error ?? 'generation produced no output')
    const source = path.resolve(engine.root, result.outputRel)
    if (!existsSync(source)) throw new Error(`output file missing: ${result.outputRel}`)
    const dir = bookDir(engine, context.bookId)
    const pagesFile = path.join(dir, 'pages.json')
    const pagesDoc = await readJsonFile(pagesFile)
    const page = pagesDoc.pages[context.page]
    if (page == null) throw new Error(`page out of range: ${context.page}`)
    const imagesDir = path.join(dir, 'images')
    await fs.mkdir(imagesDir, { recursive: true })
    const fileName = `p-${String(context.page).padStart(4, '0')}.png`
    await fs.copyFile(source, path.join(imagesDir, fileName))
    const replacing = page.image != null
    page.image = {
      file: `${BOOKS_ROOT}/${context.bookId}/images/${fileName}`,
      prompt: job.request.prompt,
      seed: job.request.seed,
      width: job.request.width,
      height: job.request.height,
      runDir: result.runDir ?? null,
      sha16: result.sha16 ?? null,
      createdAtMs: nowMs(),
    }
    await writeJsonFile(pagesFile, pagesDoc)
    const metaFile = path.join(dir, 'book.json')
    const meta = await readJsonFile(metaFile)
    if (meta.cover == null) meta.cover = page.image.file
    if (!replacing) meta.illustrated = Math.min(safeNumber(meta.illustrated, 0) + 1, safeNumber(meta.pageCount, 0))
    meta.updatedAtMs = nowMs()
    await writeJsonFile(metaFile, meta)
    return page.image
  })
}



/* ---------------------------------------------------------- generation job */

class GenerationJob {
  constructor(engine, config, request, logger) {
    this.engine = engine
    this.config = config
    this.request = request
    this.logger = logger
    this.state = 'starting'
    this.startedAtMs = nowMs()
    this.stage = null
    this.samples = []
    this.logTail = []
    this.result = null
    this.error = null
    this.processes = []
  }

  pushLog(line) {
    if (line.trim() === '') return
    this.logTail.push(line.trimEnd())
    if (this.logTail.length > 40) this.logTail.splice(0, this.logTail.length - 40)
  }

  pushSample(sample) {
    this.samples.push(sample)
    const max = safeNumber(this.config.maxSamples, 4000)
    if (this.samples.length > max) this.samples.splice(0, this.samples.length - max)
    if (typeof sample.stage === 'string' && sample.stage !== this.stage) this.stage = sample.stage
  }

  snapshot() {
    const elapsedMs = nowMs() - this.startedAtMs
    const stageWeights = STAGE_WEIGHTS[this.stage] ?? 0.05
    const stageElapsed = this.stageSinceMs === undefined ? null : nowMs() - this.stageSinceMs
    // Down-sample for the wire: the sparkline needs the trend, not every tick.
    const stride = Math.max(1, Math.ceil(this.samples.length / 240))
    const samples = stride === 1 ? this.samples : this.samples.filter((_, index) => index % stride === 0)
    return {
      state: this.state,
      request: this.request,
      purpose: this.request?.context ?? null,
      startedAtMs: this.startedAtMs,
      elapsedMs,
      stage: this.stage,
      stageElapsedMs: stageElapsed,
      progressHint: stageWeights,
      samples,
      logTail: [...this.logTail],
      result: this.result,
      attachError: this.attachError ?? null,
      error: this.error,
    }
  }

  async start() {
    const { engine, request } = this
    const caseId = `studio-${request.width}x${request.height}-s${request.seed}-${new Date().toISOString().replace(/[-:TZ.]/g, '').slice(0, 14)}`
    const caseDir = path.join(engine.root, STUDIO_CASES_DIR)
    await fs.mkdir(caseDir, { recursive: true })
    const packageReport = await readJsonFile(path.join(engine.modelsDir, 'package-report.json'))
    const caseFile = path.join(caseDir, `${caseId}.json`)
    await fs.writeFile(caseFile, JSON.stringify({
      protocol: 'onyx.loop.v1',
      case_id: caseId,
      request: {
        protocol: 'onyx.job.v1',
        request_id: caseId,
        model: {
          model_id: 'minimax-h3-image',
          source_revision: packageReport.source_revision,
          recipe_id: 'h3.pdd6.v1',
        },
        prompt: request.prompt,
        size: { width: request.width, height: request.height },
        seed: String(request.seed),
        image_count: 1,
        output: { format: 'PNG', color_space: 'srgb' },
      },
      target_id: 'mac17-14-m5-max-36gb',
      primary_metric: 'complete_generation_wall_time_ns',
      warmups: 0,
      measured_runs: 1,
      max_experiments: 2,
      max_wall_time_ns: String(20 * 60 * 1e9),
    }, null, 2))
    const python = engine.venvPython
    const args = [
      '-m', 'onyx.replay',
      '--case', caseFile,
      '--output-root', path.join(engine.root, RUNS_ROOT),
      '--backend-version', 'onyx-native-cpp-metal',
      '--plan-id', this.config.planId ?? 'native.mpsgraph.q4.stream.v1',
      '--numeric-profile', 'w4.bf16.fp32-sensitive',
      '--quantization-profile', 'onyx-w4s64-linear-v1',
      '--resolved-source-revision', packageReport.source_revision,
      '--manifest-digest', packageReport.semantic_digest,
      '--rng-algorithm', 'numpy.pcg64.standard_normal',
      '--rng-version', '2.5.4',
      '--cache-state', 'unspecified',
      '--', python, '-m', 'onyx.native',
      '--models', engine.modelsDir,
      '--native-bin', engine.nativeBin,
      '--output', '{output}',
      '--prompt', request.prompt,
      '--width', String(request.width),
      '--height', String(request.height),
      '--seed', String(request.seed),
    ]
    this.caseId = caseId
    this.state = 'running'
    const child = spawn(python, args, { cwd: engine.root, stdio: ['ignore', 'pipe', 'pipe'] })
    this.processes.push(child)
    let stdout = ''
    readline.createInterface({ input: child.stdout }).on('line', (line) => { stdout += `${line}\n`; this.pushLog(line) })
    readline.createInterface({ input: child.stderr }).on('line', (line) => this.pushLog(line))
    this.startSampler(child.pid)
    child.on('error', (error) => this.finish(`failed to spawn replay: ${error.message}`))
    child.on('close', (code) => {
      this.stopSampler()
      if (this.state !== 'running') return
      if (code !== 0) {
        this.finish(`replay exited with code ${code}; last output: ${this.logTail.slice(-4).join(' | ')}`)
        return
      }
      // replay prints one pretty-printed JSON document.
      let parsed
      try { parsed = JSON.parse(stdout) }
      catch {
        const start = stdout.indexOf('{')
        const end = stdout.lastIndexOf('}')
        if (start >= 0 && end > start) {
          try { parsed = JSON.parse(stdout.slice(start, end + 1)) } catch { /* fallthrough */ }
        }
      }
      if (parsed !== undefined && parsed.run_id !== undefined) this.finish(null, parsed)
      else this.finish(`replay output was not JSON: ${stdout.slice(0, 200)}`)
    })
  }

  startSampler(pid) {
    const sampler = spawn(this.engine.venvPython, [path.join(PLUGIN_DIR, 'sampler.py'), String(pid), String(safeNumber(this.config.sampleIntervalMs, 100) / 1000)], { stdio: ['ignore', 'pipe', 'ignore'] })
    this.sampler = sampler
    readline.createInterface({ input: sampler.stdout }).on('line', (line) => {
      try {
        const value = JSON.parse(line)
        if (typeof value.rss === 'number' && value.rss > 0) this.pushSample({ t: value.t, rss: value.rss, stage: value.stage ?? null })
      } catch { /* ignore */ }
    })
    sampler.on('error', () => { /* sampler is best-effort */ })
  }

  stopSampler() {
    if (this.sampler === undefined) return
    try { this.sampler.kill('SIGTERM') } catch { /* already gone */ }
    this.sampler = undefined
  }

  async finish(error, replayResult) {
    // Stay 'running' through the attach phase so clients never observe a
    // 'succeeded' job whose page image has not been written yet; a cancel()
    // racing this keeps its 'cancelled' state.
    this.error = error
    if (error !== null) {
      this.state = this.state === 'cancelled' ? 'cancelled' : 'failed'
      this.stopSampler()
      return
    }
    if (replayResult !== undefined) {
      try {
        const recordRel = path.relative(this.engine.root, replayResult.run_record).replaceAll(path.sep, '/')
        // readRunDetail expects <case>/<run> relative to the runs root.
        const runDir = recordRel.replace(new RegExp(`^${RUNS_ROOT}/`), '').split('/').slice(0, 2).join('/')
        const detail = await readRunDetail(this.engine, runDir)
        this.result = {
          caseId: this.caseId,
          runDir,
          outputRel: detail?.output?.relPath ?? null,
          outputBytes: detail?.output?.bytes ?? null,
          sha16: detail?.output?.sha16 ?? null,
          totalSeconds: detail?.totalSeconds ?? replayResult.seconds,
          stages: detail?.stages ?? [],
          peakMemoryBytes: detail?.peakMemoryBytes ?? null,
          replaySeconds: replayResult.seconds,
          returnCode: replayResult.return_code,
        }
      } catch (readError) {
        this.result = { caseId: this.caseId, runDir: null, replaySeconds: replayResult.seconds, error: String(readError) }
      }
      // A book illustration is copied into the book directory and indexed on
      // its page; failures surface as attachError without failing the run.
      if (this.request?.context?.kind === 'book' && this.result?.error === undefined) {
        try { this.result.image = await attachIllustration(this.engine, this.request.context, this) }
        catch (attachError) { this.attachError = String(attachError) }
      }
    }
    this.state = this.state === 'cancelled' ? 'cancelled' : 'succeeded'
    this.stopSampler()
  }

  cancel(reason = 'cancelled by user') {
    if (this.state !== 'running' && this.state !== 'starting') return
    for (const child of this.processes) {
      try { child.kill('SIGTERM') } catch { /* already gone */ }
    }
    setTimeout(() => {
      for (const child of this.processes) { try { child.kill('SIGKILL') } catch { /* already gone */ } }
    }, 3000).unref?.()
    void this.finish(`generation ${reason}`)
    this.state = 'cancelled'
  }
}

/* ------------------------------------------------------------------- tools */

function textBlock(value) {
  return [{ type: 'text', text: typeof value === 'string' ? value : JSON.stringify(value, null, 2) }]
}

function defineStudioTool(studio, { name, description, parameters }) {
  return {
    name,
    description,
    parameters,
    // Raw registration: the output schema is plain JSON Schema; every tool
    // returns a JSON object (defineTool's { type: 'json' } DSL does not apply).
    output: { schema: { type: 'object' }, render: (_args, value) => textBlock(value) },
    execute: async (args) => studio.invokeTool(name, args),
  }
}

/* ------------------------------------------------------------------- apply */

/** Cordis function-plugin name. */
export const name = 'onyx-studio'
/** The route carrier, the trust fence guarding every route, and the tool registry. */
export const inject = ['webServer', 'connection', 'tools']

export function apply(ctx, config = {}) {
  const logger = ctx.logger ?? console
  const state = {
    engine: null,
    engineReady: false,
    job: null,
    config: {
      engineRoot: config.engineRoot ?? '',
      planId: config.planId ?? 'native.mpsgraph.q4.stream.v1',
      sampleIntervalMs: safeNumber(config.sampleIntervalMs, 100),
      maxSamples: safeNumber(config.maxSamples, 4000),
    },
  }

  const studio = {
    async ensureEngine() {
      if (state.engine === null) state.engine = await resolveEngine(ctx, state.config)
      return state.engine
    },
    async overview() {
      const engine = await this.ensureEngine()
      if (engine.root === null) return { error: engine.problems.join('; ') }
      const [overview, target] = await Promise.all([readOverview(engine), readTarget(engine)])
      return { ...overview, problems: engine.problems, target }
    },
    async runs(limit) {
      const engine = await this.ensureEngine()
      if (engine.root === null) return { error: engine.problems.join('; ') }
      return listRuns(engine, Math.min(safeNumber(limit, 50), MAX_RUNS_LISTED))
    },
    async run(dir) {
      const engine = await this.ensureEngine()
      if (engine.root === null) return { error: 'engine unavailable' }
      const detail = await readRunDetail(engine, dir)
      return detail ?? { error: `run not found: ${dir}` }
    },
    async model() {
      const engine = await this.ensureEngine()
      if (engine.root === null) return { error: engine.problems.join('; ') }
      const target = await readTarget(engine)
      let manifest = null
      try { manifest = await readManifestSummary(engine) }
      catch (error) {
        return { manifest: null, packs: [], target, problems: [...engine.problems, `model manifest unavailable: ${error.message}`] }
      }
      let packs = []
      try { packs = await readPackReports(engine) }
      catch { /* reports are optional */ }
      return { manifest, packs, target, problems: engine.problems }
    },
    async operators() {
      const engine = await this.ensureEngine()
      if (engine.root === null) return { error: engine.problems.join('; ') }
      return readOperators(engine)
    },
    async startGeneration(request) {
      return this.launchGenerationJob({
        prompt: String(request.prompt ?? '').slice(0, 8000),
        width: Math.max(256, Math.round(safeNumber(request.width, 1024) / 16) * 16),
        height: Math.max(256, Math.round(safeNumber(request.height, 1024) / 16) * 16),
        seed: Math.round(safeNumber(request.seed, 42)),
        context: null,
      })
    },
    async labelGeneration({ runDir, layout }) {
      const job = state.job
      const outputRel = job?.result?.outputRel
      if (job?.state !== 'succeeded' || job.result?.runDir !== runDir || outputRel == null) {
        return { error: 'The selected image is no longer the current generation result' }
      }
      if (!Array.isArray(layout?.labels) || layout.labels.length === 0) {
        return { error: 'The layout needs a nonempty labels array' }
      }
      const engine = job.engine
      const runDirectory = path.join(engine.root, RUNS_ROOT, runDir)
      const realRunDirectory = await fs.realpath(runDirectory)
      const input = await fs.realpath(path.join(engine.root, outputRel))
      if (path.dirname(input) !== realRunDirectory || path.extname(input) !== '.png') {
        return { error: 'The generation output must be a PNG inside its run directory' }
      }
      const directory = await fs.mkdtemp(path.join(runDirectory, 'labels-'))
      const layoutFile = path.join(directory, 'layout.json')
      const output = path.join(directory, 'image.png')
      let saved = false
      try {
        await fs.writeFile(layoutFile, JSON.stringify(layout, null, 2))
        const result = await runCapture(ctx, engine.venvPython, [
          '-m', 'onyx', 'compose-labels', '--input', input, '--layout', layoutFile, '--output', output,
        ], { cwd: engine.root, timeoutMs: 15000 })
        if (result.code !== 0) return { error: result.stderr.trim().split('\n').at(-1) || 'Could not render labels' }
        saved = true
        return {
          imageRel: path.relative(engine.root, output).replaceAll(path.sep, '/'),
          layoutRel: path.relative(engine.root, layoutFile).replaceAll(path.sep, '/'),
        }
      } finally {
        if (!saved) await fs.rm(directory, { recursive: true, force: true })
      }
    },
    async launchGenerationJob(request) {
      const engine = await this.ensureEngine()
      if (engine.root === null) throw new Error(engine.problems.join('; '))
      if (engine.problems.length > 0) throw new Error(`engine not runnable: ${engine.problems.join('; ')}`)
      if (state.job !== null && (state.job.state === 'running' || state.job.state === 'starting')) {
        const elapsed = Math.round((nowMs() - state.job.startedAtMs) / 1000)
        throw new Error(`a generation is already running (${elapsed}s elapsed); wait for onyx_generate_status or cancel it first`)
      }
      const job = new GenerationJob(engine, state.config, request, logger)
      state.job = job
      try { await job.start() }
      catch (error) {
        state.job = null
        throw error
      }
      return job
    },
    async startIllustration(request) {
      const engine = await this.ensureEngine()
      if (engine.root === null) throw new Error(engine.problems.join('; '))
      const bookId = String(request.id ?? '')
      let pagesDoc
      try { pagesDoc = await readPagesDoc(engine, bookId) }
      catch { throw new Error(`book not found: ${bookId}`) }
      const page = Math.round(safeNumber(request.page, -1))
      if (page < 0 || page >= pagesDoc.pages.length) {
        throw new Error(`page out of range: 0..${pagesDoc.pages.length - 1}`)
      }
      const prompt = String(request.prompt ?? pagesDoc.pages[page]?.prompt ?? '').slice(0, 8000)
      if (prompt.trim() === '') throw new Error(`page ${page} has no illustration prompt`)
      const seed = request.seed === undefined
        ? Math.floor(Math.random() * 2 ** 31)
        : Math.round(safeNumber(request.seed, 42))
      const width = Math.max(256, Math.round(safeNumber(request.width, 1024) / 16) * 16)
      const height = Math.max(256, Math.round(safeNumber(request.height, 1024) / 16) * 16)
      return this.launchGenerationJob({
        prompt,
        width,
        height,
        seed,
        context: { kind: 'book', bookId, page },
      })
    },
    async books() {
      const engine = await this.ensureEngine()
      if (engine.root === null) return { error: engine.problems.join('; ') }
      return listBooks(engine)
    },
    async importBook(request) {
      const engine = await this.ensureEngine()
      if (engine.root === null) throw new Error(engine.problems.join('; '))
      return importBook(engine, request)
    },
    async book(id) {
      const engine = await this.ensureEngine()
      if (engine.root === null) return { error: engine.problems.join('; ') }
      try { return await readBookMeta(engine, String(id ?? '')) }
      catch { return null }
    },
    async patchBook(request) {
      const engine = await this.ensureEngine()
      if (engine.root === null) throw new Error(engine.problems.join('; '))
      return withBookLock(String(request.id ?? ''), () => patchBook(engine, request))
    },
    async deleteBook(request) {
      const engine = await this.ensureEngine()
      if (engine.root === null) throw new Error(engine.problems.join('; '))
      const job = state.job
      if (job !== null && (job.state === 'running' || job.state === 'starting')
        && job.request?.context?.bookId === request.id) {
        throw new Error('cannot delete a book while its illustration is generating')
      }
      return deleteBook(engine, String(request.id ?? ''))
    },
    async bookPages(request) {
      const engine = await this.ensureEngine()
      if (engine.root === null) return { error: engine.problems.join('; ') }
      return bookPages(engine, String(request.id ?? ''), request.from, request.to)
    },
    generationStatus() {
      return state.job === null
        ? { state: 'idle' }
        : { ...state.job.snapshot(), caseId: state.job.caseId }
    },
    cancelGeneration() {
      if (state.job === null) return { state: 'idle' }
      state.job.cancel()
      return this.generationStatus()
    },
    async awaitJob(job, timeoutMs) {
      const deadline = safeNumber(timeoutMs, 0)
      const limit = deadline > 0 ? nowMs() + deadline : Infinity
      for (;;) {
        const snapshot = job.snapshot()
        if (snapshot.state === 'succeeded') {
          return {
            ...snapshot.result,
            image: snapshot.result?.image ?? null,
            attachError: snapshot.attachError,
            purpose: snapshot.purpose,
            statusUrl: `${ROUTE_BASE}/api/generate/status`,
          }
        }
        if (snapshot.state === 'failed' || snapshot.state === 'cancelled') {
          const error = new Error(snapshot.error ?? `generation ${snapshot.state}`)
          error.code = snapshot.state
          throw error
        }
        if (nowMs() > limit) throw new Error(`generation timed out after ${Math.round(deadline / 1000)}s; poll ${ROUTE_BASE}/api/generate/status`)
        await new Promise((resolve) => setTimeout(resolve, 1000))
      }
    },
    async awaitGeneration(request) {
      const job = request.job !== undefined ? request.job : await this.startGeneration(request)
      return this.awaitJob(job, request.timeoutMs)
    },
    async invokeTool(name, args) {
      try {
        if (name === 'onyx_generate') {
          return await this.awaitGeneration({
            prompt: args.prompt, width: args.width, height: args.height, seed: args.seed, timeoutMs: safeNumber(args.timeoutMs, 0) || 0,
          })
        }
        if (name === 'onyx_book') {
          const action = args.action ?? 'list'
          if (action === 'list') return await this.books()
          if (action === 'status') return this.generationStatus()
          if (action === 'import') {
            let format = args.format
            let payloadB64 = args.payloadB64
            if (args.path != null) {
              const engine = await this.ensureEngine()
              if (engine.root === null) return { error: engine.problems.join('; ') }
              const raw = String(args.path)
              const abs = path.isAbsolute(raw) ? raw : path.resolve(engine.root, raw)
              const buffer = await fs.readFile(abs)
              payloadB64 = buffer.toString('base64')
              if (format == null) {
                const ext = path.extname(abs).toLowerCase().replace('.', '')
                format = BOOK_FORMATS.includes(ext) ? ext : null
              }
              if (format == null) return { error: `unsupported book format; use txt, md or epub` }
            }
            if (payloadB64 == null) return { error: 'import needs payloadB64 or path' }
            return this.importBook({ ...args, format, payloadB64 })
          }
          if (action === 'page') {
            const start = Math.max(0, Math.round(safeNumber(args.page, 0)))
            return this.bookPages({ id: args.id, from: start, to: start + Math.max(1, Math.round(safeNumber(args.count, 1))) })
          }
          if (action === 'illustrate') {
            const job = await this.startIllustration(args)
            return this.awaitJob(job, safeNumber(args.timeoutMs, 0) || 0)
          }
          return { error: `unknown action: ${action}` }
        }
        if (name === 'onyx_analyze') {
          const action = args.action ?? 'overview'
          let value
          if (action === 'overview') value = await this.overview()
          else if (action === 'runs') value = { runs: await this.runs(args.limit) }
          else if (action === 'run') value = await this.run(args.dir)
          else if (action === 'model') value = await this.model()
          else if (action === 'operators') value = await this.operators()
          else if (action === 'status') value = this.generationStatus()
          else value = { error: `unknown action: ${action}` }
          // An engine problem travels as the result, not a thrown error.
          if (Array.isArray(value?.runs) === false && value?.error !== undefined) return value
          return value
        }
        return { error: `unknown tool: ${name}` }
      } catch (error) {
        return { error: error instanceof Error ? error.message : String(error) }
      }
    },
  }

  /* ---- HTTP surface ---- */
  /**
   * Host/Origin fence: the composition's connection service rejects rebinding
   * and cross-site calls. The rejection carries diagnostic facts (host,
   * markers, cookie names — never values) so a misconfigured deployment is
   * debuggable from the response alone.
   */
  const rejectionOf = (req) => {
    const connection = Reflect.get(ctx, 'connection')
    if (typeof connection?.requestRejection !== 'function') return undefined
    const rejection = connection.requestRejection({ headers: req.headers })
    if (rejection === undefined) return undefined
    const cookieHeader = String(req.headers.cookie ?? '')
    return {
      code: rejection,
      hint: {
        host: req.headers.host ?? null,
        origin: req.headers.origin ?? null,
        secFetchSite: req.headers['sec-fetch-site'] ?? null,
        authCookies: cookieHeader.split(';')
          .map((part) => part.slice(0, part.indexOf('=')).trim())
          .filter((name) => name.startsWith('dsh-auth-')).length,
      },
    }
  }

  ctx.effect(() => ctx.webServer.register({ kind: 'prefix', path: `${ROUTE_BASE}/api`, handler: async (req, res) => {
    try {
      const rejection = rejectionOf(req)
      if (rejection !== undefined) return sendJson(res, rejection.code, { error: 'rejected', ...rejection.hint })
      const url = new URL(req.url, 'http://onyx-studio.local')
        const route = `${req.method} ${url.pathname}`
        if (route === 'GET /onyx-studio/api/overview') return sendJson(res, 200, await studio.overview())
        if (route === 'GET /onyx-studio/api/runs') {
          return sendJson(res, 200, await studio.runs(Number(url.searchParams.get('limit') ?? 50)))
        }
        if (route === 'GET /onyx-studio/api/run') {
          const dir = (url.searchParams.get('dir') ?? '').replace(/\\/g, '/')
          if (!/^[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+$/.test(dir)) return sendJson(res, 400, { error: 'dir must be <case>/<run>' })
          const detail = await studio.run(dir)
          return sendJson(res, detail?.error === undefined ? 200 : 404, detail)
        }
        if (route === 'GET /onyx-studio/api/model') return sendJson(res, 200, await studio.model())
        if (route === 'GET /onyx-studio/api/operators') return sendJson(res, 200, await studio.operators())
        if (route === 'POST /onyx-studio/api/generate') {
          const body = await readJsonBody(req)
          if (typeof body.prompt !== 'string' || body.prompt.trim() === '') return sendJson(res, 400, { error: 'prompt is required' })
          await studio.startGeneration(body)
          return sendJson(res, 202, studio.generationStatus())
        }
        if (route === 'GET /onyx-studio/api/generate/status') return sendJson(res, 200, studio.generationStatus())
        if (route === 'POST /onyx-studio/api/generate/cancel') return sendJson(res, 200, studio.cancelGeneration())
        if (route === 'POST /onyx-studio/api/generate/labels') {
          const body = await readJsonBody(req)
          const result = await studio.labelGeneration(body ?? {})
          return sendJson(res, result.error === undefined ? 201 : 422, result)
        }
        if (route === 'GET /onyx-studio/api/books') return sendJson(res, 200, await studio.books())
        if (route === 'POST /onyx-studio/api/books/import') {
          const body = await readJsonBody(req, MAX_BOOK_BODY_BYTES)
          if (!BOOK_FORMATS.includes(body.format)) return sendJson(res, 400, { error: 'format must be txt, md or epub' })
          const meta = await studio.importBook(body)
          return sendJson(res, 201, meta)
        }
        if (route === 'GET /onyx-studio/api/book') {
          const meta = await studio.book(url.searchParams.get('id'))
          return sendJson(res, meta == null ? 404 : 200, meta ?? { error: 'book not found' })
        }
        if (route === 'PATCH /onyx-studio/api/book') {
          const body = await readJsonBody(req)
          if (!body.id) return sendJson(res, 400, { error: 'id is required' })
          return sendJson(res, 200, await studio.patchBook(body))
        }
        if (route === 'DELETE /onyx-studio/api/book') {
          return sendJson(res, 200, await studio.deleteBook({ id: url.searchParams.get('id') }))
        }
        if (route === 'GET /onyx-studio/api/pages') {
          const query = url.searchParams
          return sendJson(res, 200, await studio.bookPages({
            id: query.get('id'),
            from: Number(query.get('from') ?? 0),
            to: Number(query.get('to') ?? 8),
          }))
        }
        if (route === 'POST /onyx-studio/api/book/illustrate') {
          const body = await readJsonBody(req)
          if (!body.id) return sendJson(res, 400, { error: 'id is required' })
          await studio.startIllustration(body)
          return sendJson(res, 202, studio.generationStatus())
        }
        sendJson(res, 404, { error: `unknown route: ${route}` })
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error)
        const code = message.includes('already running') ? 409
          : message.includes('request body too large') ? 413
          : message.includes('book not found') ? 404
          : 500
        sendJson(res, code, { error: message })
      }
    } }))

  ctx.effect(() => ctx.webServer.register({ kind: 'exact', path: `${ROUTE_BASE}/files`, handler: async (req, res) => {
      try {
        const rejection = rejectionOf(req)
        if (rejection !== undefined) return sendJson(res, rejection.code, { error: 'rejected', ...rejection.hint })
        const url = new URL(req.url, 'http://onyx-studio.local')
        // The path is relative to the engine root, as the client computes it
        // from run.json locators (for example artifacts/benchmarks/...).
        const rel = (url.searchParams.get('path') ?? '').replaceAll('\\', '/')
        if (rel.includes('..') || rel.startsWith('/')) return sendJson(res, 400, { error: 'invalid path' })
        const engine = await studio.ensureEngine()
        if (engine.root === null) return sendJson(res, 503, { error: 'engine root unavailable' })
        const root = path.resolve(engine.root)
        const artifacts = path.join(root, 'artifacts')
        const target = path.resolve(root, rel)
        // Only the engine's artifacts subtree is served, with no escape.
        const within = (base) => target === base || target.startsWith(base + path.sep)
        if (!within(root) || !within(artifacts)) return sendJson(res, 400, { error: 'path escapes artifacts' })
        const stat = await fs.stat(target).catch(() => null)
        if (stat === null || !stat.isFile()) return sendJson(res, 404, { error: 'not found' })
        const ext = path.extname(target).toLowerCase()
        const types = { '.png': 'image/png', '.json': 'application/json', '.log': 'text/plain; charset=utf-8', '.svg': 'image/svg+xml' }
        const type = types[ext]
        if (type === undefined) return sendJson(res, 400, { error: `unsupported file type: ${ext}` })
        const bytes = await fs.readFile(target)
        res.writeHead(200, { 'content-type': type, 'cache-control': 'no-store', 'content-length': bytes.length })
        res.end(bytes)
      } catch (error) {
        sendJson(res, 500, { error: error instanceof Error ? error.message : String(error) })
      }
    } }), 'onyx-studio: artifact file routes')

  /* ---- agent tools ---- */
  ctx.effect(() => ctx.tools.register(defineStudioTool(studio, {
      name: 'onyx_generate',
      description:
        'Generate one image with the local onyx-infer engine (MiniMax-H3-Image W4, Apple Silicon). Returns the output PNG path, reproducible run record, per-stage seconds and output hash. A 1024x1024 image takes about 40s; 4096x4096 about 5-6 minutes. Only one generation can run at a time.',
      parameters: {
        type: 'object',
        properties: {
          prompt: { type: 'string', description: 'Image prompt; English descriptive prompts work best.' },
          width: { type: 'integer', description: 'Output width, multiple of 16. Default 1024.' },
          height: { type: 'integer', description: 'Output height, multiple of 16. Default 1024.' },
          seed: { type: 'integer', description: 'Noise seed. Default 42.' },
          timeoutMs: { type: 'integer', description: 'Optional wait budget in milliseconds; omit to wait for completion.' },
        },
        required: ['prompt'],
        additionalProperties: false,
      },
    })), 'onyx-studio: onyx_generate tool')

  ctx.effect(() => ctx.tools.register(defineStudioTool(studio, {
      name: 'onyx_analyze',
      description:
        'Inspect the onyx-infer engine for research: overview (engine, hardware target, recent timings), runs (run-record index), run (one run: stage waterfall, memory, process log with per-step Metal allocation), model (manifest, quantization rules, dtype/parameter distribution, DiT graph), operators (Q4 GEMM benchmarks vs MLX), status (live generation).',
      parameters: {
        type: 'object',
        properties: {
          action: { type: 'string', enum: ['overview', 'runs', 'run', 'model', 'operators', 'status'] },
          dir: { type: 'string', description: 'For action=run: the <case>/<run_id> directory from runs.' },
          limit: { type: 'integer', description: 'For action=runs: how many runs to list (default 50).' },
        },
        required: ['action'],
        additionalProperties: false,
      },
    })), 'onyx-studio: onyx_analyze tool')

  ctx.effect(() => ctx.tools.register(defineStudioTool(studio, {
      name: 'onyx_book',
      description:
        'Local illustrated-book reader on the onyx-infer engine. list: imported books with progress and illustration counts. page: read a page range (text + per-page illustration prompt). import: turn a local txt/md/epub file (path, absolute or engine-root relative) into a paged book; style presets: ink, cinematic, storybook, minimal. illustrate: render one page illustration with the image engine — about 40s per 1024x1024 image and only one generation runs at a time, so batch requests must run sequentially. status: current generation job.',
      parameters: {
        type: 'object',
        properties: {
          action: { type: 'string', enum: ['list', 'import', 'page', 'illustrate', 'status'] },
          id: { type: 'string', description: 'Book id, from action=list.' },
          path: { type: 'string', description: 'For action=import: source file path (txt/md/epub).' },
          page: { type: 'integer', description: 'Zero-based page index (for page / illustrate).' },
          count: { type: 'integer', description: 'For action=page: how many pages to return (default 1).' },
          title: { type: 'string', description: 'For action=import: book title.' },
          author: { type: 'string', description: 'For action=import: author.' },
          style: { type: 'string', enum: ['ink', 'cinematic', 'storybook', 'minimal'], description: 'Illustration style preset; default ink.' },
          charsPerPage: { type: 'integer', description: 'For action=import: characters per page, 200..2000, default 800.' },
          prompt: { type: 'string', description: 'For action=illustrate: prompt override; defaults to the page prompt.' },
          seed: { type: 'integer', description: 'For action=illustrate: noise seed; default random.' },
          width: { type: 'integer', description: 'For action=illustrate: pixel width, multiple of 16, default 1024.' },
          height: { type: 'integer', description: 'For action=illustrate: pixel height, multiple of 16, default 1024.' },
          timeoutMs: { type: 'integer', description: 'For action=illustrate: wait budget; omit to wait for completion.' },
        },
        required: ['action'],
        additionalProperties: false,
      },
    })), 'onyx-studio: onyx_book tool')

  ctx.effect(() => () => {
    if (state.job !== null) state.job.cancel('plugin unloaded')
  }, 'onyx-studio: generation lifecycle')
}

/**
 * Host-half smoke test: drives the plugin's route handlers against a mock
 * Cordis context and the real engine artifacts (read-only endpoints only).
 */
import assert from 'node:assert/strict'
import { pathToFileURL } from 'node:url'
import path from 'node:path'

const engineRoot = path.resolve(process.env.ONYX_ENGINE_ROOT ?? path.join(import.meta.dirname, '../..'))
const plugin = await import(pathToFileURL(path.join(import.meta.dirname, 'index.js')).href)

const routes = new Map()
const tools = new Map()
const effects = []
const mockCtx = {
  logger: { info: () => {}, warn: () => {}, error: () => {} },
  effect(fn) { const dispose = fn?.() ; effects.push(dispose); return () => dispose?.() },
  inject(names, callback) { return callback(mockCtx) },
  webServer: {
    register(route) { routes.set(`${route.kind}:${route.path}`, route); return () => routes.delete(`${route.kind}:${route.path}`) },
  },
  tools: {
    register(definition) { tools.set(definition.name, definition); return () => tools.delete(definition.name) },
  },
}

plugin.apply(mockCtx, { engineRoot })

let outputPath

function mockRes() {
  const state = { code: 0, body: '' }
  return {
    state,
    writeHead(code, headers) { state.code = code; state.headers = headers },
    write(chunk) { state.body += String(chunk); return true },
    end(body) { if (body !== undefined) state.body += String(body) },
    on() { return this },
    once() { return this },
    emit() { return true },
  }
}

async function call(method, url, body) {
  const route = routes.get('prefix:/onyx-studio/api')
  assert.ok(route != null, 'api route registered')
  const req = {
    method, url,
    on(event, listener) { if (event === 'end') listener() ; if (event === 'data') { /* no body */ } },
    destroy() {},
  }
  if (body !== undefined) {
    req.on = (event, listener) => {
      if (event === 'data') listener(Buffer.from(JSON.stringify(body)))
      if (event === 'end') listener()
    }
  }
  const res = mockRes()
  await route.handler(req, res)
  return { code: res.state.code, json: res.state.body === '' ? null : JSON.parse(res.state.body) }
}

/* overview */
{
  const { code, json } = await call('GET', '/onyx-studio/api/overview')
  assert.equal(code, 200, `overview code ${code}: ${JSON.stringify(json)}`)
  assert.equal(json.engineRoot, engineRoot)
  assert.ok(json.runCount > 0, "run generation before testing historical run routes")
  assert.ok(json.medianTotalSeconds > 0)
  assert.ok(json.target?.chip?.includes('M5'))
  console.log(`overview: ${json.runCount} runs, median ${json.medianTotalSeconds.toFixed(1)}s, ${json.target.chip}`)
}

/* runs + one detail */
{
  const { code, json } = await call('GET', '/onyx-studio/api/runs?limit=30')
  assert.equal(code, 200)
  assert.ok(json.length > 0)
  const first = json[0]
  assert.ok(/^[^/]+\/[^/]+$/.test(first.dir))
  const detail = await call('GET', `/onyx-studio/api/run?dir=${encodeURIComponent(first.dir)}`)
  assert.equal(detail.code, 200, JSON.stringify(detail.json))
  assert.ok(detail.json.stages.length === 3, `stages: ${JSON.stringify(detail.json.stages)}`)
  assert.ok(detail.json.process?.ditSteps?.length > 0, 'dit steps parsed from process.log')
  assert.ok(detail.json.output.exists, 'output png exists')
  outputPath = detail.json.output.relPath
  console.log(`runs: ${json.length} listed; detail ${first.dir}: total ${detail.json.totalSeconds.toFixed(1)}s, dit steps ${detail.json.process.ditSteps.length}, sha ${detail.json.output.sha16}`)
}

/* model — detailed assertions only apply when the W4 package is present */
{
  const { code, json } = await call('GET', '/onyx-studio/api/model')
  assert.equal(code, 200, JSON.stringify(json))
  const dit = (json.manifest?.graphs ?? []).reduce((best, g) => (g.nodes > (best?.nodes ?? 0) ? g : best), null)
  if (dit == null) {
    assert.ok((json.problems ?? []).some((p) => p.includes('manifest')), `model without manifest must explain why: ${JSON.stringify(json.problems)}`)
    console.warn(`model: W4 package absent (${json.problems?.join('; ')}); skipping manifest assertions`)
  } else {
    assert.equal(dit.nodes, 53, `dit graph nodes: ${dit.nodes}`)
    assert.equal(dit.blockNodes, 50)
    const transformer = json.packs.find((p) => p.component === 'transformer')
    assert.ok(transformer.tensors > 400, `transformer tensors: ${transformer.tensors}`)
    assert.ok(transformer.totalElements > 1e9, `transformer params: ${(transformer.totalElements / 1e9).toFixed(2)}B`)
    const estimated = transformer.byDtype.reduce((sum, d) => sum + d.bytes, 0)
    assert.ok(estimated > 30 * 1024 ** 3 && estimated < 40 * 1024 ** 3, `transformer estimated bytes: ${(estimated / 2 ** 30).toFixed(1)} GiB`)
    assert.ok(json.manifest.quantization?.tensorRules?.length > 0 || json.manifest.quantization?.tensor_rules?.length > 0)
    console.log(`model: dit ${dit.nodes} nodes/50 blocks; transformer ${transformer.tensors} tensors, ${(transformer.totalElements / 1e9).toFixed(2)}B params; dtypes ${transformer.byDtype.map((d) => d.dtype).join(',')}`)
  }
}

/* operators */
{
  const { code, json } = await call('GET', '/onyx-studio/api/operators')
  assert.equal(code, 200)
  assert.ok(Array.isArray(json.comparisons) && Array.isArray(json.raw))
  if (json.comparisons.length > 0) assert.ok(json.comparisons[0].onyx.wallMedianMs > 0)
  if (json.raw.length > 0) assert.ok(json.raw.some((row) => row.effectiveTflops > 0))
  console.log(`operators: ${json.comparisons.length} comparisons, ${json.raw.length} raw`)
}

/* generate status is idle; start rejected without prompt */
{
  const status = await call('GET', '/onyx-studio/api/generate/status')
  assert.equal(status.code, 200)
  assert.equal(status.json.state, 'idle')
  const bad = await call('POST', '/onyx-studio/api/generate', {})
  assert.equal(bad.code, 400)
  console.log('generate: idle status ok; empty prompt rejected')
}

/* files route guards */
{
  const route = routes.get('exact:/onyx-studio/files')
  assert.ok(route != null, 'files route registered')
  const attempt = async (url) => {
    const res = mockRes()
    await route.handler({ method: 'GET', url, headers: {}, on: () => {} }, res)
    if (res.state.code >= 500) console.error('files error body:', res.state.body.slice(0, 200))
    return res.state.code
  }
  assert.equal(await attempt('/onyx-studio/files?path=../etc/passwd'), 400)
  assert.equal(await attempt('/onyx-studio/files?path=/etc/passwd'), 400)
  assert.equal(await attempt('/onyx-studio/files?path=benchmarks/../../etc/hosts'), 400)
  // The client sends engine-root-relative paths (from run.json locators).
  const png = outputPath
  assert.equal(await attempt(`/onyx-studio/files?path=${encodeURIComponent(png)}`), 200, 'client-format png URL must load')
  assert.equal(await attempt(`/onyx-studio/files?path=${png}`), 200, 'unencoded png URL must load')
  assert.equal(await attempt('/onyx-studio/files?path=benchmarks/onyx-native-h3/latest.json'), 400, 'paths outside the artifacts subtree are rejected')
  console.log('files: traversal rejected; client-format and plain URLs served')
}

/* books: import → list → read pages → patch progress → delete */
{
  const txt = [
    '第一回　测试开篇',
    '这是第一段测试文本，讲的是一位旅人翻山越岭，去寻找传说中的青瓷茶壶。',
    '山中有雾，泉水在石上流过，月亮升起来，照见满山的枫叶。',
    '第二回　茶壶现身',
    '旅人在山顶的破庙里找到一把老茶壶，壶身上刻着云纹，倒出清亮的泉水。',
    '庙外风起，吹动檐角的铜铃，远处传来隐隐的钟声。',
  ].join('\n')
  const payloadB64 = Buffer.from(txt, 'utf8').toString('base64')
  const imp = await call('POST', '/onyx-studio/api/books/import', { format: 'txt', title: '烟雨测试录', author: '测试', style: 'ink', charsPerPage: 120, payloadB64 })
  assert.equal(imp.code, 201, JSON.stringify(imp.json))
  const id = imp.json.id
  assert.ok(imp.json.pageCount >= 2, `pages: ${imp.json.pageCount}`)
  assert.equal(imp.json.classical, true)
  const list = await call('GET', '/onyx-studio/api/books')
  assert.equal(list.code, 200)
  assert.ok(list.json.some((book) => book.id === id), 'imported book listed')
  const pages = await call('GET', `/onyx-studio/api/pages?id=${encodeURIComponent(id)}&from=0&to=2`)
  assert.equal(pages.code, 200, JSON.stringify(pages.json))
  assert.ok(pages.json.pages.length >= 2, `page slice: ${pages.json.pages.length}`)
  assert.ok(pages.json.pages[0].prompt.includes('ink-wash'), 'page prompt carries the style preset')
  assert.equal(pages.json.pages[0].image, null)
  const patch = await call('PATCH', '/onyx-studio/api/book', { id, progress: { lastPage: 99 }, settings: { theme: 'night', fontSize: 21 } })
  assert.equal(patch.code, 200, JSON.stringify(patch.json))
  assert.equal(patch.json.progress.lastPage, patch.json.pageCount - 1, 'progress clamped to last page')
  assert.equal(patch.json.settings.theme, 'night')
  const book = await call('GET', `/onyx-studio/api/book?id=${encodeURIComponent(id)}`)
  assert.equal(book.code, 200)
  const missing = await call('GET', '/onyx-studio/api/book?id=b-does-not-exist-00')
  assert.equal(missing.code, 404, `missing book code: ${missing.code}`)
  // book.json is servable through the artifacts-scoped files route
  {
    const filesRoute = routes.get('exact:/onyx-studio/files')
    const res0 = mockRes()
    await filesRoute.handler(
      { method: 'GET', url: `/onyx-studio/files?path=${encodeURIComponent(`artifacts/studio/books/${id}/book.json`)}`, headers: {}, on: () => {} },
      res0)
    assert.equal(res0.state.code, 200, `book files served: ${res0.state.body.slice(0, 120)}`)
  }
  // illustrate either refuses cleanly (no model package) or starts; started jobs are cancelled
  const ill = await call('POST', '/onyx-studio/api/book/illustrate', { id, page: 0 })
  assert.ok([202, 409, 500].includes(ill.code), `illustrate code ${ill.code}`)
  if (ill.code === 202) await call('POST', '/onyx-studio/api/generate/cancel')
  else assert.ok(/engine not runnable|already running/.test(ill.json.error ?? ''), `illustrate guard: ${JSON.stringify(ill.json)}`)
  const del = await call('DELETE', `/onyx-studio/api/book?id=${encodeURIComponent(id)}`)
  assert.equal(del.code, 200)
  const after = await call('GET', `/onyx-studio/api/book?id=${encodeURIComponent(id)}`)
  assert.equal(after.code, 404, 'deleted book is gone')
  console.log(`books: ${imp.json.chapterCount} chapters / ${imp.json.pageCount} pages imported; patch, files route, delete ok; illustrate guard ${ill.code}`)
}

/* tools registered */
assert.ok(tools.has('onyx_generate') && tools.has('onyx_analyze') && tools.has('onyx_book'), `tools: ${[...tools.keys()].join(',')}`)
{
  const result = await tools.get('onyx_analyze').execute({ action: 'overview' }, { signal: new AbortController().signal })
  assert.ok(result.runCount > 0)
  const books = await tools.get('onyx_book').execute({ action: 'list' }, { signal: new AbortController().signal })
  assert.ok(Array.isArray(books), 'onyx_book list returns an array')
  console.log(`tool onyx_analyze(overview): runCount=${result.runCount}; onyx_book(list): ${books.length} books`)
}

console.log('SMOKE-OK')

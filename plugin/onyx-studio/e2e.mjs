/**
 * End-to-end generation test: starts one real 1024x1024 generation through
 * the route handler, polls live status, and verifies the recorded run.
 */
import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { pathToFileURL } from 'node:url'
import path from 'node:path'
import fs from 'node:fs'

const engineRoot = path.resolve(process.env.ONYX_ENGINE_ROOT ?? path.join(import.meta.dirname, '../..'))
const plugin = await import(pathToFileURL(path.join(import.meta.dirname, 'index.js')).href)

const routes = new Map()
const ctx = {
  logger: { info: () => {}, warn: () => {}, error: () => {} },
  effect: (fn) => fn?.() ?? (() => {}),
  inject: (names, callback) => callback(ctx),
  webServer: { register: (route) => { routes.set(`${route.kind}:${route.path}`, route); return () => {} } },
  tools: { register: () => () => {} },
}
plugin.apply(ctx, { engineRoot })

function mockRes() {
  const state = { code: 0, body: '' }
  return {
    state,
    writeHead(code) { state.code = code },
    write(chunk) { state.body += String(chunk); return true },
    end(body) { if (body !== undefined) state.body += String(body) },
    on() { return this },
    once() { return this },
    emit() { return true },
  }
}

async function call(method, url, body) {
  const route = routes.get('prefix:/onyx-studio/api')
  const req = {
    method, url,
    on(event, listener) {
      if (event === 'data' && body !== undefined) listener(Buffer.from(JSON.stringify(body)))
      if (event === 'end') listener()
    },
    destroy() {},
  }
  const res = mockRes()
  await route.handler(req, res)
  return { code: res.state.code, json: res.state.body === '' ? null : JSON.parse(res.state.body) }
}

const prompt = 'A close-up product photograph of one bright yellow enamel teapot with a black arched handle, standing on a blue wooden table. Soft morning window light, realistic enamel, no text.'
const start = await call('POST', '/onyx-studio/api/generate', { prompt, width: 1024, height: 1024, seed: 7 })
assert.equal(start.code, 202, JSON.stringify(start.json))
console.log('started:', start.json.state)

const began = Date.now()
let lastStage = null
let sawSamples = 0
for (;;) {
  await new Promise((resolve) => setTimeout(resolve, 1500))
  const status = await call('GET', '/onyx-studio/api/generate/status')
  assert.equal(status.code, 200)
  const snapshot = status.json
  if (snapshot.stage !== lastStage) { console.log(`t+${((Date.now() - began) / 1000).toFixed(1)}s stage=${snapshot.stage}`); lastStage = snapshot.stage }
  sawSamples = Math.max(sawSamples, snapshot.samples.length)
  if (snapshot.state !== 'running' && snapshot.state !== 'starting') {
    console.log(`final: state=${snapshot.state} elapsed=${(snapshot.elapsedMs / 1000).toFixed(1)}s samples=${sawSamples}`)
    if (snapshot.state !== 'succeeded') {
      console.error('failure:', snapshot.error)
      console.error(snapshot.logTail.join('\n'))
      process.exit(1)
    }
    assert.ok(snapshot.result.runDir.includes('studio-'), snapshot.result.runDir)
    assert.ok(snapshot.result.outputRel.endsWith('.png'))
    assert.ok(snapshot.result.totalSeconds > 0)
    assert.ok(snapshot.result.stages.length === 3)
    const detail = await call('GET', `/onyx-studio/api/run?dir=${encodeURIComponent(snapshot.result.runDir)}`)
    assert.equal(detail.code, 200)
    assert.ok(detail.json.process.ditSteps.length === 6, `dit steps ${detail.json.process.ditSteps.length}`)
    const png = path.join(engineRoot, snapshot.result.outputRel)
    const stat = fs.statSync(png)
    assert.ok(stat.size > 100_000, `png size ${stat.size}`)
    const imageHash = (file) => createHash('sha256').update(fs.readFileSync(file)).digest('hex')
    const originalHash = imageHash(png)
    const { runDir } = snapshot.result
    const layout = { labels: [{ text: '搪瓷茶壶', box: [0.2, 0.06, 0.6, 0.09], background: '#293e52' }] }
    const wrongRun = await call('POST', '/onyx-studio/api/generate/labels', { runDir: 'other-run', layout })
    assert.equal(wrongRun.code, 422)
    const invalid = await call('POST', '/onyx-studio/api/generate/labels', { runDir, layout: { labels: [] } })
    assert.equal(invalid.code, 422)
    const labeled = await call('POST', '/onyx-studio/api/generate/labels', { runDir, layout })
    assert.equal(labeled.code, 201, JSON.stringify(labeled.json))
    const repeated = await call('POST', '/onyx-studio/api/generate/labels', { runDir, layout })
    assert.equal(repeated.code, 201, JSON.stringify(repeated.json))
    const labelHash = imageHash(path.join(engineRoot, labeled.json.imageRel))
    assert.equal(labelHash, imageHash(path.join(engineRoot, repeated.json.imageRel)))
    assert.notEqual(labelHash, originalHash)
    assert.equal(imageHash(png), originalHash)
    assert.deepEqual(JSON.parse(fs.readFileSync(path.join(engineRoot, labeled.json.layoutRel))), layout)
    const fileResponse = mockRes()
    await routes.get('exact:/onyx-studio/files').handler({
      method: 'GET',
      url: `/onyx-studio/files?path=${encodeURIComponent(labeled.json.imageRel)}`,
    }, fileResponse)
    assert.equal(fileResponse.state.code, 200)
    console.log(`run: ${snapshot.result.runDir}`)
    console.log(`png: ${png} (${(stat.size / 1024).toFixed(0)} KiB) sha16=${snapshot.result.sha16}`)
    console.log(`stages: ${snapshot.result.stages.map((s) => `${s.id}=${s.seconds.toFixed(1)}s`).join(' ')}`)
    console.log('E2E-OK')
    process.exit(0)
  }
}

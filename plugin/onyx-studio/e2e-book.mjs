/**
 * Book illustration end-to-end: one real generation through the plugin's
 * onyx_book tool, verifying the PNG lands in the book and the page carries it.
 */
import assert from 'node:assert/strict'
import { pathToFileURL } from 'node:url'
import path from 'node:path'
import fs from 'node:fs'
import os from 'node:os'

const engineRoot = path.resolve(process.env.ONYX_ENGINE_ROOT ?? path.join(import.meta.dirname, '../..'))
const plugin = await import(pathToFileURL(path.join(import.meta.dirname, 'index.js')).href)

const toolDefs = new Map()
const ctx = {
  logger: { info: () => {}, warn: () => {}, error: () => {} },
  effect: (fn) => { fn?.(); return () => {} },
  inject: () => () => {},
  webServer: { register: () => () => {} },
  tools: { register: (def) => { toolDefs.set(def.name, def); return () => {} } },
}
plugin.apply(ctx, { engineRoot })
const book = toolDefs.get('onyx_book')
assert.ok(book != null, 'onyx_book registered')

const fixtureDir = fs.mkdtempSync(path.join(os.tmpdir(), 'onyx-book-'))
const source = path.join(fixtureDir, 'fixture.txt')
fs.writeFileSync(source, '第一回 茶壶\n\n旅人沿着山路走到一间小屋。桌上摆着一只黄色的搪瓷茶壶，清晨的阳光从窗户照进来。')
let target
try {
  target = await book.execute({ action: 'import', path: source, title: 'Onyx Studio test book' },
    { signal: new AbortController().signal })
} finally {
  fs.rmSync(fixtureDir, { recursive: true, force: true })
}
assert.ok(target.id != null, JSON.stringify(target))

const result = await book.execute(
  { action: 'illustrate', id: target.id, page: 0, seed: 42, timeoutMs: 10 * 60 * 1000 },
  { signal: new AbortController().signal },
)
assert.ok(result.image != null, `illustrate result image: ${JSON.stringify(result).slice(0, 300)}`)
console.log(`attached: ${result.image.file} ${result.image.width}x${result.image.height} sha ${result.image.sha16} total ${result.totalSeconds?.toFixed?.(1)}s`)

const abs = path.join(engineRoot, result.image.file)
assert.ok(fs.existsSync(abs), 'png exists on disk')
const pages = await book.execute({ action: 'page', id: target.id, page: 0 }, { signal: new AbortController().signal })
assert.ok(pages.pages[0].image?.file != null, 'page now carries the image')
console.log('BOOK-ILLUSTRATE-E2E-OK:', pages.pages[0].image.file)

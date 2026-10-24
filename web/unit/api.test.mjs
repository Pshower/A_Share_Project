import { test } from 'node:test'
import assert from 'node:assert/strict'
import { api } from '../src/api.ts'

test('a failed session request is retried instead of cached', async () => {
  let sessions = 0
  globalThis.fetch = async (input, init) => {
    const url = String(input)
    if (url === '/api/session') {
      sessions += 1
      if (sessions === 1) throw new Error('offline')
      return { status: 200, ok: true, json: async () => ({ token: 'token' }) }
    }
    assert.equal(url, '/api/probe')
    assert.equal(new Headers(init?.headers).get('X-CSRF-Token'), 'token')
    return { status: 200, ok: true, json: async () => ({ ok: true }) }
  }

  await assert.rejects(api('/probe'), /offline/)
  assert.deepEqual(await api('/probe'), { ok: true })
  assert.equal(sessions, 2)
})

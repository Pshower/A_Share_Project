import { test } from 'node:test'
import assert from 'node:assert/strict'
import { freshQuote, QUOTE_TTL_MS } from '../src/quoteFreshness.ts'

test('only a recent complete quote snapshot is selected automatically', () => {
  const now = Date.parse('2026-10-31T12:00:00Z')
  const at = (seconds) => new Date(now - seconds * 1000).toISOString()
  const quotes = [
    { id: 'old-ready', status: 'ready', created_at: at(QUOTE_TTL_MS / 1000 + 1) },
    { id: 'old-partial', status: 'partial', created_at: at(30) },
    { id: 'fresh-ready', status: 'ready', created_at: at(30) },
  ]
  assert.equal(freshQuote(quotes, now)?.id, 'fresh-ready')
  assert.equal(freshQuote(quotes.slice(0, 2), now), undefined)
  assert.equal(
    freshQuote([{ id: 'bad', status: 'ready', created_at: 'not-a-date' }], now),
    undefined
  )
})

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { parseCodes } from '../src/codes.ts'

test('stock codes accept every UI separator and preserve order', () => {
  assert.deepEqual(parseCodes(' 000001, 600030；000002;000001\n'), [
    '000001',
    '600030',
    '000002',
    '000001',
  ])
  assert.deepEqual(parseCodes(' ,；\n '), [])
})

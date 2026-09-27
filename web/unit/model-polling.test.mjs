import { test } from 'node:test'
import assert from 'node:assert/strict'
import { pollModels } from '../src/modelPolling.ts'

test('a failed model request never discards another completed result', async () => {
  const runs = [{ job_id: 'bad', status: 'running' }, { job_id: 'good', status: 'queued' }]
  const results = await pollModels(runs, async (path) => {
    if (path === '/jobs/bad') throw new Error('offline')
    return path === '/jobs/good' ? { status: 'succeeded', result: { snapshot_id: 'p' } } : { id: 'p', rows: [] }
  })
  assert.equal(results[0].read_error, 'offline')
  assert.equal(results[0].status, 'running')
  assert.equal(results[1].plan.id, 'p')
  const retry = await pollModels(results, async () => ({ status: 'cancelled' }))
  assert.equal(retry.length, 1)
  assert.equal(retry[0].status, 'cancelled')
  assert.equal(retry[0].read_error, null)
})

test('a missing plan is retryable without rerunning the successful job', async () => {
  const runs = [{ job_id: 'one', status: 'queued' }, { job_id: 'two', status: 'running' }]
  const results = await pollModels(runs, async (path) => {
    if (path.includes('/plans/')) throw new Error('missing artifact')
    return path === '/jobs/one' ? { status: 'succeeded', result: { snapshot_id: 'missing' } } : { status: 'failed', error: 'model error' }
  })
  assert.equal(results[0].read_error, 'missing artifact')
  assert.equal(results[1].status, 'failed')
  assert.deepEqual(results[1].errors, ['model error'])
})

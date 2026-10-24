import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
  actionMode,
  dailyPlanActionModes,
  predictionActionModes,
  supportsAction,
  webActionModes,
} from '../src/modelActions.ts'

test('web action modes distinguish evaluation, prediction and daily plans', () => {
  const target = { valid: true, trained: true, action_mode: 'target' }
  const delta = { valid: true, trained: true, action_mode: 'stock_delta' }
  const trade = { valid: true, trained: true, metadata: { contract: { action_mode: 'stock_trade' } } }
  const priority = { valid: true, trained: true, action_mode: 'stock_priority' }
  assert.equal(actionMode(priority), 'stock_priority')
  assert.equal(supportsAction(target, webActionModes), true)
  assert.equal(supportsAction(trade, webActionModes), true)
  assert.equal(supportsAction(delta, predictionActionModes), true)
  assert.equal(supportsAction(trade, predictionActionModes), false)
  assert.equal(supportsAction(target, dailyPlanActionModes), true)
  assert.equal(supportsAction(delta, dailyPlanActionModes), false)
  assert.equal(supportsAction(priority, webActionModes), false)
})

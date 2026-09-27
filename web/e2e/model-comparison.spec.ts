import { test, expect, type Page } from '@playwright/test'

// All API traffic is fulfilled here. These are synthetic UI fixtures, never market jobs.
async function fixture(page: Page, running = false) {
  const state = { submitted: [] as any[], stops: [] as string[], readFailure: !running }
  const models = ['m1', 'm2', 'm3'].map((id, i) => ({ id, run: `模拟${i + 1}`, name: 'fixture', trained: true, valid: true,
    pool_size: 1, metadata: { contract: { data_source: i === 2 ? 'eastmoney' : 'sina' } } }))
  const history = { id: 'h_fixture', status: 'ready', request: { provider: 'sina', codes: ['000001'], bases: ['hfq'] },
    entries: [{ code: '000001', basis: 'hfq', status: 'ready' }] }
  await page.route('**/api/**', async (route) => {
    const request = route.request(), path = new URL(request.url()).pathname.replace('/api', '')
    const send = (json: unknown, status = 200) => route.fulfill({ status, json })
    if (path === '/session') return send({ token: 'fixture' })
    if (path === '/system') return send({ environment: 'SYNTHETIC_TEST', packages: {}, device: 'cpu' })
    if (path === '/defaults') return send({ light_codes: [], ppo: {} })
    if (path === '/models') return send(models)
    if (['/datasets', '/reports', '/jobs', '/market/monitors', '/market/snapshots/quotes', '/market/snapshots/plans'].includes(path)) return send([])
    if (path === '/market/status') return send({ today: '2026-06-20', market_ui_version: 'model-comparison-v1', min_poll_seconds: 60 })
    if (path === '/market/snapshots/history') return send([history])
    if (path === '/market/history/h_fixture/bars') return send({ rows: Array.from({ length: 20 }, (_, i) => ({
      date: `2026-06-${String(i + 1).padStart(2, '0')}`, open: 10 + i / 10, close: 10.2 + i / 10,
      high: 10.4 + i / 10, low: 9.8 + i / 10, volume: 100000 + i * 2000, amount: 1000000 })) })
    if (path.startsWith('/market/model-requirements/')) {
      const id = path.split('/').at(-1)!
      return send({ model_id: id, source: id === 'm3' ? 'eastmoney' : 'sina', codes: ['000001'], train_end: '2026-03-31',
        history_origin: '2020-01-01', price_basis: 'hfq_research' })
    }
    if (path.startsWith('/market/model-performance/')) return send([])
    if (path.startsWith('/market/model-comparison')) {
      const body = request.postDataJSON()
      const checks = body.model_ids.map((id: string) => ({ model_id: id, ready: id !== 'm3', errors: id === 'm3' ? ['模型训练来源与日线来源不一致'] : [] }))
      if (path.endsWith('/preflight')) return send({ models: checks })
      state.submitted.push(body)
      return send({ models: checks, results: checks.map((c: any) => c.ready
        ? { model_id: c.model_id, job_id: 'j_' + c.model_id, status: 'queued', historical_label: '开发验证期回溯' }
        : { model_id: c.model_id, status: 'blocked', errors: c.errors }) }, 202)
    }
    if (path.startsWith('/jobs/j_')) {
      const id = path.split('/')[2]
      if (path.endsWith('/stop')) { state.stops.push(id); return send({ status: 'stop_requested' }) }
      if (state.stops.includes(id)) return send({ status: 'cancelled' })
      if (running) return send({ status: 'running' })
      if (id === 'j_m2') return state.readFailure ? send({ error: '模拟单模型读取失败' }, 503) : send({ status: 'failed', error: '模拟模型计算失败' })
      return send({ status: 'succeeded', result: { snapshot_id: 'p_fixture' } })
    }
    if (path === '/market/snapshots/plans/p_fixture') return send({ id: 'p_fixture', feature_as_of: '2026-06-20', target_cash: .75,
      rows: [{ code: '000001', current_research_weight: 0, target_weight: .25, research_intent: 'buy', valid: true }] })
    throw new Error('Unexpected API request: ' + request.method() + ' ' + path)
  })
  await page.goto('/#market')
  await expect(page.getByTestId('market-chart').locator('canvas')).toBeVisible()
  await page.getByLabel('对照模型1', { exact: true }).selectOption('m1')
  await page.getByRole('button', { name: '添加对照模型', exact: true }).click()
  await page.getByLabel('对照模型2', { exact: true }).selectOption('m2')
  return state
}

async function approve(page: Page) {
  await expect(page.getByRole('button', { name: '生成多模型研究意图', exact: true })).toBeDisabled()
  await page.getByLabel('模型对照账户', { exact: true }).selectOption('cash')
  await page.getByLabel('确认假设账户与HFQ研究意图，非真实股数建议', { exact: true }).check()
}

test('model comparison isolates failures, blocks mixed sources and renders successful signals', async ({ page }) => {
  const state = await fixture(page)
  await page.getByRole('button', { name: '添加对照模型', exact: true }).click()
  await page.getByLabel('对照模型3', { exact: true }).selectOption('m3')
  await expect(page.getByRole('button', { name: '添加对照模型', exact: true })).toBeDisabled()
  expect(state.submitted).toHaveLength(0)
  await approve(page)
  await page.getByRole('button', { name: '检查输入', exact: true }).click()
  await expect(page.getByRole('alert')).toContainText('模型训练来源与日线来源不一致')
  expect(state.submitted).toHaveLength(0)
  const before = await page.getByTestId('market-chart').screenshot()
  await page.getByRole('button', { name: '生成多模型研究意图', exact: true }).click()
  const success = page.getByRole('row').filter({ hasText: '模拟1' })
  await expect(success).toContainText('已完成')
  await expect(success).toContainText('0.00% → 25.00%')
  await expect(page.getByRole('row').filter({ hasText: '模拟2' })).toContainText('状态读取失败')
  await expect(page.getByRole('row').filter({ hasText: '模拟3' })).toContainText('已阻断')
  expect(state.submitted[0].model_ids).toEqual(['m1', 'm2', 'm3'])
  expect(state.submitted[0].account_mode).toBe('cash')
  const after = await page.getByTestId('market-chart').screenshot()
  expect(after.equals(before)).toBe(false)
  state.readFailure = false
  await expect(page.getByRole('row').filter({ hasText: '模拟2' })).toContainText('模拟模型计算失败')
  await page.screenshot({ path: 'test-results/model-comparison-desktop.png', fullPage: true })
  await page.setViewportSize({ width: 390, height: 844 })
  await expect.poll(() => page.locator('.sidebar').evaluate((e) => e.getBoundingClientRect().right)).toBeLessThanOrEqual(1)
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBeTruthy()
  await page.screenshot({ path: 'test-results/model-comparison-mobile.png', fullPage: true })
})

test('comparison stops each outstanding model without another submission', async ({ page }) => {
  const state = await fixture(page, true)
  await approve(page)
  await page.getByRole('button', { name: '生成多模型研究意图', exact: true }).click()
  await page.getByRole('button', { name: '停止本组预测', exact: true }).click()
  await expect(page.getByRole('row').filter({ hasText: '模拟1' })).toContainText('已取消')
  await expect(page.getByRole('row').filter({ hasText: '模拟2' })).toContainText('已取消')
  expect(state.stops.sort()).toEqual(['j_m1', 'j_m2'])
  expect(state.submitted).toHaveLength(1)
})

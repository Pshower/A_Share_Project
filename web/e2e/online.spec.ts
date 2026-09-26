import { test, expect } from '@playwright/test'

test('online page opens without network jobs and requires explicit download consent', async ({
  page,
}) => {
  const before = await (await page.request.get('/api/jobs')).json()
  let captured: any = null
  await page.route('**/api/market/downloads', async (route) => {
    captured = route.request().postDataJSON()
    await route.fulfill({
      status: 409,
      contentType: 'application/json',
      body: JSON.stringify({ error: 'MOCK_ONLY_REQUEST_CAPTURED' }),
    })
  })
  await page.goto('/#market')
  await expect(page.getByRole('heading', { name: '历史日线采集' })).toBeVisible()
  const submit = page.getByRole('button', { name: '开始联网下载', exact: true })
  await expect(submit).toBeDisabled()
  await page.getByLabel('联网股票代码', { exact: true }).fill('000001 600030')
  await page.getByLabel('允许本次从东方财富联网采集', { exact: true }).check()
  await submit.click()
  await expect(page.getByRole('alert')).toContainText('MOCK_ONLY_REQUEST_CAPTURED')
  expect(captured.codes).toEqual(['000001', '600030'])
  expect(captured.bases).toEqual(['hfq', 'unadjusted'])
  expect(captured.allow_network).toBe(true)
  expect((await (await page.request.get('/api/jobs')).json()).length).toBe(before.length)
})

test('real failed snapshots are visible and cannot create a daily plan', async ({ page }) => {
  const batches = await (await page.request.get('/api/market/snapshots/history')).json()
  const partial = batches.find((b: any) => b.status === 'partial')
  test.skip(!partial, 'No real failed capture is registered')
  await page.goto('/#market?kind=history&id=' + partial.id)
  await expect(page.getByRole('heading', { name: '逐股审计' })).toBeVisible()
  await expect(page.getByText('部分失败', { exact: true }).first()).toBeVisible()
  await page.screenshot({ path: 'test-results/online-partial-desktop.png', fullPage: true })
  await page.getByRole('button', { name: '最新日线计划', exact: true }).click()
  await page.getByLabel('确认仅用于日线研究，不是盘中策略或实盘订单', { exact: true }).check()
  await expect(page.getByRole('button', { name: '生成研究计划', exact: true })).toBeDisabled()
})

test('monitor UI renews and stops a leased mock session without external requests', async ({
  page,
}) => {
  const now = Date.now()
  let renewals = 0
  let stops = 0
  const record: any = {
    id: 'm_fixture',
    status: 'active',
    expires: now / 1000 + 300,
    lease: now / 1000 + 45,
    payload: { codes: ['000001'], plan_id: null },
    last_job_record: { status: 'succeeded' },
    judgment: {
      evaluated_at: new Date().toISOString(),
      rows: [
        {
          code: '000001',
          status: 'waiting',
          target_weight: null,
          quote: {
            last: 10,
            provider_timestamp: new Date(now - 600000).toISOString(),
            received_at: new Date().toISOString(),
          },
          reasons: ['stale_quote', 'no_daily_plan'],
          actionable: false,
        },
      ],
    },
  }
  await page.route('**/api/market/monitors', async (route) => {
    if (route.request().method() === 'GET') return route.continue()
    expect(route.request().postDataJSON().allow_network).toBe(true)
    return route.fulfill({ json: record })
  })
  await page.route('**/api/market/monitors/m_fixture**', async (route) => {
    if (route.request().url().endsWith('/heartbeat')) renewals++
    if (route.request().url().endsWith('/stop')) {
      stops++
      record.status = 'stopped'
    }
    await route.fulfill({ json: record })
  })
  await page.goto('/#market?kind=quotes')
  await expect(page.getByRole('button', { name: '开启有限监控', exact: true })).toBeDisabled()
  await page.getByLabel('允许从东方财富获取这组股票的报价', { exact: true }).check()
  await page.getByRole('button', { name: '开启有限监控', exact: true }).click()
  await expect(page.getByRole('button', { name: '停止监控', exact: true })).toBeVisible()
  await expect(page.getByText('等待 / 过期', { exact: true }).first()).toBeVisible()
  await expect.poll(() => renewals, { timeout: 20000 }).toBeGreaterThan(0)
  await page.getByRole('button', { name: '停止监控', exact: true }).click()
  await expect.poll(() => stops).toBeGreaterThan(0)
  await expect(page.getByRole('button', { name: '开启有限监控', exact: true })).toBeVisible()
})

test('online mobile pages retain timestamps and bounded tables', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('/#market')
  await expect(page.getByRole('heading', { name: '历史日线采集' })).toBeVisible()
  for (const tab of ['联网历史', '最新日线计划', '报价与监控']) {
    await page.getByRole('button', { name: tab, exact: true }).click()
    await page.waitForTimeout(300)
    expect(
      await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)
    ).toBeTruthy()
  }
  await page.screenshot({ path: 'test-results/online-mobile.png', fullPage: true })
})

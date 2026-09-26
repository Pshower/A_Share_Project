import { test, expect } from '@playwright/test'

test('supplement types submit explicit scoped parameters without real downloads', async ({
  page,
}) => {
  let request: any = null
  await page.route('**/api/market/supplements', (route) => {
    request = route.request().postDataJSON()
    return route.fulfill({ status: 409, json: { error: 'SUPPLEMENT_CAPTURE_ONLY' } })
  })
  await page.goto('/#market?kind=supplements')
  await expect(page.getByRole('heading', { name: '扩展数据采集' })).toBeVisible()
  await expect(page.getByLabel('扩展数据类型').locator('option')).toHaveCount(5)
  const submit = page.getByRole('button', { name: '下载扩展数据', exact: true })
  await expect(submit).toBeDisabled()
  await page.getByLabel('扩展数据类型').selectOption('minute')
  await page.getByLabel('分钟周期').selectOption('1')
  await page.getByLabel('扩展数据代码').fill('000001 000001 600030')
  await page.getByLabel('允许本次从所示供应商下载扩展数据').check()
  await submit.click()
  await expect(page.getByRole('alert')).toContainText('SUPPLEMENT_CAPTURE_ONLY')
  expect(request.data_type).toBe('minute')
  expect(request.period).toBe('1')
  expect(request.codes).toEqual(['000001', '600030'])
  await page.getByRole('button', { name: '关闭错误', exact: true }).click()
  await page.getByLabel('扩展数据类型').selectOption('dividend')
  await expect(submit).toBeDisabled()
  await expect(page.getByText('stock_history_dividend_detail · 新浪财经')).toBeVisible()
  await page.screenshot({ path: 'test-results/supplement-desktop.png', fullPage: true })
  await page.setViewportSize({ width: 390, height: 844 })
  await page.waitForTimeout(400)
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)
  ).toBeTruthy()
  await page.screenshot({ path: 'test-results/supplement-mobile.png', fullPage: true })
})

test('saved supplementary table can be inspected without provider calls', async ({ page }) => {
  const fixture: any = {
    id: 's_0123456789abcdef',
    created_at: '2026-09-26T01:00:00Z',
    status: 'ready',
    api: 'stock_individual_info_em',
    provider: 'eastmoney',
    akshare_version: 'fixture',
    request: { data_type: 'profile' },
    entries: [{ code: '000001', rows: 2, status: 'ready', columns: ['item', 'value'] }],
  }
  await page.route('**/api/market/snapshots/supplements', (route) =>
    route.fulfill({ json: [fixture] })
  )
  await page.route('**/api/market/supplements/s_0123456789abcdef/rows**', (route) =>
    route.fulfill({
      json: {
        columns: ['item', 'value'],
        rows: [
          ['股票代码', '000001'],
          ['股票简称', '本地测试样例'],
        ],
        total: 2,
      },
    })
  )
  await page.goto('/#market?kind=supplements&id=' + fixture.id)
  await expect(page.getByText('本地测试样例', { exact: true })).toBeVisible()
  await expect(page.getByRole('link', { name: 'CSV', exact: true })).toHaveAttribute(
    'href',
    /\/files\/000001\/csv$/
  )
  await expect(page.getByRole('button', { name: '下一页', exact: true })).toBeDisabled()
})

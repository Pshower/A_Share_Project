import { test, expect } from '@playwright/test'

test('saved real history renders candles and curves on desktop and mobile without downloads', async ({
  page,
}) => {
  const batches = await (await page.request.get('/api/market/snapshots/history')).json()
  const batch = batches.find((b: any) =>
    b.entries.some((e: any) => e.status === 'ready' && e.basis === 'unadjusted')
  )
  test.skip(!batch, 'No saved history available')
  let downloads = 0
  page.on('request', (r) => {
    if (r.method() === 'POST' && /market\/(downloads|quotes|monitors)/.test(r.url())) downloads++
  })
  await page.goto('/#market?kind=history&id=' + batch.id)
  const chart = page.getByTestId('market-chart')
  await expect(chart.locator('canvas')).toBeVisible()
  await expect(page.getByLabel('图表价格口径')).toHaveValue('unadjusted')
  const coloredPixels = () =>
    chart.locator('canvas').evaluate((canvas: HTMLCanvasElement) => {
      const data = canvas.getContext('2d')!.getImageData(0, 0, canvas.width, canvas.height).data
      let count = 0
      for (let i = 0; i < data.length; i += 4)
        if (
          data[i + 3] &&
          Math.max(data[i], data[i + 1], data[i + 2]) -
            Math.min(data[i], data[i + 1], data[i + 2]) >
            40
        )
          count++
      return count
    })
  expect(await coloredPixels()).toBeGreaterThan(200)
  await chart.scrollIntoViewIfNeeded()
  await page.screenshot({ path: 'test-results/market-chart-desktop.png', fullPage: true })
  await page.getByRole('button', { name: '收盘价曲线', exact: true }).click()
  await expect(page.getByRole('button', { name: '收盘价曲线', exact: true })).toHaveAttribute(
    'aria-pressed',
    'true'
  )
  await page.getByLabel('图表价格口径').selectOption('hfq')
  await expect(page.getByText('后复权研究价格 · 非实际报价', { exact: true })).toBeVisible()
  await page.getByLabel('图表时间范围').selectOption('0')
  expect(await coloredPixels()).toBeGreaterThan(200)
  await page.setViewportSize({ width: 390, height: 844 })
  await chart.scrollIntoViewIfNeeded()
  await page.waitForTimeout(300)
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)
  ).toBeTruthy()
  expect(await coloredPixels()).toBeGreaterThan(200)
  await page.screenshot({ path: 'test-results/market-chart-mobile.png', fullPage: true })
  expect(downloads).toBe(0)
})

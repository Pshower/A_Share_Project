import { test, expect } from '@playwright/test'

test('training dates follow selected stock intersection and local source charts are linked', async ({
  page,
}) => {
  const datasets = await (await page.request.get('/api/datasets')).json()
  const dataset = datasets.find((d: any) => d.name === 'research_v3')
  const token = (await (await page.request.get('/api/session')).json()).token
  async function coverage(codes: string[], lookback = 1) {
    return (
      await page.request.post(`/api/datasets/${dataset.id}/coverage`, {
        headers: { 'X-CSRF-Token': token },
        data: { codes, lookback },
      })
    ).json()
  }
  await page.goto('/')
  await page.getByRole('button', { name: '新建训练', exact: true }).click()
  await expect(page.getByLabel('搜索股票代码')).toBeVisible()
  await page.getByRole('button', { name: '清空', exact: true }).click()
  await expect(page.getByRole('button', { name: '开始训练', exact: true })).toBeDisabled()
  await page.getByLabel('000001', { exact: true }).check()
  const one = await coverage(['000001'])
  await expect(page.getByTestId('training-common-start')).toHaveText(one.train_start, {
    timeout: 20000,
  })
  await page.getByLabel('搜索股票代码').fill('301165')
  await page.getByLabel('301165', { exact: true }).check()
  await expect(page.getByRole('alert')).toContainText('301165', { timeout: 20000 })
  await expect(page.getByRole('button', { name: '开始训练', exact: true })).toBeDisabled()
  await page.getByLabel('301165', { exact: true }).uncheck()
  await page.getByLabel('搜索股票代码').fill('688981')
  await page.getByLabel('688981', { exact: true }).check()
  const both = await coverage(['000001', '688981'])
  expect(both.train_start > one.train_start).toBeTruthy()
  await expect(page.getByTestId('training-common-start')).toHaveText(both.train_start, {
    timeout: 20000,
  })
  await expect(page.getByTestId('training-common-end')).toHaveText(both.train_end)
  await page.getByLabel('观测窗口', { exact: true }).fill('5')
  const window = await coverage(['000001', '688981'], 5)
  await expect(page.getByTestId('training-common-start')).toHaveText(window.train_start, {
    timeout: 20000,
  })
  await page.getByText('已下载数据与日期交集', { exact: true }).click()
  await page.screenshot({ path: 'test-results/training-coverage-desktop.png', fullPage: true })
  await page.setViewportSize({ width: 390, height: 844 })
  await page.waitForTimeout(400)
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBeTruthy()
  await page.screenshot({ path: 'test-results/training-coverage-mobile.png', fullPage: true })
  await page.setViewportSize({ width: 1440, height: 1000 })
  await page.getByRole('button', { name: '查看本版本原始行情', exact: true }).click()
  await expect(page.getByLabel('历史行情来源')).toHaveValue('dataset:' + dataset.id)
  await expect(page.getByTestId('market-chart').locator('canvas')).toBeVisible({ timeout: 20000 })
  await expect(page.getByLabel('图表价格口径')).toHaveValue('hfq')
  await page.screenshot({ path: 'test-results/dataset-source-market.png', fullPage: true })
  await page.setViewportSize({ width: 390, height: 844 })
  await page.waitForTimeout(400)
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)
  ).toBeTruthy()
})

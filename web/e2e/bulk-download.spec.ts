import { test, expect } from '@playwright/test'

test('batch stock list imports dataset and CSV, deduplicates and submits only with consent', async ({
  page,
}) => {
  let submitted: any = null
  await page.route('**/api/market/downloads', (route) => {
    submitted = route.request().postDataJSON()
    return route.fulfill({ status: 409, json: { error: 'BATCH_REQUEST_CAPTURED_NO_DOWNLOAD' } })
  })
  await page.goto('/#market')
  await page.getByLabel('代码导入方式').selectOption('replace')
  await page.getByRole('button', { name: '导入数据集股票池', exact: true }).click()
  await expect(page.getByTestId('batch-code-summary')).toContainText('298 / 500')
  await page.getByLabel('代码导入方式').selectOption('append')
  await page
    .getByLabel('导入股票代码文件')
    .setInputFiles({
      name: 'stocks.csv',
      mimeType: 'text/csv',
      buffer: Buffer.from('code,name\n000001,duplicate\n999998,new\n999998,duplicate'),
    })
  await expect(page.getByTestId('batch-code-summary')).toContainText('299 / 500')
  const submit = page.getByRole('button', { name: '开始联网下载', exact: true })
  await expect(submit).toBeDisabled()
  await page.getByLabel('允许本次从东方财富联网采集', { exact: true }).check()
  await submit.click()
  await expect(page.getByRole('alert')).toContainText('BATCH_REQUEST_CAPTURED_NO_DOWNLOAD')
  expect(submitted.codes.length).toBe(299)
  expect(new Set(submitted.codes).size).toBe(299)
  await page.getByLabel('联网股票代码', { exact: true }).fill('000001 bad')
  await expect(submit).toBeDisabled()
  await page.getByLabel('联网股票代码', { exact: true }).fill('000001 000001 600030')
  await expect(page.getByTestId('batch-code-summary')).toContainText('重复 1 项')
  await page.getByRole('button', { name: '关闭错误', exact: true }).click()
  await page.screenshot({ path: 'test-results/bulk-download-desktop.png', fullPage: true })
  await page.setViewportSize({ width: 390, height: 844 })
  await page.waitForTimeout(400)
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)
  ).toBeTruthy()
  await page.screenshot({ path: 'test-results/bulk-download-mobile.png', fullPage: true })
})

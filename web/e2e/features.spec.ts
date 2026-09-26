import { test, expect } from '@playwright/test'

test('custom features preview, build a new dataset and pass PPO check without training', async ({
  page,
}) => {
  const datasets = await (await page.request.get('/api/datasets')).json()
  const original = datasets.find((d: any) => d.name === 'research_v3')
  const originalDetail = await (await page.request.get('/api/datasets/' + original.id)).json()
  const name = 'feature_ui_' + Date.now()
  await page.goto('/#data')
  await page.getByRole('button', { name: '新建特征', exact: true }).click()
  await page.getByRole('button', { name: '清空', exact: true }).click()
  await page.getByLabel('000001', { exact: true }).check()
  await page.getByLabel('搜索股票代码').fill('600030')
  await page.getByLabel('600030', { exact: true }).check()
  await page.getByRole('button', { name: '添加特征', exact: true }).click()
  await page.getByRole('button', { name: '清空追加特征', exact: true }).click()
  await expect(page.getByLabel('特征名称 1', { exact: true })).toHaveCount(0)
  await page.getByRole('button', { name: '添加特征', exact: true }).click()
  await page.getByLabel('特征名称 1', { exact: true }).fill('custom_atr_14')
  await page.getByRole('button', { name: '添加特征', exact: true }).click()
  await page.getByLabel('特征名称 2', { exact: true }).fill('custom_dd_20')
  await page.getByLabel('特征类型 2', { exact: true }).selectOption('drawdown')
  await page.getByLabel('特征窗口 2', { exact: true }).fill('20')
  await page.getByRole('button', { name: '添加特征', exact: true }).click()
  await page.getByRole('button', { name: '删除特征 3', exact: true }).click()
  await page.getByRole('button', { name: '预览特征', exact: true }).click()
  await expect(page.getByRole('heading', { name: '特征预览 · 000001' })).toBeVisible({
    timeout: 20000,
  })
  await expect(page.locator('.feature-editor canvas')).toBeVisible()
  await page.screenshot({ path: 'test-results/feature-editor-desktop.png', fullPage: true })
  await page.setViewportSize({ width: 390, height: 844 })
  await page.waitForTimeout(400)
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)
  ).toBeTruthy()
  await page.screenshot({ path: 'test-results/feature-editor-mobile.png', fullPage: true })
  await page.setViewportSize({ width: 1440, height: 1000 })
  await page.getByLabel('新版本标识', { exact: true }).fill(name)
  await page.getByRole('button', { name: '开始构建', exact: true }).click()
  await expect(page).toHaveURL(/#jobs\?job=j_/)
  const id = new URLSearchParams(page.url().split('?')[1]).get('job')!
  await expect
    .poll(async () => (await (await page.request.get('/api/jobs/' + id)).json()).status, {
      timeout: 180000,
    })
    .toBe('succeeded')
  const job = await (await page.request.get('/api/jobs/' + id)).json()
  const detail = await (await page.request.get('/api/datasets/' + job.result.dataset_id)).json()
  expect(detail.features).toBe(22)
  expect(detail.manifest.extra_features.map((d: any) => d.name)).toEqual([
    'custom_atr_14',
    'custom_dd_20',
  ])
  expect(
    (await (await page.request.get('/api/datasets/' + original.id)).json()).manifest.artifacts
  ).toEqual(originalDetail.manifest.artifacts)
  const token = (await (await page.request.get('/api/session')).json()).token
  const check = await (
    await page.request.post('/api/training/validate', {
      headers: { 'X-CSRF-Token': token },
      data: { dataset_id: detail.id, codes: ['000001', '600030'], device: 'auto' },
    })
  ).json()
  expect(check.id).toBeTruthy()
  await expect
    .poll(async () => (await (await page.request.get('/api/jobs/' + check.id)).json()).status, {
      timeout: 180000,
    })
    .toBe('succeeded')
  const verified = await (await page.request.get('/api/jobs/' + check.id)).json()
  expect(verified.check.observation_shape).toEqual([2, 1, 22])
  expect(verified.check.parameters_unchanged).toBe(true)
  expect(verified.check.trained).toBe(false)
  await page.goto('/#data')
  await page.getByLabel('数据版本', { exact: true }).selectOption(detail.id)
  await expect(page.getByText('当前版本特征清单 · 22 列', { exact: true })).toBeVisible()
})

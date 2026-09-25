import { test, expect, type Page } from '@playwright/test'

let trainedModel = ''
let trainedJob = ''
let predictionJob = ''
let evaluationJob = ''
test.describe.configure({ mode: 'serial' })

async function ready(page: Page) {
  await page.route('**/*', (route) => {
    const url = new URL(route.request().url())
    if (url.protocol === 'http:' && ['127.0.0.1', 'localhost'].includes(url.hostname))
      return route.continue()
    return route.abort()
  })
  await page.goto('/')
  await expect(page.getByRole('heading', { name: '训练工作台', exact: true })).toBeVisible()
  await expect(page.getByText('正在读取本地研究资源…')).toHaveCount(0)
}

async function waitForJob(page: Page) {
  await expect(page).toHaveURL(/#jobs\?job=j_/)
  const id = new URLSearchParams(page.url().split('?')[1]).get('job')!
  await expect
    .poll(async () => (await (await page.request.get('/api/jobs/' + id)).json()).status, {
      timeout: 180000,
      message: 'Worker must complete without errors',
    })
    .toBe('succeeded')
  await expect(page.getByText('已完成', { exact: true }).first()).toBeVisible()
  return id
}

test('desktop dashboard, real chart and local data browser', async ({ page }) => {
  const errors: string[] = []
  page.on('pageerror', (error) => errors.push(error.message))
  await ready(page)
  const canvas = page.locator('[data-testid="chart"] canvas').first()
  await expect(canvas).toBeVisible()
  const pixels = await canvas.evaluate((element: HTMLCanvasElement) => {
    const values = element.getContext('2d')!.getImageData(0, 0, element.width, element.height).data
    let count = 0
    for (let i = 0; i < values.length; i += 4) if (values[i + 3] > 100 && values[i] < 220) count++
    return count
  })
  expect(pixels).toBeGreaterThan(1000)
  await page.screenshot({ path: 'test-results/desktop-dashboard.png', fullPage: true })
  await page.getByRole('link', { name: '数据管理' }).click()
  await expect(page.getByRole('heading', { name: '股票数据质量' })).toBeVisible()
  await expect(page.locator('.quality-table tbody tr')).toHaveCount(298)
  expect(errors).toEqual([])
})

test('real 128-step training through UI, reload and stored checkpoint selection', async ({
  page,
}) => {
  await ready(page)
  await page.getByRole('button', { name: '新建训练' }).click()
  await page.getByLabel('实验名称', { exact: true }).fill('Web 验收 · 128 steps')
  await page.getByLabel('总训练步数', { exact: true }).fill('128')
  await page.getByLabel('Rollout 步数', { exact: true }).fill('64')
  await page.getByLabel('Batch 大小', { exact: true }).fill('32')
  await page.getByLabel('每轮 Epochs', { exact: true }).fill('2')
  await page.getByLabel('验证间隔 / Rollout', { exact: true }).fill('2')
  await page.getByRole('button', { name: '开始训练', exact: true }).click()
  await expect(page).toHaveURL(/#jobs\?job=j_/)
  await page.reload()
  trainedJob = await waitForJob(page)
  const job = await (await page.request.get('/api/jobs/' + trainedJob)).json()
  expect(job.metrics.filter((e: any) => e.type === 'metrics')).toHaveLength(2)
  const models = await (await page.request.get('/api/models')).json()
  trainedModel = models.find((m: any) => m.run === trainedJob && m.name === 'best_128').id
  expect(trainedModel).toBeTruthy()
  await page.getByRole('button', { name: '验证与模型', exact: true }).click()
  await page.getByRole('button', { name: 'best_128', exact: true }).click()
  await expect(page.getByRole('heading', { name: '模型详情' })).toBeVisible()
  const modelLabel = '浏览器验收 ' + trainedJob.slice(-6)
  await page.getByLabel('显示名称', { exact: true }).fill(modelLabel)
  await page.getByRole('button', { name: '保存名称', exact: true }).click()
  await expect(page.getByRole('button', { name: modelLabel, exact: true })).toBeVisible()
  await page.screenshot({ path: 'test-results/desktop-models.png', fullPage: true })
})

test('validation backtest from saved model and report inspection', async ({ page }) => {
  await ready(page)
  await page.goto('/#backtest?model=' + trainedModel)
  await page.getByRole('button', { name: '开始回测', exact: true }).click()
  evaluationJob = await waitForJob(page)
  const job = await (await page.request.get('/api/jobs/' + evaluationJob)).json()
  const report = await (await page.request.get('/api/reports/' + job.result.report_id)).json()
  expect(report.split).toBe('val')
  expect(report.metrics.ppo.trading_steps).toBe(291)
  expect(Object.keys(report.metrics)).toHaveLength(7)
  await page.goto('/#backtest?report=' + job.result.report_id)
  await expect(page.getByRole('heading', { name: '订单明细' })).toBeVisible()
  await expect(page.locator('.orders-table tbody tr')).toHaveCount(50)
  await page.getByRole('button', { name: '持仓', exact: true }).click()
  await expect(page.getByRole('heading', { name: '持仓明细' })).toBeVisible()
  await expect(page.locator('.orders-table tbody tr').first()).toBeVisible()
  await page.screenshot({ path: 'test-results/desktop-backtest.png', fullPage: true })
})

test('as-of prediction, manual portfolio and one-stock display does not shrink pool', async ({
  page,
}) => {
  await ready(page)
  await page.goto('/#predict?model=' + trainedModel)
  await page.getByLabel('观察日期', { exact: true }).fill('2023-10-20')
  await page.getByLabel('账户来源', { exact: true }).selectOption('manual')
  await page.getByLabel('可用现金', { exact: true }).fill('900000')
  await page.getByRole('button', { name: '添加持仓', exact: false }).click()
  await page.getByLabel('持仓股数 1', { exact: true }).fill('100')
  await page.getByLabel('可卖股数 1', { exact: true }).fill('0')
  await page.getByRole('button', { name: '清空', exact: true }).click()
  await page.getByLabel('000001', { exact: true }).check()
  await page.getByRole('button', { name: '预测动作', exact: true }).click()
  predictionJob = await waitForJob(page)
  const job = await (await page.request.get('/api/jobs/' + predictionJob)).json()
  expect(job.prediction.pool_size).toBe(32)
  expect(job.prediction.rows).toHaveLength(32)
  expect(job.prediction.display_codes).toEqual(['000001'])
  expect(job.prediction.rows.find((r: any) => r.code === '000001').available_shares).toBe(0)
  await page.getByRole('button', { name: '查看预测结果' }).click()
  await expect(page.getByLabel('观察日期', { exact: true })).toHaveValue('2023-10-20')
  await expect(page.getByLabel('账户来源', { exact: true })).toHaveValue('saved:' + predictionJob)
  await expect(page.getByLabel('000001', { exact: true })).toBeChecked()
  await expect(page.getByLabel('000408', { exact: true })).not.toBeChecked()
  await expect(page.locator('.prediction-result tbody tr')).toHaveCount(1)
  await page.screenshot({ path: 'test-results/desktop-prediction.png', fullPage: true })
  await page.getByLabel('账户来源', { exact: true }).selectOption('manual')
  await expect(page.locator('.prediction-result')).toHaveCount(0)
})

test('queued cancellation and active stop are real task states', async ({ page }) => {
  await ready(page)
  await page.getByRole('button', { name: '新建训练' }).click()
  await page.getByLabel('总训练步数', { exact: true }).fill('100000')
  await page.getByLabel('实验名称', { exact: true }).fill('Web 停止验收')
  await page.getByRole('button', { name: '开始训练', exact: true }).click()
  await expect(page).toHaveURL(/#jobs\?job=j_/)
  const id = new URLSearchParams(page.url().split('?')[1]).get('job')!
  await expect
    .poll(async () => (await (await page.request.get('/api/jobs/' + id)).json()).status)
    .toBe('running')
  await expect
    .poll(
      async () =>
        (await (await page.request.get('/api/jobs/' + id)).json()).events.some(
          (e: any) => e.type === 'metrics'
        ),
      { timeout: 60000 }
    )
    .toBeTruthy()
  await page.goto('/#training?new=1')
  await page.getByRole('button', { name: '检查配置', exact: true }).click()
  await expect(page).toHaveURL(/#jobs\?job=j_/)
  const queued = new URLSearchParams(page.url().split('?')[1]).get('job')!
  expect((await (await page.request.get('/api/jobs/' + queued)).json()).status).toBe('queued')
  await page.getByRole('button', { name: '请求停止', exact: true }).click()
  await expect
    .poll(async () => (await (await page.request.get('/api/jobs/' + queued)).json()).status)
    .toBe('cancelled')
  await page.goto('/#jobs?job=' + id)
  await page.getByRole('button', { name: '请求停止', exact: true }).click()
  await expect
    .poll(async () => (await (await page.request.get('/api/jobs/' + id)).json()).status, {
      timeout: 30000,
    })
    .toBe('cancelled')
  const job = await (await page.request.get('/api/jobs/' + id)).json()
  expect(job.result).toBeNull()
})

test('build a two-stock local dataset and register it without downloads', async ({ page }) => {
  await ready(page)
  await page.getByRole('link', { name: '数据管理' }).click()
  await page.getByRole('button', { name: '构建新版本', exact: true }).click()
  await page.getByLabel('新版本标识', { exact: true }).fill('web_check_' + Date.now())
  await page.getByLabel('研究起始', { exact: true }).fill('2020-01-01')
  await page.getByLabel('训练截止', { exact: true }).fill('2021-12-31')
  await page.getByLabel('验证截止', { exact: true }).fill('2022-12-30')
  await page.getByLabel('研究截止', { exact: true }).fill('2023-12-29')
  await page.getByRole('button', { name: '清空', exact: true }).click()
  await page.getByLabel('000001', { exact: true }).check()
  await page.getByLabel('600030', { exact: true }).check()
  await page.getByRole('button', { name: '开始构建', exact: true }).click()
  const jobId = await waitForJob(page)
  const job = await (await page.request.get('/api/jobs/' + jobId)).json()
  const data = await (await page.request.get('/api/datasets/' + job.result.dataset_id)).json()
  expect(data.stocks).toBe(2)
  expect(data.manifest.build_config.train_codes).toEqual(['000001', '600030'])
})

test('saved prediction account can be reused without changing the action', async ({ page }) => {
  await ready(page)
  await page.goto('/#predict?model=' + trainedModel)
  await page.getByLabel('账户来源', { exact: true }).selectOption('saved:' + predictionJob)
  await expect(page.getByLabel('观察日期', { exact: true })).toHaveValue('2023-10-20')
  await expect(page.getByLabel('观察日期', { exact: true })).toBeDisabled()
  await page.getByRole('button', { name: '预测动作', exact: true }).click()
  const id = await waitForJob(page)
  const original = await (await page.request.get('/api/jobs/' + predictionJob)).json()
  const copied = await (await page.request.get('/api/jobs/' + id)).json()
  expect(copied.prediction.rows).toEqual(original.prediction.rows)
})

test('freeze selected checkpoint without automatically evaluating the test split', async ({
  page,
}) => {
  await ready(page)
  await page.goto('/#backtest?model=' + trainedModel)
  await page.getByLabel('评价区间', { exact: true }).selectOption('test')
  await expect(page.getByRole('button', { name: '开始回测', exact: true })).toBeDisabled()
  await page.getByRole('button', { name: '冻结验证所选模型', exact: true }).click()
  const id = await waitForJob(page)
  const job = await (await page.request.get('/api/jobs/' + id)).json()
  expect(job.kind).toBe('freeze')
  expect(job.result.frozen_path).toContain('frozen_selection.json')
})

test('explicit stock-pool transfer creates a reusable model, not a training run', async ({
  page,
}) => {
  await ready(page)
  await page.goto('/#models?model=' + trainedModel)
  await page.getByRole('button', { name: '显式迁移股票池', exact: false }).click()
  await page.getByRole('button', { name: '清空', exact: true }).click()
  await page.getByLabel('000001', { exact: true }).check()
  await page.getByRole('button', { name: '创建迁移模型', exact: true }).click()
  const id = await waitForJob(page)
  const job = await (await page.request.get('/api/jobs/' + id)).json()
  const models = await (await page.request.get('/api/models')).json()
  const transferred = models.find((m: any) => m.id === job.result.model_id)
  expect(transferred.pool_size).toBe(1)
  expect(transferred.timesteps).toBe(0)
  expect(transferred.metadata.provenance.source_num_timesteps).toBe(128)
})

test('configuration check retries an expired local token without duplicating the task', async ({
  page,
}) => {
  await ready(page)
  let attempts = 0
  const keys: string[] = []
  await page.route('**/api/training/validate', async (route) => {
    keys.push(route.request().headers()['idempotency-key'])
    attempts++
    if (attempts === 1)
      return route.fulfill({
        status: 403,
        contentType: 'application/json',
        body: JSON.stringify({ error: 'Invalid local session token' }),
      })
    return route.continue()
  })
  await page.goto('/#training?new=1')
  await page.getByRole('button', { name: '检查配置', exact: true }).click()
  const id = await waitForJob(page)
  const job = await (await page.request.get('/api/jobs/' + id)).json()
  expect(keys).toHaveLength(2)
  expect(keys[0]).toBe(keys[1])
  expect(job.check.parameters_unchanged).toBeTruthy()
  expect(job.check.trained).toBe(false)
  await expect(page.getByText('参数未改变')).toBeVisible()
})

test('mobile layouts remain usable with no page-level horizontal overflow', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await ready(page)
  for (const hash of [
    'training',
    'training?new=1',
    'data',
    'models?model=' + trainedModel,
    'backtest',
    'predict?model=' + trainedModel + '&result=' + predictionJob,
    'jobs?job=' + trainedJob,
  ]) {
    await page.goto('/#' + hash)
    await page.waitForTimeout(1200)
    expect(
      await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1),
      hash
    ).toBeTruthy()
    await page.screenshot({
      path:
        'test-results/mobile-' + hash.split('?')[0] + (hash.includes('new') ? '-new' : '') + '.png',
      fullPage: true,
    })
  }
  await page.getByRole('button', { name: '打开导航' }).click()
  await expect(page.getByRole('link', { name: '数据管理' })).toBeVisible()
})

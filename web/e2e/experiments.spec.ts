import { test, expect } from '@playwright/test'

test('immutable experiment prepares data, checks seeds and trains only after approval', async ({
  page,
}) => {
  const version = 'workflow_ui_' + Date.now()
  await page.goto('/#experiments')
  await page.getByLabel('一体化实验名称').fill(version)
  await page.getByLabel('实验股票代码').fill('000001 600030')
  await page.getByLabel('构建独立数据与特征版本').check()
  await page.getByLabel('实验新数据版本').fill(version)
  await page.getByLabel('实验随机种子').fill('42 123')
  await page.getByLabel('实验计算设备').selectOption('cpu')
  await page.getByLabel('实验total_timesteps').fill('64')
  await page.getByLabel('实验n_steps').fill('32')
  await page.getByLabel('实验batch_size').fill('16')
  await page.getByLabel('实验n_epochs').fill('1')
  await page.getByLabel('实验validation_rollouts').fill('2')
  await page.getByText('PPO 高级参数与选模约束', { exact: true }).click()
  await page.getByLabel('学习率日程').selectOption('linear')
  await page.getByRole('button', { name: '保存实验配置', exact: true }).click()
  await expect(page).toHaveURL(/#experiments\?id=e_/)
  const id = new URLSearchParams(page.url().split('?')[1]).get('id')!
  const state = async () => await (await page.request.get('/api/experiments/' + id)).json()
  await expect(page.getByRole('button', { name: '启动已检查实验', exact: true })).toBeDisabled()
  await page.getByRole('button', { name: '准备数据', exact: true }).click()
  await expect.poll(async () => (await state()).can_check, { timeout: 180000 }).toBe(true)
  await page.getByRole('button', { name: '检查全部种子', exact: true }).click()
  await expect.poll(async () => (await state()).checks_ready, { timeout: 180000 }).toBe(true)
  const checked = await state()
  expect(checked.jobs.filter((j: any) => j.kind === 'training')).toHaveLength(0)
  await page.getByRole('button', { name: '刷新实验状态', exact: true }).click()
  await page.getByLabel(/批准 2 个种子的参数更新/).check()
  await page.getByRole('button', { name: '启动已检查实验', exact: true }).click()
  await expect
    .poll(
      async () =>
        (await state()).jobs.filter((j: any) => j.kind === 'training' && j.status === 'succeeded')
          .length,
      { timeout: 240000 }
    )
    .toBe(2)
  const completed = await state()
  expect(
    completed.jobs.filter((j: any) => ['failed', 'interrupted'].includes(j.status))
  ).toHaveLength(0)
  expect(completed.summaries).toHaveLength(2)
  const train = completed.jobs.find((j: any) => j.kind === 'training')
  const job = await (await page.request.get('/api/jobs/' + train.id)).json()
  const metrics = job.metrics.filter((e: any) => e.type === 'metrics').at(-1).metrics
  expect(metrics['optimizer/steps_max']).toBeGreaterThan(0)
  expect(metrics['portfolio/gross_turnover']).toBeGreaterThanOrEqual(0)
  expect(metrics['train/learning_rate']).toBeGreaterThan(0)
  await page.getByRole('button', { name: '刷新实验状态', exact: true }).click()
  await expect(page.getByText('运行中', { exact: true })).toHaveCount(0, { timeout: 30000 })
  await page.screenshot({ path: 'test-results/experiment-workflow-desktop.png', fullPage: true })
  await page.goto('/#jobs?job=' + train.id)
  await expect(page.getByRole('button', { name: '组合行为', exact: true })).toBeVisible()
  await page.getByRole('button', { name: '组合行为', exact: true }).click()
  await expect(page.getByRole('heading', { name: '目标与实际现金' })).toBeVisible()
  await page.screenshot({ path: 'test-results/ppo-diagnostics-desktop.png', fullPage: true })
  await page.setViewportSize({ width: 390, height: 844 })
  await page.waitForTimeout(400)
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)
  ).toBeTruthy()
  await page.screenshot({ path: 'test-results/ppo-diagnostics-mobile.png', fullPage: true })
})

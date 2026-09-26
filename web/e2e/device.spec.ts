import { test, expect } from '@playwright/test'

test('device selection runs a CUDA configuration check without training', async ({ page }) => {
  const system = await (await page.request.get('/api/system')).json()
  test.skip(!system.cuda_available, 'Requires configured GPU environment')
  expect(system.device).toBe('cuda')
  await page.goto('/')
  await page.getByRole('button', { name: '新建训练', exact: true }).click()
  await expect(page.getByLabel('计算设备')).toHaveValue('auto')
  await page.getByLabel('计算设备').selectOption('cpu')
  await page.getByLabel('计算设备').selectOption('cuda')
  await page.getByRole('button', { name: '清空', exact: true }).click()
  await page.getByLabel('000001', { exact: true }).check()
  await page.getByLabel('实验名称', { exact: true }).fill('CUDA configuration check only')
  await page.screenshot({ path: 'test-results/gpu-device-desktop.png', fullPage: true })
  await page.getByRole('button', { name: '检查配置', exact: true }).click()
  await expect(page).toHaveURL(/#jobs\?job=j_/)
  const id = new URLSearchParams(page.url().split('?')[1]).get('job')!
  await expect
    .poll(async () => (await (await page.request.get('/api/jobs/' + id)).json()).status, {
      timeout: 180000,
    })
    .toBe('succeeded')
  const job = await (await page.request.get('/api/jobs/' + id)).json()
  expect(job.kind).toBe('check')
  expect(job.payload.device).toBe('cuda')
  expect(job.check.actual_device).toMatch(/^cuda/)
  expect(job.check.trained).toBe(false)
  expect(job.check.parameters_unchanged).toBe(true)
  expect(job.check.save_load_equal).toBe(true)
  expect(job.check.contract.training_date_policy).toBe('intersection')
  expect(job.check.train_initial_date).toBe(job.check.contract.train_interval.train_start)
  expect(job.check.train_final_date).toBe(job.check.contract.train_interval.train_end)
})

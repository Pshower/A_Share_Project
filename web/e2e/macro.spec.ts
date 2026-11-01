import { test, expect } from '@playwright/test'

test('macro secondary page reads a saved snapshot without creating jobs', async ({ page }) => {
  test.setTimeout(120_000)
  const snapshots = await (await page.request.get('/api/market/snapshots/macro')).json()
  const snapshot = snapshots.find((row: any) => row.status === 'ready') || snapshots[0]
  test.skip(!snapshot, 'No macro snapshot is registered')
  const before = await (await page.request.get('/api/jobs')).json()
  await page.goto('/#market?kind=macro')
  await expect(page.getByRole('heading', { name: '宏观环境' })).toBeVisible({ timeout: 90_000 })
  await expect(page.getByLabel('宏观快照')).toHaveValue(snapshot.id)
  await expect(page.getByLabel('宏观指标')).not.toHaveValue('')
  await expect(page.getByTestId('chart')).toBeVisible()
  await expect(page.locator('.macro-panel .online-boundary')).toContainText(snapshot.id)
  expect((await (await page.request.get('/api/jobs')).json()).length).toBe(before.length)
  await page.screenshot({ path: 'test-results/macro-secondary.png', fullPage: true })
})

import { test, expect } from '@playwright/test'

test('close-date control is available without starting downloads or training', async ({ page }) => {
  const before = await (await page.request.get('/api/jobs')).json()
  await page.goto('/#market')
  await page.getByRole('button', { name: '最新日线计划', exact: true }).click()
  const date = page.getByLabel('指定收盘日期', { exact: true })
  await expect(date).toBeVisible()
  await date.fill('2026-09-24')
  await expect(date).toHaveValue('2026-09-24')
  await page.screenshot({ path: 'test-results/eod-desktop.png', fullPage: true })
  await page.setViewportSize({ width: 390, height: 844 })
  await expect(date).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
  await page.screenshot({ path: 'test-results/eod-mobile.png', fullPage: true })
  expect((await (await page.request.get('/api/jobs')).json()).length).toBe(before.length)
})

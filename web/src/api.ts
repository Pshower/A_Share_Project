let session: Promise<string> | undefined

export async function api<T = any>(path: string, options: RequestInit = {}): Promise<T> {
  if (!session)
    session = fetch('/api/session')
      .then((r) => r.json())
      .then((v) => v.token)
      .catch((error) => {
        session = undefined
        throw error
      })
  const token = await session
  const response = await fetch('/api' + path, {
    ...options,
    headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': token, ...options.headers },
  })
  const data = await response.json()
  if (!response.ok) {
    const message =
      data.error ||
      (Array.isArray(data.detail)
        ? data.detail.map((e: any) => `${e.loc?.slice(1).join('.')}: ${e.msg}`).join('; ')
        : data.detail)
    throw new Error(message || `请求失败 (${response.status})`)
  }
  return data
}

export const percent = (value: unknown, digits = 2) =>
  typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : '—'
export const number = (value: unknown, digits = 0) =>
  typeof value === 'number' && Number.isFinite(value)
    ? value.toLocaleString('zh-CN', { maximumFractionDigits: digits })
    : '—'

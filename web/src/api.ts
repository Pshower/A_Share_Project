let session: Promise<string> | undefined

export async function api<T = any>(
  path: string,
  options: RequestInit = {},
  retry = true
): Promise<T> {
  if (!session)
    session = fetch('/api/session')
      .then((r) => r.json())
      .then((v) => v.token)
  const token = await session
  const response = await fetch('/api' + path, {
    ...options,
    headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': token, ...options.headers },
  })
  const data = await response.json()
  if (response.status === 403 && data.error === 'Invalid local session token' && retry) {
    session = undefined
    return api<T>(path, options, false)
  }
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

export const post = (path: string, data: unknown, key = crypto.randomUUID()) =>
  api(path, {
    method: 'POST',
    body: JSON.stringify(data),
    headers: { 'Idempotency-Key': key },
  })

export const percent = (value: unknown, digits = 2) =>
  typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : '—'
export const number = (value: unknown, digits = 0) =>
  typeof value === 'number' && Number.isFinite(value)
    ? value.toLocaleString('zh-CN', { maximumFractionDigits: digits })
    : '—'
export const dateTime = (value: string) =>
  value ? new Date(value).toLocaleString('zh-CN', { hour12: false }) : '—'
export const statusName: Record<string, string> = {
  queued: '排队中',
  running: '运行中',
  stop_requested: '正在停止',
  succeeded: '已完成',
  failed: '失败',
  cancelled: '已取消',
  interrupted: '已中断',
}
export const kindName: Record<string, string> = {
  training: 'PPO 训练',
  check: '配置检查',
  build: '构建数据',
  evaluation: '回测',
  freeze: '冻结模型',
  prediction: '动作预测',
  transfer: '股票池迁移',
}
export const phaseName: Record<string, string> = {
  validating: '校验输入',
  preparing: '准备数据',
  training: '训练',
  validation: '验证选模',
  saving: '保存模型',
  comparison: '基线比较',
  evaluating: '回测',
  predicting: '计算动作',
  building: '构建数据',
}

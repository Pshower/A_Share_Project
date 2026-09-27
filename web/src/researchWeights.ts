export function weightsFrom(text: string) {
  const weights: Record<string, number> = {}
  for (const row of text.split(/[\n,，]+/).filter((v) => v.trim())) {
    const [code, value, extra] = row.trim().split(/\s*[:=]\s*/)
    if (
      !/^\d{6}$/.test(code) ||
      value === undefined ||
      extra !== undefined ||
      !Number.isFinite(Number(value)) ||
      Number(value) < 0 ||
      code in weights
    )
      throw new Error('研究仓位格式：000001=0.10；代码不得重复')
    weights[code] = Number(value)
  }
  if (Object.values(weights).reduce((a, b) => a + b, 0) > 1)
    throw new Error('研究仓位合计不能超过 1')
  return weights
}

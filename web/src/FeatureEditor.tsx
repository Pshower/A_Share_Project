import { lazy, Suspense, useEffect, useRef, useState } from 'react'
import { Plus, Trash2, RefreshCw } from 'lucide-react'
import { api, post, number } from './api'
const Chart = lazy(() => import('./Chart'))

export type FeatureDefinition = { name: string; kind: string; window: number }
type Props = {
  datasetId: string
  codes: string[]
  definitions: FeatureDefinition[]
  onChange: (value: FeatureDefinition[]) => void
  onValidity: (valid: boolean) => void
  fail: (error: unknown) => void
}

export default function FeatureEditor(p: Props) {
  const [catalog, setCatalog] = useState<any>(null)
  const [stock, setStock] = useState('')
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<any>(null)
  const code = p.codes.includes(stock) ? stock : p.codes[0] || ''
  const key = JSON.stringify([p.datasetId, code, p.definitions])
  const currentKey = useRef(key)
  currentKey.current = key
  const preview = result?.key === key ? result.data : null
  useEffect(() => {
    api('/feature-types').then(setCatalog).catch(p.fail)
  }, [p.fail])
  const invalid =
    p.definitions.length > 12 ||
    p.definitions.some(
      (d, i) =>
        !/^[a-z][a-z0-9_]{2,39}$/.test(d.name) ||
        !Number.isInteger(d.window) ||
        d.window < 2 ||
        d.window > 250 ||
        p.definitions.findIndex((other) => other.name === d.name) !== i ||
        (catalog &&
          (catalog.reserved_names.includes(d.name) ||
            !catalog.kinds.some((k: any) => k.id === d.kind)))
    )
  useEffect(() => {
    p.onValidity(!invalid && (!!catalog || !p.definitions.length))
  }, [invalid, catalog, p.onValidity, p.definitions.length])
  function update(i: number, values: Partial<FeatureDefinition>) {
    p.onChange(p.definitions.map((d, j) => (i === j ? { ...d, ...values } : d)))
  }
  function add() {
    let index = 1
    while (p.definitions.some((d) => d.name === `custom_atr_${index}`)) index++
    p.onChange([...p.definitions, { name: `custom_atr_${index}`, kind: 'atr_ratio', window: 14 }])
  }
  async function loadPreview() {
    setBusy(true)
    try {
      const data = await post(`/datasets/${p.datasetId}/features/preview`, {
        code,
        extra_features: p.definitions,
      })
      if (currentKey.current === key) setResult({ key, data })
    } catch (e) {
      if (currentKey.current === key) p.fail(e)
    } finally {
      setBusy(false)
    }
  }
  return (
    <section className="section feature-editor">
      <div className="section-heading">
        <h2>追加特征定义</h2>
        <span>
          基础 {catalog?.base_features.length || 20} + 追加 {p.definitions.length} / 12
        </span>
      </div>
      <div className="table-wrap">
        <table className="feature-definition-table">
          <thead>
            <tr>
              <th>名称</th>
              <th>指标类型</th>
              <th>窗口</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {p.definitions.map((d, i) => (
              <tr key={i}>
                <td>
                  <input
                    aria-label={`特征名称 ${i + 1}`}
                    value={d.name}
                    maxLength={40}
                    onChange={(e) => update(i, { name: e.target.value })}
                  />
                </td>
                <td>
                  <select
                    aria-label={`特征类型 ${i + 1}`}
                    value={d.kind}
                    onChange={(e) => update(i, { kind: e.target.value })}
                  >
                    {catalog?.kinds.map((k: any) => (
                      <option key={k.id} value={k.id}>
                        {k.label}
                      </option>
                    ))}
                  </select>
                  <small className="subtext feature-formula">
                    {catalog?.kinds.find((k: any) => k.id === d.kind)?.formula}
                  </small>
                </td>
                <td>
                  <input
                    aria-label={`特征窗口 ${i + 1}`}
                    type="number"
                    min={2}
                    max={250}
                    value={d.window}
                    onChange={(e) => update(i, { window: Number(e.target.value) })}
                  />
                </td>
                <td>
                  <button
                    className="icon-button"
                    title={`删除特征 ${i + 1}`}
                    onClick={() => p.onChange(p.definitions.filter((_, j) => i !== j))}
                  >
                    <Trash2 size={16} />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {!p.definitions.length && <p className="subtle-note">仅使用基础特征</p>}
      {invalid && (
        <p role="alert">
          名称需唯一且不与保留字段重名（小写字母开头，3–40 位字母、数字、下划线）；窗口为 2–250
          的整数。
        </p>
      )}
      <div className="form-actions">
        <button onClick={add} disabled={!catalog || p.definitions.length >= 12}>
          <Plus size={15} />
          添加特征
        </button>
        <button className="icon-button" title="清空追加特征" disabled={!p.definitions.length} onClick={() => p.onChange([])}><Trash2 size={15} /></button>
        <select
          aria-label="特征预览股票"
          value={code}
          onChange={(e) => setStock(e.target.value)}
          disabled={!p.codes.length}
        >
          {p.codes.map((c) => (
            <option key={c}>{c}</option>
          ))}
        </select>
        <button
          disabled={busy || invalid || !p.definitions.length || !code || !catalog}
          onClick={loadPreview}
        >
          <RefreshCw size={15} />
          {busy ? '计算中…' : '预览特征'}
        </button>
      </div>
      {preview && (
        <>
          <div className="section-heading secondary-heading">
            <h2>特征预览 · {preview.code}</h2>
            <span>
              原始值 · {preview.start_date} 至 {preview.end_date}
            </span>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>特征</th>
                  <th>有效记录</th>
                  <th>缺失记录</th>
                  <th>首个有效日</th>
                </tr>
              </thead>
              <tbody>
                {preview.stats.map((s: any) => (
                  <tr key={s.name}>
                    <td>{s.name}</td>
                    <td>{number(s.valid)}</td>
                    <td>{number(s.missing)}</td>
                    <td>{s.first_valid || '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <Suspense fallback={<div className="empty">读取图表…</div>}>
            <Chart
              height={240}
              x={preview.rows.map((r: any) => r.date)}
              series={preview.columns.map((name: string) => ({
                name,
                values: preview.rows.map((r: any) => r[name]),
              }))}
            />
          </Suspense>
          <p className="subtle-note">
            预览采用所选版本已存日历，显示末 120
            条原始指标值；新版本的有效范围及标准化结果以实际构建为准。
          </p>
        </>
      )}
    </section>
  )
}

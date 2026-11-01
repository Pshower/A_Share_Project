import { useCallback, useEffect, useState } from 'react'
import { Database, RefreshCw } from 'lucide-react'
import { api, number } from './api'
import Chart from './Chart'

type Row = Record<string, any>
const preferred = ['pmi_manufacturing', 'cpi_yoy', 'ppi_yoy', 'm2_yoy', 'gdp_yoy',
  'industrial_production_yoy', 'cn_10y_yield', 'shibor_on', 'usdcny']

export default function MacroPanel({ fail }: { fail: (e: unknown) => void }) {
  const [snapshots, setSnapshots] = useState<Row[]>([])
  const [snapshotId, setSnapshotId] = useState('')
  const [catalog, setCatalog] = useState<Row[]>([])
  const [seriesId, setSeriesId] = useState('')
  const [rows, setRows] = useState<Row[]>([])
  const [loading, setLoading] = useState(false)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const values = await api<Row[]>('/market/snapshots/macro')
      setSnapshots(values)
      setSnapshotId((current) => current || values.find((v) => v.status === 'ready')?.id || values[0]?.id || '')
    } catch (e) {
      fail(e)
    } finally {
      setLoading(false)
    }
  }, [fail])
  useEffect(() => {
    load()
  }, [load])
  useEffect(() => {
    let live = true
    setCatalog([])
    setSeriesId('')
    setRows([])
    if (!snapshotId) return
    api<Row[]>(`/market/macro/${snapshotId}/catalog`)
      .then((value) => {
        if (!live) return
        setCatalog(value)
        const preferredId = preferred.find((id) => value.some((v) => v.series_id === id))
        setSeriesId(preferredId || value[0]?.series_id || '')
      })
      .catch(fail)
    return () => {
      live = false
    }
  }, [snapshotId, fail])
  useEffect(() => {
    let live = true
    setRows([])
    if (!snapshotId || !seriesId) return
    api(`/market/macro/${snapshotId}/series/${seriesId}`)
      .then((value) => {
        if (live) setRows(value.rows || [])
      })
      .catch(fail)
    return () => {
      live = false
    }
  }, [snapshotId, seriesId, fail])

  const selectedSnapshot = snapshots.find((v) => v.id === snapshotId)
  const selected = catalog.find((v) => v.series_id === seriesId)
  const latest = rows.at(-1)
  const digits = selected?.unit === 'percent' ? 2 : selected?.unit === 'index' ? 1 : 2
  return (
    <section className="section macro-panel">
      <div className="section-heading">
        <h2>宏观环境</h2>
        <span>只读快照 · 不自动进入模型</span>
      </div>
      <div className="toolbar">
        <label className="field">
          <span>宏观快照</span>
          <select aria-label="宏观快照" value={snapshotId} onChange={(e) => setSnapshotId(e.target.value)}>
            <option value="">选择快照</option>
            {snapshots.map((v) => (
              <option key={v.id} value={v.id}>
                {v.id} · {v.status} · {v.coverage?.rows || 0} 行
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          <span>指标</span>
          <select aria-label="宏观指标" value={seriesId} onChange={(e) => setSeriesId(e.target.value)}>
            <option value="">选择指标</option>
            {catalog.map((v) => (
              <option key={v.series_id} value={v.series_id}>
                {v.metric} · {v.series_id}
              </option>
            ))}
          </select>
        </label>
        <button type="button" className="icon-button" title="刷新快照列表" onClick={() => load()} disabled={loading}>
          <RefreshCw size={15} />
        </button>
      </div>
      {!snapshots.length && <div className="empty">暂无宏观快照，请先运行 src.data.macro</div>}
      {selectedSnapshot && (
        <div className="online-boundary">
          <Database size={16} />
          <span>{selectedSnapshot.id}</span>
          <span>{selectedSnapshot.status}</span>
          <span>覆盖 {selectedSnapshot.coverage?.first_period} 至 {selectedSnapshot.coverage?.last_period}</span>
          <b>{selectedSnapshot.coverage?.rows || 0} 行长表</b>
        </div>
      )}
      {selected && latest ? (
        <>
          <div className="stats-band macro-stats">
            <div className="stat">
              <span>{selected.metric}</span>
              <strong>{number(latest.value, digits)}</strong>
              <small>{selected.unit}</small>
            </div>
            <div className="stat">
              <span>统计期</span>
              <strong className="date-value">{latest.period}</strong>
              <small>{selected.frequency}</small>
            </div>
            <div className="stat">
              <span>保守可用日</span>
              <strong className="date-value">{latest.available_from}</strong>
              <small>期后固定滞后，不是官方发布日历</small>
            </div>
            <div className="stat">
              <span>来源</span>
              <strong className="date-value">{selected.source}</strong>
              <small>{selected.rows} 个观测</small>
            </div>
          </div>
          <Chart
            x={rows.map((r) => r.period)}
            series={[{ name: selected.metric, values: rows.map((r) => r.value) }]}
            height={300}
          />
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>统计期</th>
                  <th>可用日</th>
                  <th>数值</th>
                  <th>单位</th>
                  <th>来源</th>
                </tr>
              </thead>
              <tbody>
                {rows.slice(-12).reverse().map((r) => (
                  <tr key={r.period}>
                    <td>{r.period}</td>
                    <td>{r.available_from}</td>
                    <td>{number(r.value, digits)}</td>
                    <td>{r.unit}</td>
                    <td>{r.source}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="subtle-note">
            宏观数据按统计期和保守可用日展示；未做官方发布日历校验、因子降维或缺失/年龄掩码，不自动参与训练、计划或交易判断。
          </p>
        </>
      ) : (
        snapshots.length > 0 && <div className="empty">请选择宏观指标</div>
      )}
    </section>
  )
}

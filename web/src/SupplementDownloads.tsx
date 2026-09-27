import { useEffect, useState } from 'react'
import { Download, RefreshCw, ChevronLeft, ChevronRight } from 'lucide-react'
import { api, post, dateTime, number } from './api'

type Row = Record<string, any>
export default function SupplementDownloads({
  initialId,
  fail,
  openJob,
}: {
  initialId: string
  fail: (e: unknown) => void
  openJob: (job: Row) => void
}) {
  const [types, setTypes] = useState<Row[]>([])
  const [batches, setBatches] = useState<Row[]>([])
  const [kind, setKind] = useState('dividend')
  const [codes, setCodes] = useState('000001')
  const [market, setMarket] = useState('sz')
  const [start, setStart] = useState('')
  const [end, setEnd] = useState('')
  const [today, setToday] = useState('')
  const [period, setPeriod] = useState('5')
  const [consent, setConsent] = useState(false)
  const [retryConsent, setRetryConsent] = useState(false)
  const [busy, setBusy] = useState(false)
  const [id, setId] = useState(initialId)
  const [code, setCode] = useState('')
  const [page, setPage] = useState(0)
  const [table, setTable] = useState<Row | null>(null)
  const list = [...new Set(codes.split(/[\s,，;；]+/).filter(Boolean))]
  const valid = list.length > 0 && list.length <= 20 && list.every((c) => /^\d{6}$/.test(c))
  const type = types.find((t) => t.id === kind)
  const batch = batches.find((b) => b.id === id)
  const entry = batch?.entries.find((e: Row) => e.code === code)
  async function refresh() {
    const rows = await api('/market/snapshots/supplements')
    setBatches(rows)
    setId((old) => old || rows[0]?.id || '')
  }
  useEffect(() => {
    api('/market/supplement-types').then(setTypes).catch(fail)
    api('/market/status')
      .then((s) => {
        setToday(s.today)
        setEnd(s.today)
        setStart(
          new Date(Date.parse(s.today + 'T00:00:00Z') - 6 * 86400000).toISOString().slice(0, 10)
        )
      })
      .catch(fail)
    refresh().catch(fail)
  }, [fail])
  useEffect(() => {
    setCode(
      batch?.entries.find((e: Row) => e.status === 'ready')?.code || batch?.entries[0]?.code || ''
    )
    setPage(0)
    setRetryConsent(false)
  }, [id, batch])
  useEffect(() => {
    let live = true
    setTable(null)
    if (id && code && ['ready', 'empty'].includes(entry?.status))
      api(`/market/supplements/${id}/rows?code=${code}&page=${page}`)
        .then((r) => {
          if (live) setTable(r)
        })
        .catch((e) => {
          if (live) fail(e)
        })
    return () => {
      live = false
    }
  }, [id, code, page, entry?.status, fail])
  async function submit(previous?: Row) {
    setBusy(true)
    try {
      openJob(
        await post(
          '/market/supplements',
          previous
            ? { ...previous.request, allow_network: true, resume_id: previous.id }
            : {
                data_type: kind,
                codes: list,
                market,
                start_date: start,
                end_date: end,
                period,
                allow_network: consent,
              }
        )
      )
    } catch (e) {
      fail(e)
    } finally {
      setBusy(false)
    }
  }
  return (
    <>
      <div className="form-grid">
        <section className="section form-section">
          <div className="section-heading">
            <h2>扩展数据采集</h2>
            <span>独立归档 · 不自动进入训练</span>
          </div>
          <label className="field">
            数据类型
            <select
              aria-label="扩展数据类型"
              value={kind}
              onChange={(e) => {
                setKind(e.target.value)
                setConsent(false)
              }}
            >
              {types.map((t) => (
                <option key={t.id} value={t.id}>
                  {t.label}
                </option>
              ))}
            </select>
          </label>
          <p className="subtle-note">
            {type?.api} · {type?.provider === 'sina' ? '新浪财经' : '东方财富'}
          </p>
          <label className="field">
            {kind === 'index_daily' ? '指数代码' : '股票代码'}
            <textarea
              aria-label="扩展数据代码"
              rows={3}
              value={codes}
              onChange={(e) => setCodes(e.target.value)}
            />
          </label>
          <p className="subtle-note">
            {list.length} / 20 个代码 · {type?.note}
          </p>
          {kind === 'fund_flow' && (
            <label className="field">
              交易所
              <select
                aria-label="资金流交易所"
                value={market}
                onChange={(e) => setMarket(e.target.value)}
              >
                <option value="sz">深圳</option>
                <option value="sh">上海</option>
                <option value="bj">北京</option>
              </select>
            </label>
          )}
          {['minute', 'index_daily'].includes(kind) && (
            <div className="field-grid">
              <label className="field">
                开始日期
                <input
                  aria-label="扩展开始日期"
                  type="date"
                  value={start}
                  max={today}
                  onChange={(e) => setStart(e.target.value)}
                />
              </label>
              <label className="field">
                结束日期
                <input
                  aria-label="扩展结束日期"
                  type="date"
                  value={end}
                  max={today}
                  onChange={(e) => setEnd(e.target.value)}
                />
              </label>
            </div>
          )}
          {kind === 'minute' && (
            <label className="field">
              分钟周期
              <select
                aria-label="分钟周期"
                value={period}
                onChange={(e) => setPeriod(e.target.value)}
              >
                {['1', '5', '15', '30', '60'].map((p) => (
                  <option key={p} value={p}>
                    {p} 分钟 · 未复权
                  </option>
                ))}
              </select>
            </label>
          )}
          <label className="check-label">
            <input
              type="checkbox"
              checked={consent}
              onChange={(e) => setConsent(e.target.checked)}
            />
            允许本次从所示供应商下载扩展数据
          </label>
          <div className="form-actions">
            <button
              className="primary"
              disabled={busy || !consent || !valid || !start || !end}
              onClick={() => submit()}
            >
              <Download size={15} />
              下载扩展数据
            </button>
          </div>
        </section>
        <section className="section">
          <div className="section-heading">
            <h2>扩展数据归档</h2>
            <button
              className="icon-button"
              title="刷新扩展归档"
              onClick={() => refresh().catch(fail)}
            >
              <RefreshCw size={15} />
            </button>
          </div>
          <div className="table-wrap online-history">
            <table>
              <thead>
                <tr>
                  <th>快照</th>
                  <th>类型</th>
                  <th>状态</th>
                </tr>
              </thead>
              <tbody>
                {batches.map((b) => (
                  <tr key={b.id} className={b.id === id ? 'selected' : ''}>
                    <td>
                      <button className="row-link" onClick={() => setId(b.id)}>
                        {b.id}
                      </button>
                      <small className="subtext">{dateTime(b.created_at)}</small>
                    </td>
                    <td>
                      {types.find((t) => t.id === b.request.data_type)?.label ||
                        b.request.data_type}
                    </td>
                    <td>{b.status === 'ready' ? '已归档' : '部分失败'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {!batches.length && <div className="empty">尚无扩展数据快照</div>}
          {batch && (
            <>
              <p className="subtle-note">
                {batch.api} · {batch.provider} · AKShare {batch.akshare_version}
              </p>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>代码</th>
                      <th>记录</th>
                      <th>末条时间</th>
                      <th>状态</th>
                    </tr>
                  </thead>
                  <tbody>
                    {batch.entries.map((e: Row) => (
                      <tr key={e.code}>
                        <td>
                          <button
                            className="row-link"
                            onClick={() => {
                              setCode(e.code)
                              setPage(0)
                            }}
                          >
                            {e.code}
                          </button>
                        </td>
                        <td>{number(e.rows)}</td>
                        <td>{e.last_date || '未提供'}</td>
                        <td>
                          {e.status === 'ready'
                            ? '已保存'
                            : e.status === 'empty'
                              ? '空返回'
                              : '失败'}
                          {e.error && <small className="subtext">{e.error}</small>}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {batch.status === 'partial' && (
                <label className="check-label">
                  <input
                    type="checkbox"
                    checked={retryConsent}
                    onChange={(e) => setRetryConsent(e.target.checked)}
                  />
                  允许重试 {batch.provider} 的原请求
                </label>
              )}
              {batch.status === 'partial' && (
                <button disabled={busy || !retryConsent} onClick={() => submit(batch)}>
                  <RefreshCw size={15} />
                  重试失败项
                </button>
              )}
            </>
          )}
        </section>
      </div>
      {batch && (
        <section className="section">
          <div className="section-heading">
            <h2>已存表格 · {code}</h2>
            <span>供应商原始字段与单位 · 历史完整性未验证</span>
          </div>
          {table ? (
            <>
              <div className="toolbar">
                <a className="text-button" href={`/api/market/supplements/${id}/files/${code}/csv`}>
                  <Download size={15} />
                  CSV
                </a>
                <a
                  className="text-button"
                  href={`/api/market/supplements/${id}/files/${code}/json`}
                >
                  <Download size={15} />
                  JSON
                </a>
                <span>
                  {number(table.total)} 条 · 第 {page + 1} 页
                </span>
                <button
                  className="icon-button"
                  title="上一页"
                  disabled={!page}
                  onClick={() => setPage(page - 1)}
                >
                  <ChevronLeft size={16} />
                </button>
                <button
                  className="icon-button"
                  title="下一页"
                  disabled={(page + 1) * 50 >= table.total}
                  onClick={() => setPage(page + 1)}
                >
                  <ChevronRight size={16} />
                </button>
              </div>
              <div className="table-wrap supplement-table">
                <table>
                  <thead>
                    <tr>
                      {table.columns.map((c: string) => (
                        <th key={c}>{c}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {table.rows.map((r: any[], i: number) => (
                      <tr key={i}>
                        {r.map((v, j) => (
                          <td key={j}>
                            {v == null
                              ? '—'
                              : typeof v === 'object'
                                ? JSON.stringify(v)
                                : String(v)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {!table.total && (
                <div className="empty">供应商返回空表；不等于已确认没有相关事件</div>
              )}
            </>
          ) : (
            <div className="empty">没有可预览的成功数据</div>
          )}
        </section>
      )}
    </>
  )
}

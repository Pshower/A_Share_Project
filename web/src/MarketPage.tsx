import { useCallback, useEffect, useState } from 'react'
import {
  Download,
  RefreshCw,
  Play,
  Square,
  Radio,
  ArrowUpRight,
  Database,
  Layers,
} from 'lucide-react'
import { api, post, percent, number, dateTime } from './api'
import MarketChart from './MarketChart'

type Row = Record<string, any>
type Props = {
  models: Row[]
  datasets: Row[]
  fail: (e: unknown) => void
  openJob: (j: Row) => void
  initialModel: string
  initialKind: string
  initialId: string
}
const explanations: Record<string, string> = {
  unverified_vendor_time: '供方时间字段语义待核验',
  missing_quote: '缺少报价',
  unknown_quote_time: '报价时间未知',
  stale_quote: '报价已过期',
  future_timestamp: '报价时间异常',
  incomplete_snapshot: '快照不完整',
  stale_daily_plan: '日线计划过期',
  research_account_only: '仅研究账户',
  calendar_not_verified: '交易日历未核验',
  trading_state_not_verified: '交易限制未完整核验',
  no_daily_plan: '未绑定日线计划',
  at_vendor_limit_up: '达到供应商涨停参考价',
  at_vendor_limit_down: '达到供应商跌停参考价',
}
const intent: Record<string, string> = {
  buy: '买入参考',
  sell: '减仓参考',
  exit: '清仓参考',
  unchanged: '维持参考',
}
function codesFrom(text: string) {
  return text
    .split(/[\s,，]+/)
    .map((v) => v.trim())
    .filter(Boolean)
}
function weightsFrom(text: string) {
  const weights: Record<string, number> = {}
  for (const row of text.split(/[\n,，]+/).filter((v) => v.trim())) {
    const [code, value, extra] = row.trim().split(/\s*[:=]\s*/)
    if (
      !/^\d{6}$/.test(code) ||
      !value ||
      extra ||
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
function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="field">
      <span>{label}</span>
      {children}
    </div>
  )
}

export default function MarketPage(p: Props) {
  const [tab, setTab] = useState(
    p.initialKind === 'quotes' ? 'quotes' : p.initialKind === 'plans' ? 'plan' : 'history'
  )
  const [status, setStatus] = useState<Row>({ today: '', min_poll_seconds: 60 })
  const [histories, setHistories] = useState<Row[]>([])
  const [plans, setPlans] = useState<Row[]>([])
  const [quotes, setQuotes] = useState<Row[]>([])
  const [sessions, setSessions] = useState<Row[]>([])
  const [modelId, setModelId] = useState(
    p.initialModel || p.models.find((m) => m.trained)?.id || ''
  )
  const [requirements, setRequirements] = useState<Row | null>(null)
  const [codes, setCodes] = useState('000001')
  const [start, setStart] = useState('2017-01-01')
  const [end, setEnd] = useState('')
  const [basis, setBasis] = useState('both')
  const [consent, setConsent] = useState(false)
  const [researchConsent, setResearchConsent] = useState(false)
  const [revisionConsent, setRevisionConsent] = useState(false)
  const [weights, setWeights] = useState('')
  const [historyId, setHistoryId] = useState(p.initialKind === 'history' ? p.initialId : '')
  const [planId, setPlanId] = useState(p.initialKind === 'plans' ? p.initialId : '')
  const [quoteId, setQuoteId] = useState(p.initialKind === 'quotes' ? p.initialId : '')
  const [judgment, setJudgment] = useState<Row | null>(null)
  const [monitor, setMonitor] = useState<Row | null>(null)
  const [interval, setIntervalSeconds] = useState(60)
  const [duration, setDuration] = useState(300)
  const [busy, setBusy] = useState(false)
  const [build, setBuild] = useState(false)
  const [buildForm, setBuildForm] = useState({
    version: '',
    start_date: '2017-01-01',
    train_end: '2022-08-05',
    val_end: '2023-10-20',
    end_date: '2024-12-31',
  })
  const [template, setTemplate] = useState(
    p.datasets.find((d) => d.name === 'research_v3')?.id || p.datasets[0]?.id || ''
  )
  const load = useCallback(async () => {
    const [s, h, q, plans, monitors] = await Promise.all([
      api('/market/status'),
      api('/market/snapshots/history'),
      api('/market/snapshots/quotes'),
      api('/market/snapshots/plans'),
      api('/market/monitors'),
    ])
    setStatus(s)
    setHistories(h)
    setQuotes(q)
    setPlans(plans)
    setSessions(monitors)
    setEnd((e) => e || s.today)
    setHistoryId((v) => v || h[0]?.id || '')
    setQuoteId((v) => v || q[0]?.id || '')
  }, [])
  useEffect(() => {
    load().catch(p.fail)
  }, [load, p.fail])
  useEffect(() => {
    let live = true
    setRequirements(null)
    if (modelId)
      api('/market/model-requirements/' + modelId)
        .then((value) => {
          if (live) setRequirements(value)
        })
        .catch(p.fail)
    return () => {
      live = false
    }
  }, [modelId, p.fail])
  useEffect(() => {
    let live = true
    if (monitor?.status === 'active') return
    setJudgment(null)
    const refresh = () =>
      quoteId &&
      api(`/market/snapshots/quotes/${quoteId}${planId ? '?plan_id=' + planId : ''}`)
        .then((r) => {
          if (live) setJudgment(r.judgment)
        })
        .catch((error) => {
          if (live) {
            setJudgment(null)
            p.fail(error)
          }
        })
    refresh()
    const timer = window.setInterval(refresh, 10000)
    return () => {
      live = false
      clearInterval(timer)
    }
  }, [quoteId, planId, p.fail, monitor?.status])
  const monitorId = monitor?.id
  const monitoring = monitor?.status === 'active'
  useEffect(() => {
    if (!monitorId || !monitoring) return
    const poll = window.setInterval(
      () =>
        api('/market/monitors/' + monitorId)
          .then((m) => {
            setMonitor(m)
            setJudgment(m.judgment || null)
          })
          .catch((error) => {
            setJudgment(null)
            p.fail(error)
          }),
      3000
    )
    const heartbeat = window.setInterval(
      () => post('/market/monitors/' + monitorId + '/heartbeat', {}).catch(p.fail),
      15000
    )
    return () => {
      clearInterval(poll)
      clearInterval(heartbeat)
      api('/market/monitors/' + monitorId + '/stop', {
        method: 'POST',
        body: '{}',
        keepalive: true,
      }).catch(() => {})
    }
  }, [monitorId, monitoring, p.fail])

  const selectedHistory = histories.find((h) => h.id === historyId)
  const selectedPlan = plans.find((v) => v.id === planId)
  const pool = selectedPlan ? selectedPlan.stock_codes : codesFrom(codes)
  const covered =
    requirements &&
    selectedHistory?.status === 'ready' &&
    selectedHistory.request.bases.includes('hfq') &&
    requirements.codes.every((c: string) => selectedHistory.request.codes.includes(c))
  async function job(path: string, body: Row) {
    setBusy(true)
    try {
      p.openJob(await post(path, body))
    } catch (e) {
      p.fail(e)
    } finally {
      setBusy(false)
    }
  }
  async function beginMonitor() {
    setBusy(true)
    try {
      setMonitor(
        await post('/market/monitors', {
          codes: pool,
          allow_network: consent,
          plan_id: planId || null,
          interval_seconds: interval,
          duration_seconds: duration,
        })
      )
      setJudgment(null)
    } catch (e) {
      p.fail(e)
    } finally {
      setBusy(false)
    }
  }
  async function stopMonitor(id: string) {
    try {
      const record = await post('/market/monitors/' + id + '/stop', {})
      if (monitor?.id === id) setMonitor(record)
      await load()
    } catch (e) {
      p.fail(e)
    }
  }
  function plan() {
    try {
      job('/market/plans', {
        model_id: modelId,
        history_id: historyId,
        accept_revisions: revisionConsent,
        acknowledge_research: researchConsent,
        research_weights: weightsFrom(weights),
      })
    } catch (e) {
      p.fail(e)
    }
  }
  return (
    <>
      <div className="online-boundary">
        <Radio size={17} />
        <span>东方财富 · 日线研究与报价快照</span>
        <b>不下单 · 真实账户适配未启用</b>
      </div>
      <div className="tabs">
        {[
          ['history', '联网历史'],
          ['plan', '最新日线计划'],
          ['quotes', '报价与监控'],
        ].map(([id, text]) => (
          <button key={id} className={tab === id ? 'active' : ''} onClick={() => setTab(id)}>
            {text}
          </button>
        ))}
        <button
          className="icon-button"
          title="刷新快照列表（不下载行情）"
          onClick={() => load().catch(p.fail)}
        >
          <RefreshCw size={15} />
        </button>
      </div>
      {tab === 'history' && (
        <>
          <div className="form-grid">
            <section className="section form-section">
              <div className="section-heading">
                <h2>历史日线采集</h2>
                <span>新快照，不覆盖</span>
              </div>
              <Field label="股票代码">
                <textarea
                  aria-label="联网股票代码"
                  rows={3}
                  value={codes}
                  onChange={(e) => setCodes(e.target.value)}
                />
              </Field>
              <Field label="从模型提取完整股票池">
                <div className="input-action">
                  <select
                    aria-label="采集模型股票池"
                    value={modelId}
                    onChange={(e) => setModelId(e.target.value)}
                  >
                    {p.models
                      .filter((m) => m.trained)
                      .map((m) => (
                        <option key={m.id} value={m.id}>
                          {m.run} / {m.label} · {m.pool_size} 股
                        </option>
                      ))}
                  </select>
                  <button
                    className="icon-button"
                    title="采用模型股票池和历史锚点"
                    disabled={!requirements}
                    onClick={() => {
                      setCodes(requirements!.codes.join(' '))
                      setStart(requirements!.history_origin)
                    }}
                  >
                    <Layers size={16} />
                  </button>
                </div>
              </Field>
              <div className="field-grid">
                <Field label="起始日期">
                  <input
                    aria-label="联网起始日期"
                    type="date"
                    value={start}
                    onChange={(e) => setStart(e.target.value)}
                  />
                </Field>
                <Field label="结束日期">
                  <input
                    aria-label="联网结束日期"
                    type="date"
                    value={end}
                    max={status.today}
                    onChange={(e) => setEnd(e.target.value)}
                  />
                </Field>
              </div>
              <Field label="价格口径">
                <select
                  aria-label="下载价格口径"
                  value={basis}
                  onChange={(e) => setBasis(e.target.value)}
                >
                  <option value="both">后复权 + 未复权（分开保存）</option>
                  <option value="hfq">后复权 · 模型研究特征</option>
                  <option value="unadjusted">未复权 · 报价归档</option>
                </select>
              </Field>
              <label className="check-label">
                <input
                  type="checkbox"
                  checked={consent}
                  onChange={(e) => setConsent(e.target.checked)}
                />
                允许本次从东方财富联网采集
              </label>
              <p className="subtle-note">
                {codesFrom(codes).length} 只股票 · {basis === 'both' ? '2' : '1'}{' '}
                种口径；有限重试、全局限速，不切换 IP 或绕过限流。
              </p>
              <div className="form-actions">
                <button
                  className="primary"
                  disabled={busy || !consent || !codesFrom(codes).length}
                  onClick={() =>
                    job('/market/downloads', {
                      codes: codesFrom(codes),
                      start_date: start,
                      end_date: end,
                      bases: basis === 'both' ? ['hfq', 'unadjusted'] : [basis],
                      allow_network: consent,
                    })
                  }
                >
                  <Download size={16} />
                  开始联网下载
                </button>
              </div>
            </section>
            <section className="section">
              <div className="section-heading">
                <h2>已存下载批次</h2>
                <span>{histories.length} 个</span>
              </div>
              <div className="table-wrap online-history">
                <table>
                  <thead>
                    <tr>
                      <th>批次</th>
                      <th>股票</th>
                      <th>状态</th>
                      <th />
                    </tr>
                  </thead>
                  <tbody>
                    {histories.map((h) => (
                      <tr key={h.id} className={h.id === historyId ? 'selected' : ''}>
                        <td>
                          <button className="row-link" onClick={() => setHistoryId(h.id)}>
                            {h.id}
                          </button>
                          <small className="subtext">
                            {h.request.start_date} 至 {h.request.end_date}
                          </small>
                        </td>
                        <td>{h.request.codes.length}</td>
                        <td>{h.status === 'ready' ? '完整' : '部分失败'}</td>
                        <td>
                          {h.status === 'partial' && (
                            <button
                              className="icon-button"
                              title="按原请求重试失败项"
                              disabled={!consent || busy}
                              onClick={() =>
                                job('/market/downloads', {
                                  ...h.request,
                                  provider: undefined,
                                  allow_network: true,
                                  resume_id: h.id,
                                })
                              }
                            >
                              <RefreshCw size={15} />
                            </button>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {selectedHistory && (
                <>
                  <div className="section-heading secondary-heading">
                    <h2>逐股审计</h2>
                    <span>{selectedHistory.id}</span>
                  </div>
                  <div className="table-wrap online-history">
                    <table>
                      <thead>
                        <tr>
                          <th>代码 / 口径</th>
                          <th>实际末日</th>
                          <th>行数</th>
                          <th>状态</th>
                        </tr>
                      </thead>
                      <tbody>
                        {selectedHistory.entries.map((r: Row) => (
                          <tr key={r.code + r.basis}>
                            <td>
                              {r.code}
                              <small className="subtext">{r.basis}</small>
                            </td>
                            <td>{r.last_date || '—'}</td>
                            <td>{number(r.rows)}</td>
                            <td>
                              {r.status === 'ready'
                                ? r.reused_from
                                  ? '复用成功项'
                                  : '已保存'
                                : '失败'}
                              {r.error && <small className="subtext">{r.error}</small>}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <button
                    className="text-button"
                    disabled={
                      selectedHistory.status !== 'ready' ||
                      !selectedHistory.request.bases.includes('hfq')
                    }
                    onClick={() => setBuild(!build)}
                  >
                    <Database size={14} />
                    从所选批次构建训练版本
                  </button>
                </>
              )}
            </section>
          </div>
          {selectedHistory && <MarketChart key={selectedHistory.id} batch={selectedHistory as any} />}
          {build && selectedHistory && (
            <section className="section form-section">
              <div className="section-heading">
                <h2>新训练版本（会拟合新的训练期 scaler）</h2>
              </div>
              <div className="field-grid">
                <Field label="研究配置模板">
                  <select
                    aria-label="在线构建模板"
                    value={template}
                    onChange={(e) => setTemplate(e.target.value)}
                  >
                    {p.datasets.map((d) => (
                      <option key={d.id} value={d.id}>
                        {d.name}
                      </option>
                    ))}
                  </select>
                </Field>
                <Field label="版本标识">
                  <input
                    aria-label="在线新版本标识"
                    value={buildForm.version}
                    onChange={(e) => setBuildForm({ ...buildForm, version: e.target.value })}
                  />
                </Field>
                {(['start_date', 'train_end', 'val_end', 'end_date'] as const).map((key, i) => (
                  <Field key={key} label={['研究起始', '训练截止', '验证截止', '研究截止'][i]}>
                    <input
                      aria-label={'在线构建' + key}
                      type="date"
                      value={buildForm[key]}
                      onChange={(e) => setBuildForm({ ...buildForm, [key]: e.target.value })}
                    />
                  </Field>
                ))}
              </div>
              <button
                className="primary"
                disabled={busy || !buildForm.version}
                onClick={() =>
                  job('/market/build', {
                    ...buildForm,
                    dataset_id: template,
                    history_id: historyId,
                    codes: selectedHistory.request.codes,
                  })
                }
              >
                <Database size={16} />
                构建训练数据
              </button>
              <p className="subtle-note">
                这与旧模型的新数据推理不同。旧模型推理必须沿用原 scaler，不使用这里新拟合的参数。
              </p>
            </section>
          )}
        </>
      )}
      {tab === 'plan' && (
        <div className="form-grid">
          <section className="section form-section">
            <div className="section-heading">
              <h2>生成日线研究计划</h2>
            </div>
            <Field label="已存模型">
              <select
                aria-label="日线计划模型"
                value={modelId}
                onChange={(e) => setModelId(e.target.value)}
              >
                {p.models
                  .filter((m) => m.trained)
                  .map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.run} / {m.label} · {m.pool_size} 股
                    </option>
                  ))}
              </select>
            </Field>
            <Field label="历史批次">
              <select
                aria-label="日线历史批次"
                value={historyId}
                onChange={(e) => setHistoryId(e.target.value)}
              >
                <option value="">选择完整 HFQ 批次</option>
                {histories.map((h) => (
                  <option key={h.id} value={h.id}>
                    {h.id} · {h.request.codes.length} 股 · {h.status}
                  </option>
                ))}
              </select>
            </Field>
            <p className="subtle-note">
              完整股票池 {requirements?.codes.length || '—'} 只；历史锚点{' '}
              {requirements?.history_origin || '—'}。{!covered && '当前批次尚未完整覆盖模型池。'}
            </p>
            <Field label="假设研究仓位（比例，非真实股数）">
              <textarea
                aria-label="假设研究仓位"
                rows={3}
                placeholder="留空为全现金；例如 000001=0.10"
                value={weights}
                onChange={(e) => setWeights(e.target.value)}
              />
            </Field>
            <label className="check-label">
              <input
                type="checkbox"
                checked={researchConsent}
                onChange={(e) => setResearchConsent(e.target.checked)}
              />
              确认仅用于日线研究，不是盘中策略或实盘订单
            </label>
            <label className="check-label">
              <input
                type="checkbox"
                checked={revisionConsent}
                onChange={(e) => setRevisionConsent(e.target.checked)}
              />
              历史特征有修订时，明确接受为研究输入并记录差异
            </label>
            <div className="form-actions">
              <button
                className="primary"
                disabled={busy || !covered || !researchConsent}
                onClick={plan}
              >
                <Play size={15} />
                生成研究计划
              </button>
            </div>
          </section>
          <section className="section">
            <div className="section-heading">
              <h2>计划目录</h2>
              <span>{plans.length} 个</span>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>计划</th>
                    <th>特征截至</th>
                    <th>股票</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {plans.map((v) => (
                    <tr key={v.id}>
                      <td>
                        {v.id}
                        <small className="subtext">
                          {v.revisions_accepted ? '已接受历史修订' : '历史对齐通过'} · 仅研究
                        </small>
                      </td>
                      <td>{v.feature_as_of}</td>
                      <td>{v.stock_codes.length}</td>
                      <td>
                        <button
                          title="查看计划与报价"
                          className="icon-button"
                          onClick={() => {
                            setPlanId(v.id)
                            setTab('quotes')
                          }}
                        >
                          <ArrowUpRight size={16} />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </div>
      )}
      {tab === 'quotes' && (
        <>
          <section className="section form-section">
            <div className="section-heading">
              <h2>报价快照与有限监控</h2>
              <span>{monitor ? monitor.status : '未开启自动联网'}</span>
            </div>
            <div className="field-grid">
              <Field label="参考日线计划">
                <select
                  aria-label="行情参考计划"
                  disabled={!!monitoring}
                  value={planId}
                  onChange={(e) => setPlanId(e.target.value)}
                >
                  <option value="">只查看行情，不生成策略判断</option>
                  {plans.map((v) => (
                    <option key={v.id} value={v.id}>
                      {v.id} · {v.feature_as_of} · {v.stock_codes.length} 股
                    </option>
                  ))}
                </select>
              </Field>
              <Field label="报价股票">
                <input
                  aria-label="报价股票代码"
                  disabled={!!selectedPlan || !!monitoring}
                  value={pool.join(' ')}
                  onChange={(e) => setCodes(e.target.value)}
                />
              </Field>
              <Field label="刷新间隔（秒）">
                <input
                  aria-label="行情刷新间隔"
                  type="number"
                  min={60}
                  max={600}
                  disabled={!!monitoring}
                  value={interval}
                  onChange={(e) => setIntervalSeconds(Number(e.target.value))}
                />
              </Field>
              <Field label="会话时长（秒）">
                <input
                  aria-label="行情会话时长"
                  type="number"
                  min={60}
                  max={1800}
                  disabled={!!monitoring}
                  value={duration}
                  onChange={(e) => setDuration(Number(e.target.value))}
                />
              </Field>
            </div>
            <label className="check-label">
              <input
                type="checkbox"
                checked={consent}
                disabled={!!monitoring}
                onChange={(e) => setConsent(e.target.checked)}
              />
              允许从东方财富获取这组股票的报价
            </label>
            <div className="form-actions">
              <button
                disabled={busy || !consent || monitoring || !pool.length || pool.length > 50}
                onClick={() =>
                  job('/market/quotes', {
                    codes: pool,
                    allow_network: consent,
                    plan_id: planId || null,
                  })
                }
              >
                <RefreshCw size={15} />
                刷新一次报价
              </button>
              {monitoring ? (
                <button className="danger" onClick={() => stopMonitor(monitor!.id)}>
                  <Square size={14} />
                  停止监控
                </button>
              ) : (
                <button
                  className="primary"
                  disabled={busy || !consent || !pool.length || pool.length > 50}
                  onClick={beginMonitor}
                >
                  <Radio size={15} />
                  开启有限监控
                </button>
              )}
            </div>
            <p className="subtle-note">
              报价最多 50 股。离开页面会请求停止；断连超过 45
              秒租约失效。刷新价格不会重新训练或更换日线计划。
            </p>
            {sessions
              .filter((s) => s.status === 'active' && s.id !== monitorId)
              .map((s) => (
                <div key={s.id} className="online-boundary">
                  <span>已有会话 {s.id}，不会自动接管</span>
                  <button onClick={() => stopMonitor(s.id)}>停止已有会话</button>
                </div>
              ))}
          </section>
          <div className="toolbar">
            <Field label="已保存报价">
              <select
                aria-label="已保存报价快照"
                value={quoteId}
                disabled={!!monitoring}
                onChange={(e) => setQuoteId(e.target.value)}
              >
                <option value="">选择报价快照</option>
                {quotes.map((q) => (
                  <option key={q.id} value={q.id}>
                    {q.id} · {dateTime(q.created_at)} · {q.status}
                  </option>
                ))}
              </select>
            </Field>
          </div>
          {monitor && (
            <div className="online-boundary">
              <span>会话 {monitor.id}</span>
              <span>到期 {dateTime(new Date(monitor.expires * 1000).toISOString())}</span>
              <b>{monitor.last_job_record?.status || '等待首个请求'}</b>
              {monitor.last_job_record?.error && <span>{monitor.last_job_record.error}</span>}
            </div>
          )}
          {selectedPlan && (
            <section className="section">
              <div className="section-heading">
                <h2>日线研究目标 · {selectedPlan.feature_as_of}</h2>
                <span>目标现金 {percent(selectedPlan.target_cash)}</span>
              </div>
              <p className="subtle-note">
                固定的假设研究账户计划；没有独立交易日历确认，不能保证已补齐交易所最近一日。
              </p>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>股票</th>
                      <th>假设当前仓位</th>
                      <th>目标仓位</th>
                      <th>研究意图</th>
                    </tr>
                  </thead>
                  <tbody>
                    {selectedPlan.rows.map((r: Row) => (
                      <tr key={r.code}>
                        <td>{r.code}</td>
                        <td>{percent(r.current_research_weight)}</td>
                        <td>{percent(r.target_weight)}</td>
                        <td>{intent[r.research_intent]}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <details>
                <summary>兼容性与历史修订审计</summary>
                <pre>{JSON.stringify(selectedPlan.revision, null, 2)}</pre>
              </details>
            </section>
          )}
          {judgment ? (
            <section className="section">
              <div className="section-heading">
                <h2>当前报价检查</h2>
                <span>检查于 {dateTime(judgment.evaluated_at)}</span>
              </div>
              <div className="table-wrap market-quotes-table">
                <table>
                  <thead>
                    <tr>
                      <th>股票</th>
                      <th>未复权最新价</th>
                      <th>供方 f86 时间</th>
                      <th>接收时间</th>
                      <th>状态</th>
                      <th>研究目标</th>
                      <th>限制 / 等待原因</th>
                    </tr>
                  </thead>
                  <tbody>
                    {judgment.rows.map((r: Row) => (
                      <tr key={r.code}>
                        <td className="mono">{r.code}</td>
                        <td>{number(r.quote?.last, 3)}</td>
                        <td className="nowrap">
                          {r.quote?.provider_timestamp
                            ? dateTime(r.quote.provider_timestamp)
                            : '未知'}
                        </td>
                        <td className="nowrap">{r.quote ? dateTime(r.quote.received_at) : '—'}</td>
                        <td>
                          {r.status === 'waiting'
                            ? '等待 / 过期'
                            : r.status === 'research_review'
                              ? '仅研究复核'
                              : '仅行情'}
                        </td>
                        <td>{percent(r.target_weight)}</td>
                        <td>{r.reasons.map((s: string) => explanations[s] || s).join('；')}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="subtle-note">
                f86 按 Unix 秒解析，字段语义仍待进一步核验，不视为交易所认证成交时间。
                不会用真实股数乘以后复权价格估值；当前不输出可执行订单。
              </p>
            </section>
          ) : (
            <div className="empty">尚无可查看报价；不会用模拟值替代网络结果。</div>
          )}
        </>
      )}
    </>
  )
}

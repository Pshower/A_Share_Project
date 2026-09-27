import { useEffect, useMemo, useState } from 'react'
import { Play, Plus, X, Square, Check, ArrowUpRight } from 'lucide-react'
import MarketChart, { type ChartContext, type ModelSignal } from './MarketChart'
import { api, post, percent, number, statusName } from './api'
import { weightsFrom } from './researchWeights'
import { pollModels, terminal } from './modelPolling'

type Row = Record<string, any>
const sourceName = (s?: string) =>
  s === 'sina' ? '新浪' : s === 'eastmoney' ? '东方财富' : '来源未认证'
const intentName = (r?: Row) =>
  !r
    ? '—'
    : r.valid === false ? '数据不足，不参与配置'
    : r.target_weight === 0 && r.current_research_weight === 0
      ? '不配置'
      : (
          {
            buy: '买入/加仓意图',
            sell: '卖出/减仓意图',
            exit: '退出意图',
            unchanged: '目标基本不变',
          } as Row
        )[r.research_intent] || '—'

export default function MarketResearch({
  batch,
  datasetId,
  models,
  initialModel = '',
  apiReady,
  fail,
}: {
  batch: Row
  datasetId?: string
  models: Row[]
  initialModel?: string
  apiReady: boolean
  fail: (e: unknown) => void
}) {
  const [context, setContext] = useState<ChartContext | null>(null)
  const [modelIds, setModelIds] = useState<string[]>([initialModel])
  const [asOf, setAsOf] = useState('')
  const [accountMode, setAccountMode] = useState('')
  const [weights, setWeights] = useState('')
  const [consent, setConsent] = useState(false)
  const [revisions, setRevisions] = useState(false)
  const [profiles, setProfiles] = useState<Row[]>([])
  const [reports, setReports] = useState<Record<string, Row[]>>({})
  const [reportIds, setReportIds] = useState<Record<string, string>>({})
  const [runs, setRuns] = useState<Row[]>([])
  const [preflight, setPreflight] = useState<Row | null>(null)
  const [busy, setBusy] = useState(false)
  const active = runs.some((r) => r.job_id && !terminal.has(r.status))
  const pendingIds = runs
    .filter((r) => r.job_id && !terminal.has(r.status))
    .map((r) => r.job_id)
    .join(',')
  const ids = modelIds.filter(Boolean)
  const selection = ids.join(',')
  const historyId = datasetId ? batch.source_history_id : batch.id
  const provider = batch.request?.provider || batch.manifest?.data_source
  const trained = models.filter((m) => m.trained && m.valid)

  useEffect(() => {
    if (context) setAsOf((date) => date || context.lastDate)
  }, [context])
  useEffect(() => {
    let live = true
    setProfiles([])
    setReports({})
    setPreflight(null)
    setRuns([])
    Promise.all(
      ids.map(async (id) => {
        try {
          const [profile, saved] = await Promise.all([
            api('/market/model-requirements/' + id),
            api('/market/model-performance/' + id),
          ])
          return { profile, saved }
        } catch (e) {
          return { profile: { model_id: id, error: String(e) }, saved: [] }
        }
      })
    ).then((all) => {
      if (!live) return
      setProfiles(all.map((x) => x.profile))
      setReports(Object.fromEntries(all.map((x) => [x.profile.model_id, x.saved])))
      setReportIds(
        Object.fromEntries(
          all.map((x) => [
            x.profile.model_id,
            (x.saved.find((r: Row) => r.name === 'scenario_cash') || x.saved[0])?.id || '',
          ])
        )
      )
    })
    return () => {
      live = false
    }
  }, [selection])

  useEffect(() => {
    if (!active) return
    let live = true,
      polling = false
    async function poll() {
      if (polling) return
      polling = true
      try {
        const updates = await pollModels(runs, api)
        if (live)
          setRuns((current) => current.map((r) => updates.find((u) => u.job_id === r.job_id) || r))
      } catch (e) {
        if (live) fail(e)
      } finally {
        polling = false
      }
    }
    const timer = window.setInterval(poll, 2000)
    poll()
    return () => {
      live = false
      clearInterval(timer)
    }
  }, [pendingIds, fail])

  function clearResults() {
    setRuns([])
    setPreflight(null)
  }
  function request() {
    if (!context || !asOf || !accountMode || !historyId || !consent || !ids.length)
      throw new Error('请选择模型、收盘日期和明确的研究账户，并确认研究用途')
    return {
      model_ids: ids,
      history_id: historyId,
      close_as_of: asOf,
      display_code: context.code,
      account_mode: accountMode,
      research_weights: accountMode === 'weights' ? weightsFrom(weights) : {},
      accept_revisions: revisions,
      acknowledge_research: true,
    }
  }
  async function execute(checkOnly: boolean) {
    setBusy(true)
    try {
      const result = await post(
        '/market/model-comparison' + (checkOnly ? '/preflight' : ''),
        request()
      )
      setPreflight(result)
      if (!checkOnly) setRuns(result.results)
    } catch (e) {
      fail(e)
    } finally {
      setBusy(false)
    }
  }
  async function stop() {
    try {
      await Promise.all(
        runs
          .filter((r) => r.job_id && !terminal.has(r.status))
          .map((r) => post('/jobs/' + r.job_id + '/stop', {}))
      )
    } catch (e) {
      fail(e)
    }
  }
  const planKey = runs.map((r) => r.plan?.id || '').join(',')
  const signals = useMemo<ModelSignal[]>(
    () =>
      runs.flatMap((r) => {
        if (!r.plan || r.plan.feature_as_of !== asOf || !context) return []
        const row = r.plan.rows.find((x: Row) => x.code === context.code)
        if (!row) return []
        return [
          {
            modelId: r.model_id,
            label: `模型${modelIds.indexOf(r.model_id) + 1}`,
            date: r.plan.feature_as_of,
            targetWeight: row.target_weight,
            intent: intentName(row),
            direction: row.research_intent,
          },
        ]
      }),
    [planKey, context?.code, asOf, selection]
  )

  return (
    <>
      <section className="section model-research-controls">
        <div className="section-heading">
          <h2>模型对照</h2>
          <span>{sourceName(provider)} · 研究信号 · 不下单</span>
        </div>
        {!apiReady && (
          <div role="alert" className="error-banner">
            服务版本不支持多模型预测，请使用新版工作台。
          </div>
        )}
        {!historyId && (
          <div role="status" className="subtle-note">
            此数据版本没有已登记日线快照，暂不能生成新日线计划。
          </div>
        )}
        <div className="model-selectors">
          {modelIds.map((id, i) => (
            <div className="model-selector" key={i}>
              <label>
                模型 {i + 1}
                <select
                  aria-label={`对照模型${i + 1}`}
                  value={id}
                  disabled={busy || active}
                  onChange={(e) => {
                    setModelIds(modelIds.map((v, n) => (n === i ? e.target.value : v)))
                    clearResults()
                  }}
                >
                  <option value="">选择已保存模型</option>
                  {trained.map((m) => (
                    <option key={m.id} value={m.id} disabled={m.id !== id && ids.includes(m.id)}>
                      {m.run} / {m.label || m.name} ·{' '}
                      {sourceName(m.metadata?.contract?.data_source)} · {m.pool_size}股
                    </option>
                  ))}
                </select>
              </label>
              {i > 0 && (
                <button
                  className="icon-button"
                  aria-label={`移除模型${i + 1}`}
                  title="移除模型"
                  disabled={busy || active}
                  onClick={() => setModelIds(modelIds.filter((_, n) => n !== i))}
                >
                  <X size={16} />
                </button>
              )}
            </div>
          ))}
          <button
            className="icon-button"
            aria-label="添加对照模型"
            title="添加对照模型（最多3个）"
            disabled={modelIds.length >= 3 || busy || active}
            onClick={() => setModelIds([...modelIds, ''])}
          >
            <Plus size={17} />
          </button>
        </div>
        <div className="market-research-inputs">
          <label>
            收盘决策日
            <input
              type="date"
              aria-label="模型对照收盘日期"
              value={asOf}
              max={context?.lastDate}
              disabled={busy || active}
              onChange={(e) => {
                setAsOf(e.target.value)
                clearResults()
              }}
            />
          </label>
          <label>
            账户情景
            <select
              aria-label="模型对照账户"
              value={accountMode}
              disabled={busy || active}
              onChange={(e) => {
                setAccountMode(e.target.value)
                clearResults()
              }}
            >
              <option value="">未指定账户</option>
              <option value="cash">假设全现金</option>
              <option value="weights">假设持仓比例</option>
            </select>
          </label>
          {accountMode === 'weights' && (
            <label className="research-weights">
              持仓比例
              <textarea
                aria-label="模型对照持仓比例"
                value={weights}
                rows={2}
                placeholder="000001=0.10"
                disabled={busy || active}
                onChange={(e) => {
                  setWeights(e.target.value)
                  clearResults()
                }}
              />
            </label>
          )}
        </div>
        <div className="model-research-actions">
          <label className="check-label">
            <input
              type="checkbox"
              checked={consent}
              disabled={busy || active}
              onChange={(e) => setConsent(e.target.checked)}
            />
            确认假设账户与HFQ研究意图，非真实股数建议
          </label>
          <label className="check-label">
            <input
              type="checkbox"
              checked={revisions}
              disabled={busy || active}
              onChange={(e) => {
                setRevisions(e.target.checked)
                clearResults()
              }}
            />
            明确接受并记录历史特征修订
          </label>
          <button
            disabled={
              !apiReady ||
              busy ||
              active ||
              !consent ||
              !historyId ||
              !context ||
              !accountMode ||
              !ids.length ||
              !asOf
            }
            onClick={() => execute(true)}
          >
            <Check size={15} />
            {busy ? '正在检查输入' : '检查输入'}
          </button>
          <button
            className="primary"
            disabled={
              !apiReady ||
              busy ||
              active ||
              !consent ||
              !historyId ||
              !context ||
              !accountMode ||
              !ids.length ||
              !asOf
            }
            onClick={() => execute(false)}
          >
            <Play size={15} />
            生成多模型研究意图
          </button>
          {active && (
            <button className="danger" onClick={stop}>
              <Square size={15} />
              停止本组预测
            </button>
          )}
        </div>
      </section>
      <MarketChart
        batch={batch as any}
        datasetId={datasetId}
        onContext={setContext}
        signals={signals}
        selectedDate={asOf}
        onDateChange={(date) => {
          if (!active && !busy) {
            setAsOf(date)
            clearResults()
          }
        }}
      />
      <section className="section">
        <div className="section-heading">
          <h2>收盘配置对照</h2>
          <span>
            {context?.code || '—'} · {asOf || '—'} · 下一交易时段意图
          </span>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>模型</th>
                <th>数据与训练</th>
                <th>当前→目标</th>
                <th>研究方向</th>
                <th>状态</th>
              </tr>
            </thead>
            <tbody>
              {ids.map((id) => {
                const profile = profiles.find((p) => p.model_id === id)
                const run = runs.find((r) => r.model_id === id)
                const row = run?.plan?.rows.find((r: Row) => r.code === context?.code)
                const check = preflight?.models.find((r: Row) => r.model_id === id)
                const errors = run?.errors?.length
                  ? run.errors
                  : check?.errors || (profile?.error ? [profile.error] : [])
                return (
                  <tr key={id}>
                    <td>
                      模型 {modelIds.indexOf(id) + 1}
                      <small className="subtext">{models.find((m) => m.id === id)?.run}</small>
                    </td>
                    <td>
                      {sourceName(profile?.source)}
                      <small className="subtext">训练截至 {profile?.train_end || '—'}</small>
                      <small className="subtext">
                        {run?.historical_label ||
                          check?.historical_label ||
                          profile?.price_basis ||
                          '读取模型契约'}
                      </small>
                    </td>
                    <td>
                      {row
                        ? `${percent(row.current_research_weight)} → ${percent(row.target_weight)}`
                        : '—'}
                    </td>
                    <td>{intentName(row)}</td>
                    <td>
                      {run?.read_error ? '状态读取失败，正在重试' : run
                        ? statusName[run.status] ||
                          (run.status === 'blocked' ? '已阻断' : run.status)
                        : check
                          ? check.ready
                            ? '预检通过'
                            : '已阻断'
                          : '未生成'}
                      {errors.length > 0 && (
                        <div role="alert" className="prediction-errors">
                          {errors.slice(0, 8).join('；')}
                          {errors.length > 8 && `；另有${errors.length - 8}项`}
                        </div>
                      )}
                      {run?.job_id && (
                        <a href={'#jobs?id=' + run.job_id} className="text-button">
                          任务 <ArrowUpRight size={13} />
                        </a>
                      )}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
        {!ids.length && <div className="empty">尚未选择模型</div>}
        {runs
          .filter((r) => r.plan)
          .map((r) => (
            <details className="model-allocation" key={r.model_id}>
              <summary>
                模型 {modelIds.indexOf(r.model_id) + 1} 完整配置 · {r.plan.rows.length}股 · 现金{' '}
                {percent(r.plan.target_cash)} · {r.plan.feature_as_of}
              </summary>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>股票</th>
                      <th>当前权重</th>
                      <th>目标权重</th>
                      <th>变化</th>
                      <th>研究方向</th>
                    </tr>
                  </thead>
                  <tbody>
                    {r.plan.rows.map((row: Row) => (
                      <tr key={row.code}>
                        <td>{row.code}</td>
                        <td>{percent(row.current_research_weight)}</td>
                        <td>{percent(row.target_weight)}</td>
                        <td>{percent(row.target_weight - row.current_research_weight)}</td>
                        <td>{intentName(row)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
          ))}
      </section>
      <section className="section">
        <div className="section-heading">
          <h2>已存验证表现</h2>
          <span>报告各自账户口径 · 非当前单日预测收益</span>
        </div>
        {ids.map((id, i) => {
          const list = reports[id] || [],
            report = list.find((r) => r.id === reportIds[id])
          const dataset = report?.experiment?.research_config?.dataset
          return (
            <details className="model-performance" key={id}>
              <summary>
                模型 {i + 1} ·{' '}
                {list.length ? `${list.length}份匹配模型哈希的验证报告` : '无匹配验证报告'}
              </summary>
              {list.length > 0 && (
                <>
                  <label>
                    报告
                    <select
                      aria-label={`模型${i + 1}验证报告`}
                      value={reportIds[id] || ''}
                      onChange={(e) => setReportIds({ ...reportIds, [id]: e.target.value })}
                    >
                      {list.map((r) => (
                        <option key={r.id} value={r.id}>
                          {r.run} / {r.name}
                        </option>
                      ))}
                    </select>
                  </label>
                  <p className="subtle-note">
                    配置区间 {dataset?.train_end || '—'} → {dataset?.val_end || '—'} ·
                    初始账户以报告持仓记录为准
                  </p>
                  <div className="table-wrap">
                    <table>
                      <thead>
                        <tr>
                          <th>策略/基线</th>
                          <th>收益</th>
                          <th>Sharpe</th>
                          <th>最大回撤</th>
                          <th>平均现金</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(report?.metrics || {}).map(([name, value]) => {
                          const metric = value as Row
                          return (
                            <tr key={name}>
                              <td>{name}</td>
                              <td>{percent(metric.cumulative_return)}</td>
                              <td>{number(metric.sharpe, 2)}</td>
                              <td>{percent(metric.max_drawdown)}</td>
                              <td>{percent(metric.mean_cash_ratio)}</td>
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>
                  {report && (
                    <>
                      <img
                        src={`/api/artifacts/${report.id}/curves.png`}
                        alt={`模型${i + 1}验证净值、回撤与成本曲线`}
                        loading="lazy"
                      />
                      <a className="text-button" href={`/api/artifacts/${report.id}/report.md`}>
                        完整报告 <ArrowUpRight size={14} />
                      </a>
                    </>
                  )}
                </>
              )}
            </details>
          )
        })}
      </section>
    </>
  )
}

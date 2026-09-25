import {
  cloneElement,
  isValidElement,
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useState,
  type ComponentProps,
  type ReactElement,
  type ReactNode,
} from 'react'
import {
  Activity,
  ArrowUpRight,
  Box,
  Check,
  ChevronLeft,
  ChevronRight,
  Database,
  Download,
  FileText,
  FlaskConical,
  FolderOpen,
  ListTodo,
  LockKeyhole,
  Menu,
  Play,
  Plus,
  RefreshCw,
  Save,
  Search,
  SlidersHorizontal,
  Square,
  Target,
  X,
} from 'lucide-react'
import { api, post, percent, number, dateTime, statusName, kindName, phaseName } from './api'
const AsyncChart = lazy(() => import('./Chart'))
function Chart(props: ComponentProps<typeof AsyncChart>) {
  return (
    <Suspense
      fallback={
        <div className="empty" style={{ height: props.height || 270 }}>
          加载图表…
        </div>
      }
    >
      <AsyncChart {...props} />
    </Suspense>
  )
}

type Row = Record<string, any>
type Route = { view: string; query: URLSearchParams }
const views = [
  { id: 'training', name: '训练工作台', icon: Activity },
  { id: 'data', name: '数据管理', icon: Database },
  { id: 'models', name: '模型库', icon: Box },
  { id: 'backtest', name: '回测评估', icon: FlaskConical },
  { id: 'predict', name: '动作预测', icon: Target },
  { id: 'jobs', name: '任务记录', icon: ListTodo },
]
const strategyNames: Record<string, string> = {
  ppo: 'PPO',
  cash: '现金',
  buy_hold: '买入持有',
  equal_weight: '20 日等权',
  momentum: '20 日动量',
  equal_weight_daily: '每日等权',
  momentum_daily: '每日动量',
}
const terminal = new Set(['succeeded', 'failed', 'cancelled', 'interrupted'])
const parseRoute = (): Route => {
  const [path, query] = location.hash.slice(1).split('?')
  return { view: path || 'training', query: new URLSearchParams(query) }
}
function navigate(view: string, params: Record<string, string> = {}) {
  location.hash = view + (Object.keys(params).length ? '?' + new URLSearchParams(params) : '')
}
function Field({
  label,
  children,
  className = '',
}: {
  label: string
  children: ReactNode
  className?: string
}) {
  const control =
    isValidElement(children) && ['input', 'select', 'textarea'].includes(String(children.type))
      ? cloneElement(children as ReactElement<any>, { 'aria-label': label })
      : children
  return (
    <label className={'field ' + className}>
      <span>{label}</span>
      {control}
    </label>
  )
}
function Badge({ status }: { status: string }) {
  return <span className={'badge ' + status}>{statusName[status] || status}</span>
}
function Empty({ children }: { children: ReactNode }) {
  return (
    <div className="empty">
      <FolderOpen size={25} />
      <p>{children}</p>
    </div>
  )
}
function Stat({ label, value, detail }: { label: string; value: ReactNode; detail?: string }) {
  return (
    <div className="stat">
      <span>{label}</span>
      <strong>{value}</strong>
      {detail && <small>{detail}</small>}
    </div>
  )
}
function StockPicker({
  stocks,
  selected,
  onChange,
  preset = [],
}: {
  stocks: Row[]
  selected: string[]
  onChange: (v: string[]) => void
  preset?: string[]
}) {
  const [filter, setFilter] = useState('')
  const visible = stocks.filter((s) => s.code.includes(filter.trim()))
  return (
    <div className="stock-picker">
      <div className="picker-tools">
        <div className="search">
          <Search size={15} />
          <input
            aria-label="搜索股票代码"
            placeholder="搜索股票代码"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          />
        </div>
        <span className="mono">
          {selected.length} / {stocks.length}
        </span>
      </div>
      <div className="picker-actions">
        <button type="button" onClick={() => onChange(stocks.map((s) => s.code))}>
          全选
        </button>
        <button
          type="button"
          onClick={() => onChange(preset.filter((c) => stocks.some((s) => s.code === c)))}
        >
          轻量 32 只
        </button>
        <button type="button" onClick={() => onChange([])}>
          清空
        </button>
      </div>
      <div className="stock-grid">
        {visible.map((s) => (
          <label key={s.code} title={`有效行数 ${s.valid_rows ?? '—'}`}>
            <input
              type="checkbox"
              checked={selected.includes(s.code)}
              onChange={(e) =>
                onChange(
                  e.target.checked ? [...selected, s.code] : selected.filter((c) => c !== s.code)
                )
              }
            />
            <span>{s.code}</span>
          </label>
        ))}
      </div>
    </div>
  )
}

export default function App() {
  const [route, setRoute] = useState(parseRoute)
  const [system, setSystem] = useState<Row>({})
  const [datasets, setDatasets] = useState<Row[]>([])
  const [models, setModels] = useState<Row[]>([])
  const [reports, setReports] = useState<Row[]>([])
  const [jobs, setJobs] = useState<Row[]>([])
  const [defaults, setDefaults] = useState<Row>({ light_codes: [] })
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [online, setOnline] = useState(true)
  const [mobile, setMobile] = useState(false)
  const fail = useCallback((e: unknown) => setError(e instanceof Error ? e.message : String(e)), [])
  const refresh = useCallback(async () => {
    try {
      const [s, d, m, r, j, f] = await Promise.all([
        api('/system'),
        api('/datasets'),
        api('/models'),
        api('/reports'),
        api('/jobs'),
        api('/defaults'),
      ])
      setSystem(s)
      setDatasets(d)
      setModels(m)
      setReports(r)
      setJobs(j)
      setDefaults(f)
      setOnline(true)
    } catch (e) {
      setOnline(false)
      fail(e)
    } finally {
      setLoading(false)
    }
  }, [fail])
  useEffect(() => {
    refresh()
    const interval = setInterval(
      () =>
        api<Row[]>('/jobs')
          .then((j) => {
            setJobs(j)
            setOnline(true)
          })
          .catch(() => setOnline(false)),
      2500
    )
    return () => clearInterval(interval)
  }, [refresh])
  useEffect(() => {
    const change = () => {
      setRoute(parseRoute())
      setMobile(false)
      setError('')
    }
    addEventListener('hashchange', change)
    return () => removeEventListener('hashchange', change)
  }, [])
  const openJob = (job: Row) => {
    refresh()
    navigate('jobs', { job: job.id })
  }
  const active = jobs.filter((j) => !terminal.has(j.status)).length
  const current = views.find((v) => v.id === route.view) || views[0]
  const shared = { datasets, models, reports, jobs, fail, refresh, openJob, defaults }
  return (
    <div className="app">
      <aside className={'sidebar ' + (mobile ? 'open' : '')}>
        <a className="brand" href="#training">
          <span className="brand-mark">
            <Activity size={23} />
          </span>
          <span>
            A-SHARE <b>LAB</b>
            <small>量化研究工作台</small>
          </span>
        </a>
        <div className="nav-caption">研究空间</div>
        <nav>
          {views.map((v) => (
            <a
              key={v.id}
              href={'#' + v.id}
              className={route.view === v.id ? 'active' : ''}
              aria-current={route.view === v.id ? 'page' : undefined}
            >
              <v.icon size={18} />
              <span>{v.name}</span>
              {v.id === 'jobs' && active > 0 && <em>{active}</em>}
            </a>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <span className="environment-label">LOCAL ENVIRONMENT</span>
          <strong>
            Graduate <span>CPU</span>
          </strong>
          <small>本地数据 · HFQ 研究口径</small>
          <a href="#jobs">
            运行记录 <ArrowUpRight size={13} />
          </a>
        </div>
      </aside>
      {mobile && (
        <button className="scrim" aria-label="关闭导航" onClick={() => setMobile(false)} />
      )}
      <div className="workspace">
        <header className="topbar">
          <button
            className="icon-button menu-button"
            aria-label="打开导航"
            onClick={() => setMobile(!mobile)}
          >
            <Menu size={20} />
          </button>
          <div className="breadcrumb">
            研究工作区 <ChevronRight size={13} />
            <span>{current.name}</span>
          </div>
          <div className="topbar-right">
            <span className="connection">
              {online ? <Check size={13} /> : <X size={13} />}
              {loading ? '连接中' : online ? '本地服务' : '服务离线'}
            </span>
            <button
              className="icon-button"
              title="刷新资源"
              aria-label="刷新资源"
              onClick={refresh}
            >
              <RefreshCw size={16} />
            </button>
          </div>
        </header>
        <main>
          <div className="page-heading">
            <div>
              <div className="eyebrow">A-SHARE / RESEARCH</div>
              <h1>{current.name}</h1>
            </div>
            <div className="heading-actions">
              <span className="research-tag">HFQ · 非实盘</span>
              {route.view === 'training' && (
                <button className="primary" onClick={() => navigate('training', { new: '1' })}>
                  <Plus size={16} />
                  新建训练
                </button>
              )}
            </div>
          </div>
          {error && (
            <div className="error-banner" role="alert">
              <span>{error}</span>
              <button className="icon-button" aria-label="关闭错误" onClick={() => setError('')}>
                <X size={16} />
              </button>
            </div>
          )}
          {loading ? (
            <Empty>正在读取本地研究资源…</Empty>
          ) : (
            <>
              {route.view === 'training' &&
                (route.query.has('new') ? (
                  <TrainingForm {...shared} />
                ) : (
                  <Dashboard {...shared} active={active} />
                ))}
              {route.view === 'data' && <DataPage {...shared} />}
              {route.view === 'models' && (
                <ModelsPage {...shared} initial={route.query.get('model') || ''} />
              )}
              {route.view === 'backtest' && (
                <BacktestPage
                  {...shared}
                  initial={route.query.get('model') || ''}
                  reportId={route.query.get('report') || ''}
                />
              )}
              {route.view === 'predict' && (
                <PredictionPage
                  {...shared}
                  initial={route.query.get('model') || ''}
                  resultJob={route.query.get('result') || ''}
                />
              )}
              {route.view === 'jobs' &&
                (route.query.get('job') ? (
                  <JobPage {...shared} id={route.query.get('job')!} />
                ) : (
                  <section className="section">
                    <div className="section-heading">
                      <h2>全部任务</h2>
                      <span>{jobs.length} 条记录</span>
                    </div>
                    <JobsTable jobs={jobs} />
                  </section>
                ))}
            </>
          )}
          <footer>
            <span>研究结果不等于现实可成交收益</span>
            <span>
              PyTorch {system.packages?.torch || '—'} · PPO / SB3{' '}
              {system.packages?.['stable-baselines3'] || '—'}
            </span>
          </footer>
        </main>
      </div>
    </div>
  )
}

type Shared = {
  datasets: Row[]
  models: Row[]
  reports: Row[]
  jobs: Row[]
  defaults: Row
  fail: (e: unknown) => void
  refresh: () => Promise<void>
  openJob: (j: Row) => void
}
function JobsTable({ jobs }: { jobs: Row[] }) {
  if (!jobs.length) return <Empty>还没有工作台任务</Empty>
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>任务</th>
            <th>类型</th>
            <th>状态</th>
            <th>创建时间</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {jobs.map((j) => (
            <tr key={j.id}>
              <td>
                <strong>{j.payload.name || kindName[j.kind]}</strong>
                <small className="mono subtext">{j.id}</small>
              </td>
              <td>{kindName[j.kind]}</td>
              <td>
                <Badge status={j.status} />
              </td>
              <td className="nowrap muted">{dateTime(j.created)}</td>
              <td>
                <button
                  className="icon-button"
                  title="打开任务"
                  onClick={() => navigate('jobs', { job: j.id })}
                >
                  <ArrowUpRight size={16} />
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
function Dashboard(p: Shared & { active: number }) {
  const preferred =
    p.reports.find((r) => r.name === 'final_test' && r.run === 'train_8192_20260925') ||
    p.reports[0]
  const [report, setReport] = useState<Row | null>(null)
  useEffect(() => {
    if (preferred)
      api('/reports/' + preferred.id)
        .then(setReport)
        .catch(p.fail)
  }, [preferred?.id, p.fail])
  const primary = report?.metrics.ppo
  const series = report
    ? Object.entries(report.series)
        .filter(([k]) => ['ppo', 'equal_weight', 'cash'].includes(k))
        .map(([name, rows]) => ({
          name: strategyNames[name],
          values: (rows as Row[]).map((r) => r.net_value),
        }))
    : []
  const dates = report
    ? (report.series.ppo || Object.values(report.series)[0] || []).map((r: Row) =>
        r.date.slice(0, 10)
      )
    : []
  return (
    <>
      <div className="stats-band">
        <Stat label="数据版本" value={p.datasets.length} detail="本地已构建" />
        <Stat
          label="已训练模型"
          value={p.models.filter((m) => m.trained && m.valid).length}
          detail="含候选 checkpoint"
        />
        <Stat label="活动任务" value={p.active} detail="单重任务执行槽" />
        <Stat
          label="最新研究收益"
          value={percent(primary?.cumulative_return)}
          detail={report ? '32 股历史实验 · 测试区间已查看' : '暂无报告'}
        />
      </div>
      <div className="dashboard-grid">
        <section className="section">
          <div className="section-heading">
            <h2>近期任务</h2>
            <button className="text-button" onClick={() => navigate('jobs')}>
              全部记录 <ChevronRight size={14} />
            </button>
          </div>
          <JobsTable jobs={p.jobs.slice(0, 5)} />
          <div className="section-heading secondary-heading">
            <h2>最近保存的模型</h2>
            <button className="text-button" onClick={() => navigate('models')}>
              模型库 <ChevronRight size={14} />
            </button>
          </div>
          <div className="recent-models">
            {p.models
              .filter((m) => m.trained)
              .slice(0, 3)
              .map((m) => (
                <button
                  key={m.id}
                  className="recent-model"
                  onClick={() => navigate('models', { model: m.id })}
                >
                  <Box size={18} />
                  <span>
                    <strong>{m.label}</strong>
                    <small>
                      {m.run} · {m.pool_size} 只股票
                    </small>
                  </span>
                  <span className="mono">
                    {number(m.timesteps)}
                    <small>steps</small>
                  </span>
                  <ChevronRight size={16} />
                </button>
              ))}
          </div>
        </section>
        <section className="section chart-section">
          <div className="section-heading">
            <h2>已存实验 · 净值</h2>
            {report && (
              <button
                className="icon-button"
                title="打开报告"
                onClick={() => navigate('backtest', { report: report.id })}
              >
                <ArrowUpRight size={16} />
              </button>
            )}
          </div>
          {report ? (
            <>
              <div className="chart-summary">
                <div>
                  <strong>{percent(primary?.cumulative_return)}</strong>
                  <span>累计收益</span>
                </div>
                <div>
                  <b>{number(primary?.sharpe, 3)}</b>
                  <span>Sharpe</span>
                </div>
                <div>
                  <b>{percent(primary?.max_drawdown)}</b>
                  <span>最大回撤</span>
                </div>
              </div>
              <Chart series={series} x={dates} height={280} />
              <div className="subtle-note">
                {report.split === 'test' ? '最终测试' : '验证集'} · {report.run}
              </div>
            </>
          ) : (
            <Empty>尚无回测报告</Empty>
          )}
        </section>
      </div>
    </>
  )
}

function TrainingForm(p: Shared) {
  const [dataset, setDataset] = useState(
    p.datasets.find((d) => d.name === 'research_v3')?.id || p.datasets[0]?.id || ''
  )
  const [detail, setDetail] = useState<Row | null>(null)
  const [codes, setCodes] = useState<string[]>([])
  const [form, setForm] = useState({
    name: 'PPO 研究实验',
    total_timesteps: 8192,
    seed: 42,
    lookback: 1,
    n_steps: 256,
    batch_size: 64,
    n_epochs: 10,
    learning_rate: 0.0003,
    validation_rollouts: 10,
  })
  const [busy, setBusy] = useState(false)
  useEffect(() => {
    let live = true
    if (dataset)
      api('/datasets/' + dataset)
        .then((d) => {
          if (live) {
            setDetail(d)
            setCodes(
              p.defaults.light_codes.filter((c: string) => d.quality.some((s: Row) => s.code === c))
            )
          }
        })
        .catch(p.fail)
    return () => {
      live = false
    }
  }, [dataset, p.defaults, p.fail])
  const field = (key: keyof typeof form, value: string) =>
    setForm((prev) => ({ ...prev, [key]: key === 'name' ? value : Number(value) }))
  async function submit(check: boolean) {
    setBusy(true)
    try {
      p.openJob(
        await post(check ? '/training/validate' : '/training/jobs', {
          ...form,
          dataset_id: dataset,
          codes,
          hidden_sizes: [128, 64],
        })
      )
    } catch (e) {
      p.fail(e)
    } finally {
      setBusy(false)
    }
  }
  return (
    <>
      <button className="text-button back" onClick={() => navigate('training')}>
        <ChevronLeft size={15} />
        返回实验列表
      </button>
      <div className="form-grid">
        <section className="section form-section">
          <div className="section-heading">
            <h2>数据与股票池</h2>
            <span>01</span>
          </div>
          <Field label="数据版本">
            <select value={dataset} onChange={(e) => setDataset(e.target.value)}>
              {p.datasets.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name} · {d.stocks} 只股票
                </option>
              ))}
            </select>
          </Field>
          {detail && (
            <>
              <div className="date-strip">
                <div>
                  <span>训练截止</span>
                  <b>{detail.train_end}</b>
                </div>
                <div>
                  <span>验证截止</span>
                  <b>{detail.val_end}</b>
                </div>
                <div>
                  <span>数据截止</span>
                  <b>{detail.end_date}</b>
                </div>
              </div>
              <StockPicker
                stocks={detail.quality}
                selected={codes}
                onChange={setCodes}
                preset={p.defaults.light_codes}
              />
              <p className="subtle-note">
                scaler 拟合池：{detail.manifest.build_config.train_codes.length} 只 · 训练池：
                {codes.length} 只
              </p>
            </>
          )}
        </section>
        <section className="section form-section">
          <div className="section-heading">
            <h2>训练设置</h2>
            <span>02</span>
          </div>
          <Field label="实验名称">
            <input
              value={form.name}
              onChange={(e) => field('name', e.target.value)}
              maxLength={80}
            />
          </Field>
          <div className="field-grid">
            <Field label="总训练步数">
              <input
                type="number"
                min={8}
                max={1000000}
                value={form.total_timesteps}
                onChange={(e) => field('total_timesteps', e.target.value)}
              />
            </Field>
            <Field label="随机种子">
              <input
                type="number"
                min={0}
                value={form.seed}
                onChange={(e) => field('seed', e.target.value)}
              />
            </Field>
            <Field label="Rollout 步数">
              <input
                type="number"
                value={form.n_steps}
                onChange={(e) => field('n_steps', e.target.value)}
              />
            </Field>
            <Field label="Batch 大小">
              <input
                type="number"
                value={form.batch_size}
                onChange={(e) => field('batch_size', e.target.value)}
              />
            </Field>
          </div>
          <details open>
            <summary>
              <SlidersHorizontal size={15} />
              高级参数
            </summary>
            <div className="field-grid">
              <Field label="学习率">
                <input
                  type="number"
                  step="0.0001"
                  value={form.learning_rate}
                  onChange={(e) => field('learning_rate', e.target.value)}
                />
              </Field>
              <Field label="每轮 Epochs">
                <input
                  type="number"
                  value={form.n_epochs}
                  onChange={(e) => field('n_epochs', e.target.value)}
                />
              </Field>
              <Field label="观测窗口">
                <input
                  type="number"
                  min={1}
                  max={60}
                  value={form.lookback}
                  onChange={(e) => field('lookback', e.target.value)}
                />
              </Field>
              <Field label="验证间隔 / Rollout">
                <input
                  type="number"
                  min={1}
                  value={form.validation_rollouts}
                  onChange={(e) => field('validation_rollouts', e.target.value)}
                />
              </Field>
            </div>
          </details>
          <div className="definition-list">
            <span>网络</span>
            <b>共享 MLP · 128 → 64</b>
            <span>运行环境</span>
            <b>Graduate · CPU · 单进程</b>
            <span>执行预算</span>
            <b>{number(Math.ceil(form.total_timesteps / form.n_steps) * form.n_steps)} steps</b>
            <span>保存位置</span>
            <b>reports/runs/workbench/</b>
          </div>
          <p className="subtle-note">
            沿用研究版本的资金与费用配置；只训练所选股票，不重新拟合 scaler。
          </p>
          <div className="form-actions">
            <button disabled={busy || !codes.length} onClick={() => submit(true)}>
              <Check size={16} />
              检查配置
            </button>
            <button
              className="primary"
              disabled={busy || !codes.length}
              onClick={() => submit(false)}
            >
              <Play size={15} />
              {busy ? '提交中…' : '开始训练'}
            </button>
          </div>
        </section>
      </div>
    </>
  )
}

function DataPage(p: Shared) {
  const [id, setId] = useState(
    p.datasets.find((d) => d.name === 'research_v3')?.id || p.datasets[0]?.id || ''
  )
  const [detail, setDetail] = useState<Row | null>(null)
  const [build, setBuild] = useState(false)
  const [codes, setCodes] = useState<string[]>([])
  const [form, setForm] = useState({
    version: '',
    start_date: '2017-01-01',
    train_end: '2022-08-05',
    val_end: '2023-10-20',
    end_date: '2024-12-31',
  })
  const [busy, setBusy] = useState(false)
  useEffect(() => {
    let live = true
    if (id)
      api('/datasets/' + id)
        .then((d) => {
          if (live) {
            setDetail(d)
            setCodes(
              p.defaults.light_codes.filter((c: string) => d.quality.some((s: Row) => s.code === c))
            )
            setForm((f) => ({
              ...f,
              ...Object.fromEntries(
                ['start_date', 'train_end', 'val_end', 'end_date'].map((k) => [
                  k,
                  d.manifest.build_config[k],
                ])
              ),
            }))
          }
        })
        .catch(p.fail)
    return () => {
      live = false
    }
  }, [id, p.defaults, p.fail])
  async function startBuild() {
    setBusy(true)
    try {
      p.openJob(await post('/datasets/build', { ...form, dataset_id: id, codes }))
    } catch (e) {
      p.fail(e)
    } finally {
      setBusy(false)
    }
  }
  return (
    <>
      <div className="toolbar">
        <Field label="数据版本">
          <select value={id} onChange={(e) => setId(e.target.value)}>
            {p.datasets.map((d) => (
              <option key={d.id} value={d.id}>
                {d.name}
              </option>
            ))}
          </select>
        </Field>
        <button onClick={() => setBuild(!build)}>
          <Plus size={16} />
          {build ? '关闭构建表单' : '构建新版本'}
        </button>
      </div>
      {detail && (
        <>
          <div className="stats-band">
            <Stat label="股票数" value={detail.stocks} />
            <Stat label="交易日" value={number(detail.n_dates)} />
            <Stat label="特征数" value={detail.features} />
            <Stat label="数据截止" value={<span className="date-value">{detail.end_date}</span>} />
          </div>
          {build ? (
            <div className="form-grid">
              <section className="section form-section">
                <div className="section-heading">
                  <h2>新版本范围</h2>
                  <span>本地原始文件</span>
                </div>
                <Field label="新版本标识">
                  <input
                    placeholder="research_custom_01"
                    value={form.version}
                    onChange={(e) => setForm({ ...form, version: e.target.value })}
                  />
                </Field>
                <div className="field-grid">
                  {(['start_date', 'train_end', 'val_end', 'end_date'] as const).map((k, i) => (
                    <Field key={k} label={['研究起始', '训练截止', '验证截止', '研究截止'][i]}>
                      <input
                        type="date"
                        value={form[k]}
                        onChange={(e) => setForm({ ...form, [k]: e.target.value })}
                      />
                    </Field>
                  ))}
                </div>
                <p className="subtle-note">
                  新版本只保留勾选股票，并在所选训练日期拟合
                  scaler。现有版本不会被覆盖，不自动下载。
                </p>
                <div className="form-actions">
                  <button
                    className="primary"
                    disabled={busy || !codes.length || !form.version}
                    onClick={startBuild}
                  >
                    <Database size={16} />
                    {busy ? '提交中…' : '开始构建'}
                  </button>
                </div>
              </section>
              <section className="section form-section">
                <div className="section-heading">
                  <h2>构建 / 拟合股票池</h2>
                </div>
                <StockPicker
                  stocks={detail.quality}
                  selected={codes}
                  onChange={setCodes}
                  preset={p.defaults.light_codes}
                />
              </section>
            </div>
          ) : (
            <section className="section">
              <div className="section-heading">
                <h2>股票数据质量</h2>
                <span>
                  {detail.start_date} 至 {detail.end_date}
                </span>
              </div>
              <div className="table-wrap quality-table">
                <table>
                  <thead>
                    <tr>
                      <th>股票代码</th>
                      <th>有效特征行</th>
                      <th>特征缺失比例</th>
                      <th>缺少源行情天数</th>
                    </tr>
                  </thead>
                  <tbody>
                    {detail.quality.map((s: Row) => (
                      <tr key={s.code}>
                        <td className="mono">{s.code}</td>
                        <td>{number(s.valid_rows)}</td>
                        <td>{percent(s.missing_feature_fraction)}</td>
                        <td>{number(s.missing_source_dates)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}
        </>
      )}
    </>
  )
}

function ModelsPage(p: Shared & { initial: string }) {
  const [selected, setSelected] = useState(p.initial || p.models.find((m) => m.trained)?.id || '')
  const [filter, setFilter] = useState('')
  const [label, setLabel] = useState('')
  const [transfer, setTransfer] = useState(false)
  const [codes, setCodes] = useState<string[]>([])
  const [stocks, setStocks] = useState<Row[]>([])
  const [busy, setBusy] = useState(false)
  const model = p.models.find((m) => m.id === selected)
  useEffect(() => {
    if (p.initial) setSelected(p.initial)
  }, [p.initial])
  useEffect(() => {
    setLabel(model?.label || '')
    setCodes(model?.metadata.contract.stock_codes || [])
    const dataset = p.datasets.find((d) => d.version === model?.dataset)
    if (dataset)
      api('/datasets/' + dataset.id)
        .then((d) => setStocks(d.quality))
        .catch(p.fail)
  }, [selected, model?.label, p.datasets, p.fail])
  async function saveLabel() {
    try {
      await api('/models/' + selected + '/labels', {
        method: 'PATCH',
        body: JSON.stringify({ label }),
      })
      await p.refresh()
    } catch (e) {
      p.fail(e)
    }
  }
  async function migrate() {
    setBusy(true)
    try {
      p.openJob(await post('/models/' + selected + '/transfer', { model_id: selected, codes }))
    } catch (e) {
      p.fail(e)
    } finally {
      setBusy(false)
    }
  }
  return (
    <div className="master-detail">
      <section className="section">
        <div className="section-heading">
          <h2>模型目录</h2>
          <span>{p.models.length} 个 checkpoint</span>
        </div>
        <div className="search table-search">
          <Search size={16} />
          <input
            aria-label="搜索模型"
            placeholder="搜索模型或实验"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          />
        </div>
        <div className="table-wrap model-table">
          <table>
            <thead>
              <tr>
                <th>模型 / 来源</th>
                <th>步数</th>
                <th>股票</th>
                <th>状态</th>
              </tr>
            </thead>
            <tbody>
              {p.models
                .filter((m) => (m.label + m.run).toLowerCase().includes(filter.toLowerCase()))
                .map((m) => (
                  <tr key={m.id} className={selected === m.id ? 'selected' : ''}>
                    <td>
                      <button
                        className="row-link"
                        onClick={() => {
                          setSelected(m.id)
                          setTransfer(false)
                        }}
                      >
                        {m.label}
                      </button>
                      <small className="subtext">{m.run}</small>
                    </td>
                    <td className="mono">{number(m.timesteps)}</td>
                    <td>{m.pool_size}</td>
                    <td>
                      <span className={'badge ' + (m.valid && m.trained ? 'succeeded' : 'queued')}>
                        {!m.valid
                          ? '损坏'
                          : m.trained
                            ? m.timesteps === 0
                              ? '迁移模型'
                              : '已训练'
                            : '初始化'}
                      </span>
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </section>
      <section className="section model-detail">
        {model ? (
          <>
            <div className="section-heading">
              <h2>模型详情</h2>
              <Box size={19} />
            </div>
            <Field label="显示名称">
              <div className="input-action">
                <input value={label} onChange={(e) => setLabel(e.target.value)} maxLength={80} />
                <button
                  className="icon-button"
                  title="保存名称"
                  aria-label="保存名称"
                  onClick={saveLabel}
                >
                  <Save size={17} />
                </button>
              </div>
            </Field>
            <div className="definition-list">
              <span>来源实验</span>
              <b className="break">{model.run}</b>
              <span>数据版本</span>
              <b>{model.dataset}</b>
              <span>股票池</span>
              <b>{model.pool_size} 只</b>
              <span>特征 / 窗口</span>
              <b>
                {model.metadata.contract.feature_cols.length} / {model.metadata.contract.lookback}
              </b>
              <span>价格口径</span>
              <b>{model.price_basis}</b>
              <span>模型文件</span>
              <b>{model.valid ? '哈希一致' : '校验失败'}</b>
              <span>保存时间</span>
              <b>{dateTime(model.modified)}</b>
              {model.metadata.provenance?.optimizer_reset && (
                <>
                  <span>迁移来源步数</span>
                  <b>{number(model.metadata.provenance.source_num_timesteps)}</b>
                  <span>优化器</span>
                  <b>已重置，未继续训练</b>
                </>
              )}
            </div>
            <div className="model-actions">
              <button
                className="primary"
                disabled={!model.valid || !model.trained}
                onClick={() => navigate('backtest', { model: model.id })}
              >
                <FlaskConical size={16} />
                回测
              </button>
              <button
                disabled={!model.valid || !model.trained}
                onClick={() => navigate('predict', { model: model.id })}
              >
                <Target size={16} />
                预测动作
              </button>
            </div>
            <button
              className="text-button"
              disabled={!model.trained}
              onClick={() => setTransfer(!transfer)}
            >
              显式迁移股票池 <ArrowUpRight size={14} />
            </button>
            {transfer && (
              <div className="transfer-form">
                <StockPicker
                  stocks={stocks}
                  selected={codes}
                  onChange={setCodes}
                  preset={p.defaults.light_codes}
                />
                <p className="subtle-note">
                  新建推理实例并复用共享参数；优化器重置，不保证迁移后的效果。
                </p>
                <button disabled={busy || !codes.length} onClick={migrate}>
                  <Plus size={15} />
                  创建迁移模型
                </button>
              </div>
            )}
            <details>
              <summary>数据契约与参数</summary>
              <pre>{JSON.stringify(model.metadata.contract, null, 2)}</pre>
            </details>
          </>
        ) : (
          <Empty>选择一个模型</Empty>
        )}
      </section>
    </div>
  )
}

function BacktestPage(p: Shared & { initial: string; reportId: string }) {
  const eligible = p.models.filter((m) => m.trained && m.valid)
  const [modelId, setModelId] = useState(p.initial || eligible[0]?.id || '')
  const [split, setSplit] = useState('val')
  const [ack, setAck] = useState(false)
  const [frozen, setFrozen] = useState('')
  const [reportId, setReportId] = useState(p.reportId || p.reports[0]?.id || '')
  const [busy, setBusy] = useState(false)
  const freezes = p.jobs.filter(
    (j) => j.kind === 'freeze' && j.status === 'succeeded' && j.payload.model_id === modelId
  )
  useEffect(() => {
    if (p.reportId) setReportId(p.reportId)
  }, [p.reportId])
  useEffect(() => {
    if (p.initial) setModelId(p.initial)
  }, [p.initial])
  async function run(freeze = false) {
    setBusy(true)
    try {
      p.openJob(
        await post(
          freeze ? '/freezes' : '/evaluations',
          freeze
            ? { model_id: modelId }
            : {
                model_id: modelId,
                split,
                frozen_job_id: split === 'test' ? frozen : null,
                acknowledge_test: ack,
              }
        )
      )
    } catch (e) {
      p.fail(e)
    } finally {
      setBusy(false)
    }
  }
  return (
    <>
      <section className="section form-section">
        <div className="section-heading">
          <h2>新建回测</h2>
          <span>相同股票池与成本口径</span>
        </div>
        <div className="evaluation-controls">
          <Field label="已存模型">
            <select
              value={modelId}
              onChange={(e) => {
                setModelId(e.target.value)
                setFrozen('')
              }}
            >
              <option value="">选择模型</option>
              {eligible.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.run} / {m.label}
                </option>
              ))}
            </select>
          </Field>
          <Field label="评价区间">
            <select value={split} onChange={(e) => setSplit(e.target.value)}>
              <option value="val">验证集</option>
              <option value="test">最终测试 / 已查看区间</option>
            </select>
          </Field>
          <button
            className="primary"
            disabled={busy || !modelId || (split === 'test' && (!ack || !frozen))}
            onClick={() => run()}
          >
            <Play size={16} />
            开始回测
          </button>
        </div>
        {split === 'test' && (
          <div className="test-guard">
            <div className="evaluation-controls">
              <Field label="冻结记录">
                <select value={frozen} onChange={(e) => setFrozen(e.target.value)}>
                  <option value="">选择本模型的成功冻结记录</option>
                  {freezes.map((f) => (
                    <option value={f.id} key={f.id}>
                      {f.id} · {dateTime(f.created)}
                    </option>
                  ))}
                </select>
              </Field>
              <button disabled={busy || !modelId} onClick={() => run(true)}>
                <LockKeyhole size={15} />
                冻结验证所选模型
              </button>
            </div>
            <label className="check-label">
              <input type="checkbox" checked={ack} onChange={(e) => setAck(e.target.checked)} />
              记录本次测试查看；不将重复测试作为独立样本外证据
            </label>
            <p className="subtle-note">
              源码或契约不匹配会拒绝测试。旧报告始终可查看，不自动改写旧冻结文件。
            </p>
          </div>
        )}
      </section>
      <div className="toolbar report-toolbar">
        <Field label="已有报告">
          <select value={reportId} onChange={(e) => setReportId(e.target.value)}>
            {p.reports.map((r) => (
              <option key={r.id} value={r.id}>
                {r.run} / {r.name} · {r.split}
              </option>
            ))}
          </select>
        </Field>
      </div>
      {reportId && <ReportView id={reportId} fail={p.fail} />}
    </>
  )
}

function ReportView({ id, fail }: { id: string; fail: (e: unknown) => void }) {
  const [report, setReport] = useState<Row | null>(null)
  const [tab, setTab] = useState('net_value')
  const [strategy, setStrategy] = useState('ppo')
  const [orders, setOrders] = useState<Row>({ rows: [], total: 0 })
  const [page, setPage] = useState(0)
  const [date, setDate] = useState('')
  const [detailTab, setDetailTab] = useState('orders')
  useEffect(() => {
    let live = true
    setReport(null)
    setPage(0)
    setDate('')
    api('/reports/' + id)
      .then((r) => {
        if (live) {
          setReport(r)
          setStrategy(r.metrics.ppo ? 'ppo' : Object.keys(r.metrics)[0])
        }
      })
      .catch(fail)
    return () => {
      live = false
    }
  }, [id, fail])
  useEffect(() => {
    let live = true
    setOrders({ rows: [], total: 0 })
    if (report && strategy)
      api(
        `/reports/${id}/${detailTab}?strategy=${encodeURIComponent(strategy)}&page=${page}${date ? '&date=' + date : ''}`
      )
        .then((v) => {
          if (live) setOrders(v)
        })
        .catch((e) => {
          if (live) fail(e)
        })
    return () => {
      live = false
    }
  }, [id, report, strategy, page, date, detailTab, fail])
  if (!report) return <Empty>正在读取报告…</Empty>
  const series = Object.entries(report.series).map(([name, rows]) => ({
    name: strategyNames[name] || name,
    values: (rows as Row[]).map((r) => r[tab]),
  }))
  const dates = ((Object.values(report.series)[0] || []) as Row[]).map((r) => r.date.slice(0, 10))
  return (
    <section className="section">
      <div className="section-heading">
        <h2>{report.split === 'test' ? '最终测试' : '验证集'}回测</h2>
        <div className="button-group">
          <a className="button" href={`/api/artifacts/${id}/report.md`}>
            <FileText size={14} />
            报告
          </a>
          <a
            className="button icon-button"
            title="下载指标 CSV"
            href={`/api/artifacts/${id}/metrics.csv`}
          >
            <Download size={16} />
          </a>
        </div>
      </div>
      <div className="tabs">
        {[
          ['net_value', '净值'],
          ['drawdown', '回撤'],
          ['cash', '现金'],
        ].map(([key, label]) => (
          <button key={key} className={tab === key ? 'active' : ''} onClick={() => setTab(key)}>
            {label}
          </button>
        ))}
      </div>
      <Chart series={series} x={dates} height={310} percentage={tab === 'drawdown'} />
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>策略</th>
              <th>累计收益</th>
              <th>年化波动</th>
              <th>Sharpe</th>
              <th>最大回撤</th>
              <th>费用</th>
              <th>现金比例</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(report.metrics).map(([name, m]: [string, any]) => (
              <tr key={name} className={name === 'ppo' ? 'highlight' : ''}>
                <td>
                  <strong>{strategyNames[name] || name}</strong>
                </td>
                <td className={m.cumulative_return >= 0 ? 'positive' : 'negative'}>
                  {percent(m.cumulative_return)}
                </td>
                <td>{percent(m.annual_volatility)}</td>
                <td>{number(m.sharpe, 3)}</td>
                <td>{percent(m.max_drawdown)}</td>
                <td>{number(m.total_fees, 2)}</td>
                <td>{percent(m.mean_cash_ratio)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="section-heading secondary-heading">
        <h2>{detailTab === 'orders' ? '订单明细' : '持仓明细'}</h2>
        <span>
          {orders.total} 条{orders.date ? ' · ' + orders.date : ''}
        </span>
      </div>
      <div className="tabs">
        <button
          className={detailTab === 'orders' ? 'active' : ''}
          onClick={() => {
            setDetailTab('orders')
            setPage(0)
          }}
        >
          订单
        </button>
        <button
          className={detailTab === 'positions' ? 'active' : ''}
          onClick={() => {
            setDetailTab('positions')
            setPage(0)
          }}
        >
          持仓
        </button>
      </div>
      <div className="order-controls">
        <Field label="策略">
          <select
            value={strategy}
            onChange={(e) => {
              setStrategy(e.target.value)
              setPage(0)
            }}
          >
            {Object.keys(report.metrics).map((s) => (
              <option key={s} value={s}>
                {strategyNames[s] || s}
              </option>
            ))}
          </select>
        </Field>
        <Field label="交易日期">
          <input
            type="date"
            value={date}
            onChange={(e) => {
              setDate(e.target.value)
              setPage(0)
            }}
          />
        </Field>
        <button
          className="icon-button"
          title="清除日期筛选"
          onClick={() => {
            setDate('')
            setPage(0)
          }}
        >
          <X size={16} />
        </button>
        <a
          className="button icon-button"
          title="下载持仓 Parquet"
          href={`/api/artifacts/${id}/${strategy}/positions.parquet`}
        >
          <Box size={16} />
        </a>
        <a
          className="button icon-button"
          title="下载订单 CSV"
          href={`/api/artifacts/${id}/${strategy}/orders.csv`}
        >
          <Download size={16} />
        </a>
      </div>
      <div className="table-wrap orders-table">
        {detailTab === 'orders' ? (
          <table>
            <thead>
              <tr>
                <th>日期</th>
                <th>股票</th>
                <th>方向</th>
                <th>目标权重</th>
                <th>成交股数</th>
                <th>状态 / 原因</th>
                <th>费用</th>
              </tr>
            </thead>
            <tbody>
              {orders.rows.map((o: Row, i: number) => (
                <tr key={i}>
                  <td className="nowrap">{o.date?.slice(0, 10)}</td>
                  <td className="mono">{o.code}</td>
                  <td>{o.side === 'buy' ? '买入' : '卖出'}</td>
                  <td>{percent(o.target_weight)}</td>
                  <td>{number(o.filled_shares)}</td>
                  <td>
                    {o.status}
                    <small className="subtext">{o.reasons}</small>
                  </td>
                  <td>{number(o.fee, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <table>
            <thead>
              <tr>
                <th>日期</th>
                <th>股票</th>
                <th>持仓股数（研究口径）</th>
              </tr>
            </thead>
            <tbody>
              {orders.rows.map((o: Row) => (
                <tr key={o.code}>
                  <td>{o.date}</td>
                  <td className="mono">{o.code}</td>
                  <td>{number(o.shares)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
      <div className="pagination">
        <button
          className="icon-button"
          aria-label="上一页"
          disabled={page === 0}
          onClick={() => setPage(page - 1)}
        >
          <ChevronLeft size={16} />
        </button>
        <span>
          {page + 1} / {Math.max(1, Math.ceil(orders.total / 50))}
        </span>
        <button
          className="icon-button"
          aria-label="下一页"
          disabled={(page + 1) * 50 >= orders.total}
          onClick={() => setPage(page + 1)}
        >
          <ChevronRight size={16} />
        </button>
      </div>
    </section>
  )
}

function PredictionPage(p: Shared & { initial: string; resultJob: string }) {
  const eligible = p.models.filter((m) => m.trained && m.valid)
  const [modelId, setModelId] = useState(p.initial || eligible[0]?.id || '')
  const [date, setDate] = useState('2024-12-31')
  const [manual, setManual] = useState(false)
  const [savedAccount, setSavedAccount] = useState<Row | null>(null)
  const [accountSource, setAccountSource] = useState('cash')
  const [cash, setCash] = useState(1000000)
  const [positions, setPositions] = useState<
    { code: string; shares: number; available_shares: number }[]
  >([])
  const [receivable, setReceivable] = useState(0)
  const [due, setDue] = useState('2025-01-02')
  const [codes, setCodes] = useState<string[]>([])
  const [result, setResult] = useState<Row | null>(null)
  const [filter, setFilter] = useState('')
  const [busy, setBusy] = useState(false)
  const model = p.models.find((m) => m.id === modelId)
  const allCodes: string[] = model?.metadata.contract.stock_codes || []
  useEffect(() => {
    if (p.initial) setModelId(p.initial)
  }, [p.initial])
  useEffect(() => {
    setCodes(allCodes)
    setPositions([])
    setSavedAccount(null)
    setAccountSource('cash')
    setManual(false)
    setCash(model?.metadata.contract.initial_capital || 1000000)
    const dataset = p.datasets.find((d) => d.version === model?.dataset)
    if (dataset) setDate(dataset.end_date)
  }, [modelId])
  useEffect(() => {
    let live = true
    if (p.resultJob)
      api('/jobs/' + p.resultJob)
        .then((j) => {
          if (!live || !j.prediction) return
          const snapshot = j.prediction
          setResult(snapshot)
          setDate(snapshot.as_of)
          setCodes(snapshot.display_codes)
          setAccountSource('saved:' + p.resultJob)
          setSavedAccount(snapshot.account)
          setManual(false)
          setCash(snapshot.account.cash)
          setPositions(snapshot.account.positions || [])
          setReceivable(snapshot.account.receivables?.[0]?.amount || 0)
          setDue(snapshot.account.receivables?.[0]?.date || '2025-01-02')
        })
        .catch(p.fail)
    return () => {
      live = false
    }
  }, [p.resultJob, p.fail])
  async function selectAccount(value: string) {
    if (value === 'manual' && savedAccount && savedAccount.receivables?.length > 1) {
      p.fail(new Error('此快照包含多笔应收，请保留快照模式；当前手工表单只支持一笔应收。'))
      return
    }
    setAccountSource(value)
    setManual(value === 'manual')
    setSavedAccount(null)
    if (value.startsWith('saved:')) {
      setBusy(true)
      try {
        const prior = await api('/jobs/' + value.slice(6))
        const snapshot = prior.prediction
        setSavedAccount(snapshot.account)
        setDate(snapshot.as_of)
        setCash(snapshot.account.cash)
        setPositions(snapshot.account.positions || [])
        setReceivable(snapshot.account.receivables?.[0]?.amount || 0)
        setDue(snapshot.account.receivables?.[0]?.date || '2025-01-02')
      } catch (e) {
        p.fail(e)
      } finally {
        setBusy(false)
      }
    }
  }
  async function predict() {
    setBusy(true)
    try {
      p.openJob(
        await post('/predictions', {
          model_id: modelId,
          as_of: date,
          display_codes: codes,
          account:
            savedAccount ||
            (manual
              ? {
                  cash,
                  positions,
                  receivables: receivable ? [{ date: due, amount: receivable }] : [],
                }
              : null),
        })
      )
    } catch (e) {
      p.fail(e)
    } finally {
      setBusy(false)
    }
  }
  const rows =
    result?.rows.filter(
      (r: Row) => result.display_codes.includes(r.code) && r.code.includes(filter)
    ) || []
  return (
    <>
      <div className="form-grid prediction-form">
        <section className="section form-section" onChangeCapture={() => setResult(null)}>
          <div className="section-heading">
            <h2>推理输入</h2>
            <span>确定性 · 只读</span>
          </div>
          <Field label="已存模型">
            <select
              value={modelId}
              onChange={(e) => {
                setModelId(e.target.value)
                setResult(null)
                navigate('predict', { model: e.target.value })
              }}
            >
              <option value="">选择模型</option>
              {eligible.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.run} / {m.label}
                </option>
              ))}
            </select>
          </Field>
          <div className="field-grid">
            <Field label="观察日期">
              <input
                type="date"
                value={date}
                disabled={!!savedAccount}
                max={p.datasets.find((d) => d.version === model?.dataset)?.end_date}
                onChange={(e) => setDate(e.target.value)}
              />
            </Field>
            <Field label="账户来源">
              <select value={accountSource} onChange={(e) => selectAccount(e.target.value)}>
                <option value="cash">全现金假设账户</option>
                <option value="manual">手工研究账户</option>
                {p.jobs
                  .filter(
                    (j) =>
                      j.kind === 'prediction' &&
                      j.status === 'succeeded' &&
                      j.payload.model_id === modelId
                  )
                  .map((j) => (
                    <option key={j.id} value={'saved:' + j.id}>
                      已存账户 · {j.payload.as_of} · {j.id.slice(-6)}
                    </option>
                  ))}
              </select>
            </Field>
          </div>
          {manual && (
            <>
              <Field label="可用现金">
                <input
                  type="number"
                  min={0}
                  value={cash}
                  onChange={(e) => setCash(Number(e.target.value))}
                />
              </Field>
              <div className="account-rows">
                {positions.map((r, i) => (
                  <div key={i}>
                    <select
                      aria-label={`持仓股票 ${i + 1}`}
                      value={r.code}
                      onChange={(e) =>
                        setPositions(
                          positions.map((v, j) => (i === j ? { ...v, code: e.target.value } : v))
                        )
                      }
                    >
                      {allCodes.map((c) => (
                        <option key={c}>{c}</option>
                      ))}
                    </select>
                    <input
                      aria-label={`持仓股数 ${i + 1}`}
                      type="number"
                      min={0}
                      value={r.shares}
                      onChange={(e) =>
                        setPositions(
                          positions.map((v, j) =>
                            i === j ? { ...v, shares: Number(e.target.value) } : v
                          )
                        )
                      }
                    />
                    <input
                      aria-label={`可卖股数 ${i + 1}`}
                      type="number"
                      min={0}
                      value={r.available_shares}
                      onChange={(e) =>
                        setPositions(
                          positions.map((v, j) =>
                            i === j ? { ...v, available_shares: Number(e.target.value) } : v
                          )
                        )
                      }
                    />
                    <button
                      className="icon-button"
                      title="移除持仓"
                      onClick={() => setPositions(positions.filter((_, j) => i !== j))}
                    >
                      <X size={15} />
                    </button>
                  </div>
                ))}
              </div>
              <button
                className="text-button"
                onClick={() =>
                  setPositions([
                    ...positions,
                    {
                      code:
                        allCodes.find((c) => !positions.some((v) => v.code === c)) || allCodes[0],
                      shares: 100,
                      available_shares: 100,
                    },
                  ])
                }
              >
                <Plus size={14} />
                添加持仓（总股数 / 可卖股数）
              </button>
              <details>
                <summary>分红应收</summary>
                <div className="field-grid">
                  <Field label="未到账金额">
                    <input
                      type="number"
                      value={receivable}
                      min={0}
                      onChange={(e) => setReceivable(Number(e.target.value))}
                    />
                  </Field>
                  <Field label="预计到账日">
                    <input type="date" value={due} onChange={(e) => setDue(e.target.value)} />
                  </Field>
                </div>
              </details>
            </>
          )}
          <div className="form-actions">
            <button
              className="primary"
              disabled={
                busy ||
                !modelId ||
                !codes.length ||
                (accountSource.startsWith('saved:') && !savedAccount)
              }
              onClick={predict}
            >
              <Target size={16} />
              {busy ? '提交中…' : '预测动作'}
            </button>
          </div>
          <p className="subtle-note">历史收盘后的目标仓位，不读取次日价格，不执行订单。</p>
        </section>
        <section className="section form-section">
          <div className="section-heading">
            <h2>关注股票</h2>
            <span>完整推理池：{allCodes.length} 只</span>
          </div>
          <StockPicker
            stocks={allCodes.map((code) => ({ code }))}
            selected={codes}
            onChange={(values) => {
              setCodes(values)
              setResult(null)
            }}
            preset={p.defaults.light_codes}
          />
          <p className="subtle-note">
            这里只筛选结果，其他股票仍参与组合计算。改变股票池需从模型库显式迁移。
          </p>
        </section>
      </div>
      {result && (
        <section className="section prediction-result">
          <div className="section-heading">
            <h2>动作结果 · {result.as_of}</h2>
            <a className="button" href={`/api/predictions/${p.resultJob}/download`}>
              <Download size={15} />
              导出快照
            </a>
          </div>
          <div className="stats-band compact">
            <Stat label="研究账户净值" value={number(result.equity, 2)} />
            <Stat label="当前现金" value={percent(result.current_cash_ratio)} />
            <Stat label="目标现金" value={percent(result.target_cash_ratio)} />
            <Stat label="完整股票池" value={result.pool_size} />
          </div>
          <div className="search table-search">
            <Search size={15} />
            <input
              placeholder="筛选结果代码"
              aria-label="筛选结果代码"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
            />
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>股票</th>
                  <th>动作意图</th>
                  <th>当前权重</th>
                  <th>目标权重</th>
                  <th>调整参考金额</th>
                  <th>持仓 / 可卖</th>
                  <th>特征</th>
                  <th>市场买 / 卖</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r: Row) => (
                  <tr key={r.code}>
                    <td className="mono">{r.code}</td>
                    <td
                      className={
                        r.intent === 'buy' ? 'positive' : r.intent === 'sell' ? 'negative' : ''
                      }
                    >
                      {
                        { buy: '买入', sell: '卖出', exit: '清仓', unchanged: '仓位基本不变' }[
                          r.intent as string
                        ]
                      }
                    </td>
                    <td>{percent(r.current_weight)}</td>
                    <td>{percent(r.target_weight)}</td>
                    <td>{number(r.reference_amount, 2)}</td>
                    <td>
                      {number(r.shares)} / {number(r.available_shares)}
                    </td>
                    <td>{r.valid ? '有效' : '无效 / 预热不足'}</td>
                    <td>
                      {r.buy_allowed ? '允许' : '受限'} / {r.sell_allowed ? '允许' : '受限'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="subtle-note">
            参考金额以当日收盘净值估算；HFQ 研究口径，不代表下一日可成交金额。
          </p>
        </section>
      )}
    </>
  )
}

function JobPage(p: Shared & { id: string }) {
  const [job, setJob] = useState<Row | null>(null)
  const [events, setEvents] = useState<Row[]>([])
  const [tab, setTab] = useState('metrics')
  const [connected, setConnected] = useState(false)
  const reload = useCallback(
    () =>
      api('/jobs/' + p.id)
        .then((j) => {
          setJob(j)
          setEvents((previous) =>
            [...new Map([...j.events, ...previous].map((e: Row) => [e.sequence, e])).values()].sort(
              (a, b) => a.sequence - b.sequence
            )
          )
        })
        .catch(p.fail),
    [p.id, p.fail]
  )
  useEffect(() => {
    setEvents([])
    setJob(null)
    reload()
    const sse = new EventSource('/api/jobs/' + p.id + '/events')
    sse.onopen = () => setConnected(true)
    sse.onerror = () => setConnected(false)
    sse.onmessage = (message) => {
      const e = JSON.parse(message.data)
      setEvents((previous) =>
        [...new Map([...previous, e].map((v) => [v.sequence, v])).values()].sort(
          (a, b) => a.sequence - b.sequence
        )
      )
    }
    sse.addEventListener('complete', () => {
      sse.close()
      setConnected(false)
      reload()
      p.refresh()
    })
    const timer = setInterval(reload, 2500)
    return () => {
      sse.close()
      clearInterval(timer)
    }
  }, [p.id, reload])
  if (!job) return <Empty>读取任务…</Empty>
  const progress = [...events].reverse().find((e) => e.type === 'progress')
  const phase = [...events].reverse().find((e) => e.phase)?.phase
  const metrics = events.filter((e) => e.type === 'metrics')
  const validations = events.filter((e) => e.type === 'validation')
  const budget = job.payload.total_timesteps
    ? Math.ceil(job.payload.total_timesteps / job.payload.n_steps) * job.payload.n_steps
    : 0
  const steps = Math.max(progress?.timesteps || 0, metrics.at(-1)?.timesteps || 0)
  const done = terminal.has(job.status)
  const associatedModels = p.models.filter((m) => m.run === job.id)
  const relatedReports = p.reports.filter(
    (r) => r.run === job.id || r.relative_path.includes(job.id)
  )
  return (
    <>
      <button className="text-button back" onClick={() => navigate('jobs')}>
        <ChevronLeft size={15} />
        全部任务
      </button>
      <section className="section job-heading">
        <div>
          <div className="section-heading">
            <h2>{job.payload.name || kindName[job.kind]}</h2>
            <Badge status={job.status} />
          </div>
          <span className="mono muted">{job.id}</span>
        </div>
        {!done && (
          <button
            className="danger"
            onClick={() =>
              post('/jobs/' + job.id + '/stop', {})
                .then(reload)
                .catch(p.fail)
            }
          >
            <Square size={14} />
            请求停止
          </button>
        )}
      </section>
      {job.error && (
        <div className="error-banner" role="alert">
          {job.error}
        </div>
      )}
      <div className="stats-band">
        <Stat
          label="当前阶段"
          value={
            <span className="phase-value">
              {done ? statusName[job.status] : phaseName[phase] || '等待工作进程'}
            </span>
          }
          detail={connected ? '事件流已连接' : done ? '事件已归档' : '正在连接事件流'}
        />
        <Stat
          label="训练步数"
          value={number(steps)}
          detail={budget ? `有效预算 ${number(budget)}` : '非训练任务'}
        />
        <Stat label="更新 Epochs" value={number(metrics.at(-1)?.metrics.updates)} />
        <Stat
          label="创建时间"
          value={<span className="date-value">{dateTime(job.created)}</span>}
        />
      </div>
      {budget > 0 && (
        <div className="progress-track" aria-label="训练进度">
          <div style={{ width: `${Math.min(100, (steps / budget) * 100)}%` }} />
        </div>
      )}
      <section className="section">
        <div className="tabs">
          {[
            ['metrics', '训练指标'],
            ['validation', '验证与模型'],
            ['log', '日志'],
            ['config', '配置快照'],
          ].map(([key, label]) => (
            <button key={key} className={tab === key ? 'active' : ''} onClick={() => setTab(key)}>
              {label}
            </button>
          ))}
        </div>
        {tab === 'metrics' &&
          (metrics.length ? (
            <div className="two-charts">
              <div>
                <h3>价值损失</h3>
                <Chart
                  x={metrics.map((m) => m.timesteps)}
                  series={[
                    {
                      name: 'Value loss',
                      values: metrics.map((m) => m.metrics['train/value_loss'] ?? null),
                    },
                  ]}
                />
              </div>
              <div>
                <h3>策略更新</h3>
                <Chart
                  x={metrics.map((m) => m.timesteps)}
                  series={[
                    {
                      name: 'KL',
                      values: metrics.map((m) => m.metrics['train/approx_kl'] ?? null),
                    },
                    {
                      name: 'Clip fraction',
                      values: metrics.map((m) => m.metrics['train/clip_fraction'] ?? null),
                    },
                  ]}
                />
              </div>
            </div>
          ) : (
            <Empty>{job.kind === 'training' ? '等待首轮参数更新' : '此任务不更新模型参数'}</Empty>
          ))}
        {tab === 'validation' && (
          <>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>步数</th>
                    <th>累计收益</th>
                    <th>Sharpe</th>
                    <th>回撤</th>
                    <th>当前选择</th>
                  </tr>
                </thead>
                <tbody>
                  {validations.map((v) => (
                    <tr key={v.sequence}>
                      <td>{number(v.timesteps)}</td>
                      <td>{percent(v.metrics.cumulative_return)}</td>
                      <td>{number(v.metrics.sharpe, 3)}</td>
                      <td>{percent(v.metrics.max_drawdown)}</td>
                      <td>{v.best_path || '未选出'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {associatedModels.map((m) => (
              <button
                className="artifact-link"
                key={m.id}
                onClick={() => navigate('models', { model: m.id })}
              >
                <Box size={16} />
                {m.name}
                <ArrowUpRight size={14} />
              </button>
            ))}
          </>
        )}
        {tab === 'log' && (
          <>
            <a className="button" href={`/api/jobs/${job.id}/log`}>
              <Download size={14} />
              下载日志
            </a>
            <pre className="log-view">
              {job.log || events.map((e) => JSON.stringify(e)).join('\n') || '等待任务日志…'}
            </pre>
          </>
        )}
        {tab === 'config' && <pre className="log-view">{JSON.stringify(job.payload, null, 2)}</pre>}
      </section>
      {done && (
        <section className="section result-links">
          <div className="section-heading">
            <h2>任务产物</h2>
          </div>
          {job.check && (
            <div className="audit-strip">
              <span>
                股票 <b>{job.check.stock_count}</b>
              </span>
              <span>
                观测 <b>{job.check.observation_shape.join(' × ')}</b>
              </span>
              <span>
                有效训练区间{' '}
                <b>
                  {job.check.train_initial_date} 至 {job.check.train_final_date}
                </b>
              </span>
              <span>
                参数未改变 <b>{job.check.parameters_unchanged ? '是' : '否'}</b>
              </span>
            </div>
          )}
          {job.audit && (
            <div className="audit-strip">
              <span>
                参数变化{' '}
                <b>
                  {job.audit.changed_tensors}/{job.audit.tensor_count}
                </b>
              </span>
              <span>
                优化器更新 <b>{number(job.audit.optimizer_steps_min)}</b>
              </span>
              <span>
                保存加载 <b>{job.audit.save_load_deterministic_equal ? '一致' : '待校验'}</b>
              </span>
              <span>
                数值检查 <b>{job.audit.all_parameters_finite ? '有限' : '异常'}</b>
              </span>
            </div>
          )}
          {job.result?.prediction_path && (
            <button
              className="primary"
              onClick={() => navigate('predict', { model: job.payload.model_id, result: job.id })}
            >
              <Target size={16} />
              查看预测结果
            </button>
          )}
          {job.result?.model_id && (
            <button onClick={() => navigate('models', { model: job.result.model_id })}>
              <Box size={16} />
              打开迁移模型
            </button>
          )}
          {relatedReports.map((r) => (
            <button key={r.id} onClick={() => navigate('backtest', { report: r.id })}>
              <FileText size={16} />
              {r.name}
              <ArrowUpRight size={14} />
            </button>
          ))}
          {job.result?.output && <code className="output-path">{job.result.output}</code>}
          {!job.result && (
            <p className="subtle-note">
              没有登记完成的产物。此前已经保存的 checkpoint 可在模型库查看。
            </p>
          )}
        </section>
      )}
    </>
  )
}

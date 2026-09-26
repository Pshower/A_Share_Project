import { useEffect, useRef, useState } from 'react'
import { Plus, Save, RefreshCw, Download, Database, Check, Play, Copy, Square } from 'lucide-react'
import { api, post, number, percent, statusName } from './api'
import FeatureEditor, { type FeatureDefinition } from './FeatureEditor'
import PpoSettings, { advancedDefaults } from './PpoSettings'

type Row = Record<string, any>
const initialTraining = {
  ...advancedDefaults,
  total_timesteps: 8192,
  seed: 42,
  lookback: 1,
  n_steps: 256,
  batch_size: 64,
  n_epochs: 10,
  learning_rate: 0.0003,
  validation_rollouts: 10,
  device: 'auto',
  training_date_policy: 'intersection',
}
const stages: Row = { download: '缺项下载', prepare: '数据构建', check: '配置检查', train: '训练' }

export default function ExperimentsPage({
  datasets,
  fail,
  initialId,
  refreshResources,
}: {
  datasets: Row[]
  fail: (error: unknown) => void
  initialId: string
  refreshResources: () => Promise<void>
}) {
  const [items, setItems] = useState<Row[]>([])
  const [id, setId] = useState(initialId)
  const [detail, setDetail] = useState<Row | null>(null)
  const [editing, setEditing] = useState(!initialId)
  const [name, setName] = useState('PPO 数据与训练实验')
  const [dataset, setDataset] = useState(
    datasets.find((d) => d.name === 'research_v3')?.id || datasets[0]?.id || ''
  )
  const [template, setTemplate] = useState<Row | null>(null)
  const [codes, setCodes] = useState('000001 600030')
  const [seeds, setSeeds] = useState('42')
  const [training, setTraining] = useState<Row>({ ...initialTraining })
  const [build, setBuild] = useState(false)
  const [sourceMode, setSourceMode] = useState('dataset')
  const [histories, setHistories] = useState<Row[]>([])
  const [historyIds, setHistoryIds] = useState<string[]>([])
  const [version, setVersion] = useState('')
  const [dates, setDates] = useState<Row>({
    start_date: '',
    train_end: '',
    val_end: '',
    end_date: '',
  })
  const [extras, setExtras] = useState<FeatureDefinition[]>([])
  const [featuresValid, setFeaturesValid] = useState(true)
  const [parentId, setParentId] = useState<string | null>(null)
  const [network, setNetwork] = useState(false)
  const [trainApproval, setTrainApproval] = useState(false)
  const [busy, setBusy] = useState(false)
  const inherit = useRef(true)
  const currentId = useRef(id)
  const refreshedDatasets = useRef(new Set<string>())
  const refreshing = useRef(new Set<string>())
  currentId.current = id
  const pool = [...new Set(codes.split(/[\s,，]+/).filter(Boolean))]
  async function refresh() {
    if (refreshing.current.has(id)) return
    refreshing.current.add(id)
    try {
      setItems(await api('/experiments'))
      if (id) {
        const state = await api('/experiments/' + id)
        if (currentId.current === id) setDetail(state)
        if (state.dataset_id && !datasets.some((d) => d.id === state.dataset_id) && !refreshedDatasets.current.has(state.dataset_id)) {
          refreshedDatasets.current.add(state.dataset_id)
          await refreshResources()
        }
      }
    } finally {
      refreshing.current.delete(id)
    }
  }
  useEffect(() => {
    setId(initialId)
  }, [initialId])
  useEffect(() => {
    refresh().catch(fail)
    api('/market/snapshots/history').then(setHistories).catch(fail)
  }, [id, fail])
  useEffect(() => {
    setDetail(null)
    setNetwork(false)
    setTrainApproval(false)
  }, [id])
  const active = detail?.jobs.some((j: Row) =>
    ['queued', 'running', 'stop_requested'].includes(j.status)
  )
  useEffect(() => {
    if (!active) return
    const timer = setInterval(() => refresh().catch(fail), 3000)
    return () => clearInterval(timer)
  }, [active, id, fail])
  useEffect(() => {
    let live = true
    if (dataset)
      api('/datasets/' + dataset)
        .then((d) => {
          if (!live) return
          setTemplate(d)
          if (inherit.current) {
            setDates(
              Object.fromEntries(
                ['start_date', 'train_end', 'val_end', 'end_date'].map((k) => [
                  k,
                  d.manifest.build_config[k],
                ])
              )
            )
            setExtras(d.manifest.extra_features || [])
            setCodes(
              d.manifest.stock_codes
                .filter((c: string) => ['000001', '600030'].includes(c))
                .join(' ') || d.manifest.stock_codes.slice(0, 2).join(' ')
            )
          }
        })
        .catch(fail)
    return () => {
      live = false
    }
  }, [dataset, fail])
  function clone() {
    const recipe = detail!.recipe
    inherit.current = false
    setParentId(detail!.id)
    setName(recipe.name + ' · 修订')
    setDataset(recipe.training.dataset_id)
    setTraining(recipe.training)
    setCodes(recipe.training.codes.join(' '))
    setSeeds(recipe.seeds.join(' '))
    setBuild(!!recipe.build)
    setSourceMode(recipe.source_mode)
    setHistoryIds(recipe.history_ids)
    if (recipe.build) {
      setDates(recipe.build)
      setExtras(recipe.build.extra_features || [])
      setVersion('revision_' + Date.now())
    }
    setEditing(true)
  }
  async function save() {
    setBusy(true)
    try {
      const seedList = seeds
        .split(/[\s,，]+/)
        .filter(Boolean)
        .map(Number)
      const saved = await post('/experiments', {
        name,
        training: { ...training, name, dataset_id: dataset, codes: pool },
        build: build
          ? {
              version,
              dataset_id: dataset,
              codes: pool,
              ...Object.fromEntries(
                ['start_date', 'train_end', 'val_end', 'end_date'].map((k) => [k, dates[k]])
              ),
              extra_features: extras,
            }
          : null,
        source_mode: build ? sourceMode : 'dataset',
        history_ids: build && sourceMode === 'history' ? historyIds : [],
        seeds: seedList,
        parent_id: parentId,
      })
      setId(saved.id)
      setEditing(false)
      window.location.hash = 'experiments?id=' + saved.id
      setItems(await api('/experiments'))
    } catch (e) {
      fail(e)
    } finally {
      setBusy(false)
    }
  }
  async function run(stage: string) {
    setBusy(true)
    try {
      await post(`/experiments/${id}/${stage}`, {
        allow_network: network,
        allow_training: trainApproval,
      })
      await refresh()
      setNetwork(false)
      setTrainApproval(false)
    } catch (e) {
      fail(e)
    } finally {
      setBusy(false)
    }
  }
  return (
    <>
      <div className="toolbar">
        <label className="field">
          已保存实验
          <select
            aria-label="已保存实验"
            value={id}
            onChange={(e) => {
              setId(e.target.value)
              setEditing(!e.target.value)
              window.location.hash = 'experiments' + (e.target.value ? '?id=' + e.target.value : '')
            }}
          >
            <option value="">新实验</option>
            {items.map((e) => (
              <option key={e.id} value={e.id}>
                {e.recipe.name} · {e.id}
              </option>
            ))}
          </select>
        </label>
        <button
          onClick={() => {
            setEditing(true)
            setParentId(null)
          }}
        >
          <Plus size={15} />
          新建实验配置
        </button>
        {detail && (
          <button onClick={clone}>
            <Copy size={15} />
            复制为新修订
          </button>
        )}
        <button className="icon-button" title="刷新实验状态" onClick={() => refresh().catch(fail)}>
          <RefreshCw size={16} />
        </button>
        {active && (
          <button className="danger" disabled={busy} onClick={() => run('stop')}>
            <Square size={15} />
            停止本实验未完成任务
          </button>
        )}
      </div>
      {editing && (
        <>
          <div className="form-grid">
            <section className="section form-section">
              <div className="section-heading">
                <h2>实验数据</h2>
                <span>{parentId || '新配置'}</span>
              </div>
              <label className="field">
                实验名称
                <input
                  aria-label="一体化实验名称"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                />
              </label>
              <label className="field">
                数据模板
                <select
                  aria-label="实验数据模板"
                  value={dataset}
                  onChange={(e) => {
                    inherit.current = true
                    setDataset(e.target.value)
                  }}
                >
                  {datasets.map((d) => (
                    <option key={d.id} value={d.id}>
                      {d.name} · {d.features} 特征
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                股票代码
                <textarea
                  aria-label="实验股票代码"
                  rows={3}
                  value={codes}
                  onChange={(e) => setCodes(e.target.value)}
                />
              </label>
              <button
                className="text-button"
                onClick={() => setCodes(template?.manifest.stock_codes.join(' ') || '')}
              >
                采用模板完整股票池
              </button>
              <label className="check-label">
                <input
                  type="checkbox"
                  checked={build}
                  onChange={(e) => setBuild(e.target.checked)}
                />
                构建独立数据与特征版本
              </label>
              {build && (
                <>
                  <label className="field">
                    新数据版本
                    <input
                      aria-label="实验新数据版本"
                      value={version}
                      onChange={(e) => setVersion(e.target.value)}
                    />
                  </label>
                  <label className="field">
                    原始来源
                    <select
                      aria-label="实验原始来源"
                      value={sourceMode}
                      onChange={(e) => setSourceMode(e.target.value)}
                    >
                      <option value="dataset">模板的本地原始文件</option>
                      <option value="history">HFQ 下载快照组合</option>
                    </select>
                  </label>
                  <div className="field-grid">
                    {['start_date', 'train_end', 'val_end', 'end_date'].map((k, i) => (
                      <label className="field" key={k}>
                        {['研究起始', '训练截止', '验证截止', '研究截止'][i]}
                        <input
                          aria-label={'实验' + k}
                          type="date"
                          value={dates[k]}
                          onChange={(e) => setDates({ ...dates, [k]: e.target.value })}
                        />
                      </label>
                    ))}
                  </div>
                  {sourceMode === 'history' && (
                    <div className="experiment-sources">
                      {histories.map((h) => (
                        <label className="check-label" key={h.id}>
                          <input
                            type="checkbox"
                            checked={historyIds.includes(h.id)}
                            onChange={(e) =>
                              setHistoryIds(
                                e.target.checked
                                  ? [...historyIds, h.id]
                                  : historyIds.filter((i) => i !== h.id)
                              )
                            }
                          />
                          {h.id} · {h.request.codes.length} 股 · {h.status}
                        </label>
                      ))}
                    </div>
                  )}
                </>
              )}
            </section>
            <section className="section form-section">
              <div className="section-heading">
                <h2>PPO 与预算</h2>
              </div>
              <label className="field">
                随机种子列表
                <input
                  aria-label="实验随机种子"
                  value={seeds}
                  onChange={(e) => setSeeds(e.target.value)}
                />
              </label>
              <div className="field-grid">
                {[
                  ['total_timesteps', '每种子训练步数'],
                  ['n_steps', 'Rollout 步数'],
                  ['batch_size', 'Batch 大小'],
                  ['n_epochs', '每轮 Epochs'],
                  ['learning_rate', '学习率'],
                  ['lookback', '观察窗口'],
                  ['validation_rollouts', '验证间隔 / Rollout'],
                ].map(([k, label]) => (
                  <label className="field" key={k}>
                    {label}
                    <input
                      aria-label={'实验' + k}
                      type="number"
                      value={training[k]}
                      step={k === 'learning_rate' ? 0.0001 : 1}
                      onChange={(e) => setTraining({ ...training, [k]: Number(e.target.value) })}
                    />
                  </label>
                ))}
              </div>
              <label className="field">
                设备
                <select
                  aria-label="实验计算设备"
                  value={training.device}
                  onChange={(e) => setTraining({ ...training, device: e.target.value })}
                >
                  <option value="auto">自动</option>
                  <option value="cuda">CUDA</option>
                  <option value="cpu">CPU</option>
                </select>
              </label>
              <PpoSettings
                value={training}
                onChange={(patch) => setTraining((current) => ({ ...current, ...patch }))}
              />
            </section>
          </div>
          {build && (
            <FeatureEditor
              datasetId={dataset}
              codes={pool}
              definitions={extras}
              onChange={setExtras}
              onValidity={setFeaturesValid}
              fail={fail}
            />
          )}
          <div className="form-actions">
            <button
              className="primary"
              disabled={busy || !name || !pool.length || (build && (!version || !featuresValid))}
              onClick={save}
            >
              <Save size={16} />
              保存实验配置
            </button>
          </div>
        </>
      )}
      {detail && !editing && (
        <>
          <section className="section">
            <div className="section-heading">
              <h2>{detail.recipe.name}</h2>
              <span>{detail.id}</span>
            </div>
            <div className="stats-band">
              <div className="stat">
                <span>请求总步数</span>
                <strong>{number(detail.total_requested_steps)}</strong>
              </div>
              <div className="stat">
                <span>种子</span>
                <strong>{detail.recipe.seeds.join(' / ')}</strong>
              </div>
              <div className="stat">
                <span>估计 Rollout 内存</span>
                <strong>{detail.estimated_rollout_mb < 0.1 ? '<0.1' : number(detail.estimated_rollout_mb, 1)} MB</strong>
              </div>
              <div className="stat">
                <span>可用数据版本</span>
                <strong className="experiment-artifact-id">
                  {detail.dataset_id ? (
                    <a href={'#data?dataset=' + detail.dataset_id}>{detail.dataset_name}</a>
                  ) : (
                    '待准备'
                  )}
                </strong>
              </div>
            </div>
            {detail.coverage && (
              <p>
                共同训练日期：{detail.coverage.train_start || '—'} 至{' '}
                {detail.coverage.train_end || '—'} · {detail.coverage.train_rows} 条
              </p>
            )}
            {detail.source_report.missing.length > 0 && (
              <p role="status">缺少覆盖来源：{detail.source_report.missing.join('、')}</p>
            )}
            {detail.errors.map((error: string) => (
              <p role="alert" key={error}>
                {error}
              </p>
            ))}
            {detail.source_report.conflicts.length > 0 && (
              <pre>{JSON.stringify(detail.source_report.conflicts, null, 2)}</pre>
            )}
            <div className="form-actions">
              {detail.recipe.source_mode === 'history' && (
                <>
                  <label className="check-label">
                    <input
                      type="checkbox"
                      checked={network}
                      onChange={(e) => setNetwork(e.target.checked)}
                    />
                    允许下载本实验缺失的 HFQ 数据
                  </label>
                  <button
                    disabled={busy || !network || !detail.source_report.missing.length}
                    onClick={() => run('download')}
                  >
                    <Download size={15} />
                    下载缺项
                  </button>
                </>
              )}
              <button disabled={busy || !detail.can_prepare} onClick={() => run('prepare')}>
                <Database size={15} />
                准备数据
              </button>
              <button disabled={busy || active || !detail.can_check} onClick={() => run('check')}>
                <Check size={15} />
                检查全部种子
              </button>
            </div>
            <details>
              <summary>冻结配置与来源绑定</summary>
              <pre>
                {JSON.stringify(
                  {
                    recipe: detail.recipe,
                    bindings: detail.source_report.bindings,
                    fingerprint: detail.preparation_fingerprint,
                  },
                  null,
                  2
                )}
              </pre>
            </details>
            <label className="check-label">
              <input
                type="checkbox"
                checked={trainApproval}
                onChange={(e) => setTrainApproval(e.target.checked)}
              />
              批准 {detail.recipe.seeds.length} 个种子的参数更新，共{' '}
              {number(detail.total_effective_steps)} 步（请求 {number(detail.total_requested_steps)}{' '}
              步）
            </label>
            <button
              className="primary"
              disabled={busy || active || !trainApproval || !detail.checks_ready}
              onClick={() => run('train')}
            >
              <Play size={15} />
              启动已检查实验
            </button>
          </section>
          <section className="section">
            <div className="section-heading">
              <h2>阶段与任务</h2>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>阶段</th>
                    <th>种子</th>
                    <th>状态</th>
                    <th>任务</th>
                  </tr>
                </thead>
                <tbody>
                  {detail.jobs.map((j: Row) => (
                    <tr key={j.id}>
                      <td>{stages[j.stage.split(':')[0]] || j.stage}</td>
                      <td>{j.payload.seed ?? '—'}</td>
                      <td>
                        {statusName[j.status] || j.status}
                        {j.error && <small className="subtext">{j.error}</small>}
                      </td>
                      <td>
                        <a href={'#jobs?job=' + j.id}>{j.id}</a>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
          <section className="section">
            <div className="section-heading">
              <h2>已完成种子 · 验证对照</h2>
              <span>不是最终测试</span>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>种子</th>
                    <th>通过选模</th>
                    <th>收益</th>
                    <th>Sharpe</th>
                    <th>回撤</th>
                    <th>等权收益</th>
                    <th>手续费 + 滑点</th>
                  </tr>
                </thead>
                <tbody>
                  {detail.summaries.map((r: Row) => (
                    <tr key={r.job_id}>
                      <td>{r.seed}</td>
                      <td>{r.selected ? '是' : '无合格候选'}</td>
                      <td>{percent(r.metrics.ppo?.cumulative_return)}</td>
                      <td>{number(r.metrics.ppo?.sharpe, 3)}</td>
                      <td>{percent(r.metrics.ppo?.max_drawdown)}</td>
                      <td>{percent(r.metrics.equal_weight?.cumulative_return)}</td>
                      <td>
                        {r.metrics.ppo
                          ? number(r.metrics.ppo.total_fees + r.metrics.ppo.total_slippage_cost, 2)
                          : '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </>
      )}
    </>
  )
}

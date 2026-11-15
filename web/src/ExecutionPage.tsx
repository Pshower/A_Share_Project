import { useEffect, useState } from 'react'
import { RefreshCw, Wallet } from 'lucide-react'
import { api, number, percent } from './api'
import Chart from './Chart'

type Row = Record<string, any>

const bookNames: Record<string, string> = {
  live: '实盘',
  paper: '模拟盘',
  backtest: '回测账',
}
const bookNotes: Record<string, string> = {
  live: '只记真实成交（source = manual）',
  paper: '按计划用真实开盘价成交（source = model_open）',
  backtest: '129 起点扫描的同一引擎导出，可逐位复算',
}
const statusNames: Record<string, string> = {
  decided: '已决策',
  stale: '未推进',
  no_score: '无可用分',
}
const changeNames: Record<string, string> = { held: '持有', in: '换入', new: '新开' }
const decisionNames: Record<string, string> = { rebalance: '再平衡', hold: '持有不动', legacy: '旧日频' }
const TOP_CHOICES = [5, 10, 20, 30]

export default function ExecutionPage({
  fail,
  section,
}: {
  fail: (error: unknown) => void
  section: string
}) {
  const [books, setBooks] = useState<Row[]>([])
  const [name, setName] = useState('paper')
  const [detail, setDetail] = useState<Row | null>(null)
  const [shadow, setShadow] = useState<Row | null>(null)
  const [rec, setRec] = useState<Row | null>(null)
  const [busy, setBusy] = useState(false)
  const [day, setDay] = useState('')
  const [topK, setTopK] = useState(10)

  async function load() {
    setBusy(true)
    try {
      const [list, sh, advice] = await Promise.all([
        api<Row[]>('/execution/books'),
        api<Row>('/execution/shadow'),
        api<Row>('/recommend/unified?book=paper').catch(() => null),
      ])
      setBooks(list)
      setShadow(sh)
      setRec(advice)
      const days = sh.days || []
      if (days.length && !days.some((d: Row) => d.panel_date === day)) {
        setDay(days[days.length - 1].panel_date)
      }
      const picked = list.find((b) => b.name === name)?.opened ? name : list.find((b) => b.opened)?.name
      if (picked) {
        setName(picked)
        setDetail(await api<Row>(`/execution/books/${picked}`))
      } else {
        setDetail(null)
      }
    } catch (error) {
      fail(error)
    } finally {
      setBusy(false)
    }
  }

  async function select(next: string) {
    setName(next)
    setBusy(true)
    try {
      setDetail(await api<Row>(`/execution/books/${next}`))
    } catch (error) {
      fail(error)
    } finally {
      setBusy(false)
    }
  }

  useEffect(() => {
    void load()
  }, [])

  const curve = detail?.curve || {}
  const latest = detail?.last_snapshot || {}
  const days = shadow?.days || []
  const selected = days.find((d: Row) => d.panel_date === day)
  const rows = selected ? selected.names.slice(0, topK) : []
  const cutoff = rows.length ? rows[rows.length - 1].score : null

  return (
    <>
      <div className="page-heading">
        <div>
          <p className="eyebrow">
            {section === 'unified'
              ? '全资产建议'
              : section === 'recommend'
                ? '每日推荐'
                : section === 'shadow'
                  ? '影子记录'
                  : '执行层'}
          </p>
          <h1>
            {section === 'unified'
              ? '今天该持有什么'
              : section === 'recommend'
                ? '每天推荐的股票'
                : section === 'shadow'
                  ? 'S2 影子运行记录'
                  : '三本账'}
          </h1>
          <p>
            {section === 'unified'
              ? '按账本自己的净值和峰值算 CPPI 风险仓位；每一类资产都列出来，包括权重为 0 的。'
              : section === 'recommend'
                ? '选择交易日与 topK，名单按分数从低到高排列（分数越低越优先）。'
                : section === 'shadow'
                  ? '记录器每天写一行；decided 才有名单，stale 表示面板没有推进。'
                  : '净值、持仓与成交都从账本文件直接读出，与命令行一致。'}
          </p>
        </div>
        <button className="icon-button" aria-label="刷新" onClick={load} disabled={busy}>
          <RefreshCw size={18} className={busy ? 'spin' : ''} />
        </button>
      </div>

      {section === 'books' && (
        <>
          <section className="section">
            <div className="button-group">
              {['live', 'paper', 'backtest'].map((id) => {
                const row = books.find((b) => b.name === id)
                return (
                  <button
                    key={id}
                    className={name === id ? 'primary' : ''}
                    onClick={() => select(id)}
                    disabled={busy || !row?.opened}
                    title={bookNotes[id]}
                  >
                    {bookNames[id]}
                    {row?.opened ? '' : '（未开账）'}
                  </button>
                )
              })}
            </div>
          </section>

          {detail?.opened ? (
            <>
              <section className="section">
                <div className="section-heading">
                  <h2>{bookNames[name]} · 净值</h2>
                  <span>
                    {detail.start_date} 起 · {number(curve.days)} 个交易日
                  </span>
                </div>
                <div className="stats-band">
                  <div className="stat">
                    <span>净值</span>
                    <strong>{number(curve.nav, 2)}</strong>
                    <small>本金 {number(detail.capital, 0)}</small>
                  </div>
                  <div className="stat">
                    <span>累计收益</span>
                    <strong>{percent(curve.total_return)}</strong>
                    <small>峰值 {number(curve.peak, 2)}</small>
                  </div>
                  <div className="stat">
                    <span>当前回撤</span>
                    <strong>{percent(curve.drawdown)}</strong>
                    <small>最差 {percent(curve.worst_drawdown)}</small>
                  </div>
                  <div className="stat">
                    <span>现金 / 成交</span>
                    <strong>{number(detail.cash, 0)}</strong>
                    <small>
                      {number(detail.trades.count)} 笔 · 费用 {number(detail.trades.fees, 2)}
                    </small>
                  </div>
                </div>
              </section>

              <section className="section chart-section">
                <Chart x={detail.dates} series={[{ name: '净值', values: detail.navs }]} height={300} />
              </section>

              <section className="section">
                <div className="section-heading">
                  <h2>持仓</h2>
                  <span>
                    {detail.positions.length} 只 · 快照 {latest.date || '—'}
                    {latest.provisional ? '（临时价）' : ''}
                  </span>
                </div>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>代码</th>
                        <th>股数</th>
                        <th>价格</th>
                        <th>市值</th>
                        <th>权重</th>
                      </tr>
                    </thead>
                    <tbody>
                      {detail.positions.map((p: Row) => (
                        <tr key={p.code}>
                          <td>{p.code}</td>
                          <td>{number(p.shares)}</td>
                          <td>{number(p.price, 3)}</td>
                          <td>{number(p.value, 2)}</td>
                          <td>{percent(p.weight)}</td>
                        </tr>
                      ))}
                      {detail.positions.length === 0 && (
                        <tr>
                          <td colSpan={5}>空仓</td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </section>

              <section className="section">
                <div className="section-heading">
                  <h2>最近成交</h2>
                  <span>{number(detail.trades.count)} 笔</span>
                </div>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>日期</th>
                        <th>代码</th>
                        <th>方向</th>
                        <th>股数</th>
                        <th>价格</th>
                        <th>来源</th>
                      </tr>
                    </thead>
                    <tbody>
                      {detail.trades.last.map((t: Row, index: number) => (
                        <tr key={index}>
                          <td>{t.date}</td>
                          <td>{t.code}</td>
                          <td>{t.action === 'buy' ? '买入' : '卖出'}</td>
                          <td>{number(t.shares)}</td>
                          <td>{number(t.price, 3)}</td>
                          <td>{t.source}</td>
                        </tr>
                      ))}
                      {detail.trades.last.length === 0 && (
                        <tr>
                          <td colSpan={6}>还没有成交</td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </section>
            </>
          ) : (
            <section className="section">
              <p className="empty">这本账还没有开账。</p>
            </section>
          )}
        </>
      )}

      {section === 'unified' && (
        <>
          {rec ? (
            <>
              <section className="section">
                <div className="stats-band">
                  <div className="stat">
                    <span>数据截至</span>
                    <strong>{rec.as_of}</strong>
                    <small>{rec.action === 'hold' ? '持有，未触发带宽' : '需要调仓'}</small>
                  </div>
                  <div className="stat">
                    <span>账本净值</span>
                    <strong>{number(rec.nav, 2)}</strong>
                    <small>峰值 {number(rec.peak, 0)}</small>
                  </div>
                  <div className="stat">
                    <span>回撤</span>
                    <strong>{percent(rec.drawdown)}</strong>
                    <small>楼层 90%</small>
                  </div>
                  <div className="stat">
                    <span>风险腿目标</span>
                    <strong>{percent(rec.risky_target)}</strong>
                    <small>现在 {percent(rec.risky_now)}</small>
                  </div>
                  <div className="stat">
                    <span>闲置现金</span>
                    <strong>{number(rec.idle_cash, 2)}</strong>
                    <small>整手取整留下的</small>
                  </div>
                </div>
              </section>

              <section className="section">
                <div className="section-heading">
                  <h2>资产配置</h2>
                  <span>{rec.assets.length} 类 · 权重为 0 的也列出理由</span>
                </div>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>资产类别</th>
                        <th>目标权重</th>
                        <th>目标金额</th>
                        <th>为什么是这个权重</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rec.assets.map((row: Row) => (
                        <tr key={row.name}>
                          <td>
                            {row.name}
                            {row.code ? <small> · {row.code}</small> : null}
                          </td>
                          <td>{percent(row.weight)}</td>
                          <td>{number(row.value, 2)}</td>
                          <td>{row.note}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>

              <section className="section">
                <div className="section-heading">
                  <h2>四只 ETF 的目标与差额</h2>
                  <span>整手 100 份；差额为 0 才不需要下单</span>
                </div>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>代码</th>
                        <th>目标权重</th>
                        <th>现价</th>
                        <th>持有</th>
                        <th>目标份额</th>
                        <th>差额</th>
                        <th>费用</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rec.sleeves.map((row: Row) => (
                        <tr key={row.code}>
                          <td>{row.code}</td>
                          <td>{percent(row.weight)}</td>
                          <td>{number(row.price, 3)}</td>
                          <td>{number(row.held)}</td>
                          <td>{number(row.target_shares)}</td>
                          <td>{row.delta === 0 ? '—' : (row.delta > 0 ? '+' : '') + number(row.delta)}</td>
                          <td>{number(row.fee, 2)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>

              <section className="section">
                <div className="section-heading">
                  <h2>怎么落到账本</h2>
                  <span>本页只读；写入是命令行动作</span>
                </div>
                <p>
                  <code>python -B -m tools.daily_allocation --apply</code>
                  ：先卖后买，写成交，再按当日收盘重新记账。带宽内（{percent(rec.band, 0)}）不产生成交。
                </p>
                {rec.warnings.length > 0 && (
                  <ul>
                    {rec.warnings.map((text: string) => (
                      <li key={text}>{text}</li>
                    ))}
                  </ul>
                )}
              </section>
            </>
          ) : (
            <section className="section">
              <p>还没有建议数据。</p>
            </section>
          )}
        </>
      )}

      {section === 'recommend' && (
        <>
          <section className="section">
            <div className="section-heading">
              <h2>交易日</h2>
              <span>{days.length} 天有决策</span>
            </div>
            <div className="button-group">
              {days.map((d: Row) => (
                <button
                  key={d.panel_date}
                  className={day === d.panel_date ? 'primary' : ''}
                  onClick={() => setDay(d.panel_date)}
                >
                  {d.panel_date}
                </button>
              ))}
            </div>
          </section>

          <section className="section">
            <div className="section-heading">
              <h2>topK</h2>
              <span>名单固定 30 只，这里只决定显示前几只</span>
            </div>
            <div className="button-group">
              {TOP_CHOICES.map((k) => (
                <button key={k} className={topK === k ? 'primary' : ''} onClick={() => setTopK(k)}>
                  前 {k}
                </button>
              ))}
            </div>
          </section>

          {selected ? (
            <>
              <section className="section">
                <div className="stats-band">
                  <div className="stat">
                    <span>显示</span>
                    <strong>
                      前 {Math.min(topK, selected.names.length)} / {selected.names.length}
                    </strong>
                    <small>面板 {selected.panel_date}</small>
                  </div>
                  <div className="stat">
                    <span>分数截断</span>
                    <strong>{cutoff === null ? '—' : number(cutoff, 4)}</strong>
                    <small>低于此值入榜</small>
                  </div>
                  <div className="stat">
                    <span>换入 / 换出</span>
                    <strong>
                      {selected.entries.length} / {selected.exits.length}
                    </strong>
                    <small>
                      {selected.overlap === null ? '开仓日' : `与上一日重合 ${selected.overlap}`}
                    </small>
                  </div>
                  <div className="stat">
                    <span>决策</span>
                    <strong>{decisionNames[selected.decision] || selected.decision}</strong>
                    <small>
                      {selected.sessions_held === null ? '未记交易日数' : `已持有 ${selected.sessions_held} 个交易日`}
                    </small>
                  </div>
                  <div className="stat">
                    <span>本段收益</span>
                    <strong>
                      {selected.int_return === null ? '未记' : percent(selected.int_return)}
                    </strong>
                    <small>自 {selected.interval_from || '—'}</small>
                  </div>
                  <div className="stat">
                    <span>链式净值</span>
                    <strong>{number(selected.chained_nav, 4)}</strong>
                    <small>同池 {number(selected.chained_pool, 4)}</small>
                  </div>
                  <div className="stat">
                    <span>沪深300</span>
                    <strong>{number(selected.chained_hs300, 4)}</strong>
                    <small>
                      {selected.hs300_return === null
                        ? '本段未取到指数'
                        : `本段 ${percent(selected.hs300_return)}`}
                    </small>
                  </div>
                </div>
              </section>

              <section className="section">
                <div className="section-heading">
                  <h2>{selected.panel_date} 推荐</h2>
                  <span>分数越低越优先</span>
                </div>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>名次</th>
                        <th>代码</th>
                        <th>名称</th>
                        <th>分数</th>
                        <th>状态</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((item: Row, index: number) => (
                        <tr key={item.code}>
                          <td>{index + 1}</td>
                          <td>{item.code}</td>
                          <td>{item.name || '—'}</td>
                          <td>{item.score === null ? '—' : number(item.score, 4)}</td>
                          <td>{changeNames[item.change] || item.change}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {selected.exits.length > 0 && (
                  <p className="eyebrow">
                    换出：
                    {selected.exits
                      .map((x: Row) => `${x.code} ${x.name || ''}`.trim())
                      .join('、')}
                  </p>
                )}
              </section>
            </>
          ) : (
            <section className="section">
              <p className="empty">还没有产生过决策。</p>
            </section>
          )}
        </>
      )}

      {section === 'shadow' && (
        <>
          <section className="section">
            <div className="section-heading">
              <h2>当前状态</h2>
              <span>
                {number(shadow?.decided)} 次决策 / {number(shadow?.stale)} 次未推进
              </span>
            </div>
            <div className="definition-list">
              <div>
                <span>持有</span>
                <strong>
                  <Wallet size={14} /> {number(shadow?.state?.holdings)} 只
                </strong>
              </div>
              <div>
                <span>入场日</span>
                <strong>{shadow?.state?.entry_date || '—'}</strong>
              </div>
              <div>
                <span>最近面板</span>
                <strong>{shadow?.state?.last_panel_date || '—'}</strong>
              </div>
              <div>
                <span>记录行数</span>
                <strong>{number(shadow?.rows)}</strong>
              </div>
              <div>
                <span>链式净值</span>
                <strong>{number(shadow?.chained_nav, 4)}</strong>
              </div>
            </div>
          </section>

          <section className="section">
            <div className="section-heading">
              <h2>最近记录</h2>
              <span>只追加，不重写</span>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>记录时间</th>
                    <th>面板日期</th>
                    <th>状态</th>
                    <th>决策</th>
                    <th>持仓</th>
                    <th>入场日</th>
                    <th>说明</th>
                  </tr>
                </thead>
                <tbody>
                  {(shadow?.recent || [])
                    .slice()
                    .reverse()
                    .map((row: Row, index: number) => (
                      <tr key={index}>
                        <td>{String(row.recorded_at || '').slice(0, 19).replace('T', ' ')}</td>
                        <td>{row.panel_date || '—'}</td>
                        <td>{statusNames[row.status] || row.status}</td>
                        <td>{decisionNames[row.decision] || row.decision || '—'}</td>
                        <td>{number(row.holdings)}</td>
                        <td>{row.held_from || '—'}</td>
                        <td>{row.reason || '—'}</td>
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

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

export default function ExecutionPage({ fail }: { fail: (error: unknown) => void }) {
  const [books, setBooks] = useState<Row[]>([])
  const [name, setName] = useState('paper')
  const [detail, setDetail] = useState<Row | null>(null)
  const [shadow, setShadow] = useState<Row | null>(null)
  const [busy, setBusy] = useState(false)
  const [day, setDay] = useState('')

  async function load() {
    setBusy(true)
    try {
      const [list, sh] = await Promise.all([
        api<Row[]>('/execution/books'),
        api<Row>('/execution/shadow'),
      ])
      setBooks(list)
      setShadow(sh)
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

  return (
    <>
      <div className="page-heading">
        <div>
          <p className="eyebrow">执行层</p>
          <h1>三本账与 S2 影子</h1>
          <p>净值、持仓与影子决策都从账本文件与影子日志直接读出，数字与命令行一致。</p>
        </div>
        <button className="icon-button" aria-label="刷新" onClick={load} disabled={busy}>
          <RefreshCw size={18} className={busy ? 'spin' : ''} />
        </button>
      </div>

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

      <section className="section">
        <div className="section-heading">
          <h2>每日推荐股票</h2>
          <span>
            {(shadow?.days || []).length} 个交易日有决策 · 最新{' '}
            {(shadow?.days || []).slice(-1)[0]?.panel_date || '—'}
          </span>
        </div>
        <div className="button-group">
          {(shadow?.days || []).map((d: Row) => (
            <button
              key={d.panel_date}
              className={day === d.panel_date ? 'primary' : ''}
              onClick={() => setDay(d.panel_date)}
            >
              {String(d.panel_date).slice(5)}
            </button>
          ))}
        </div>
        {(() => {
          const selected = (shadow?.days || []).find((d: Row) => d.panel_date === day)
          if (!selected) return <p className="empty">还没有产生过决策。</p>
          const changeName: Record<string, string> = { held: '持有', in: '换入', new: '新开' }
          return (
            <>
              <div className="stats-band">
                <div className="stat">
                  <span>名单</span>
                  <strong>{selected.names.length} 只</strong>
                  <small>面板 {selected.panel_date}</small>
                </div>
                <div className="stat">
                  <span>换入</span>
                  <strong>{selected.entries.length}</strong>
                  <small>{selected.overlap === null ? '新开仓' : `与上一日重合 ${selected.overlap}`}</small>
                </div>
                <div className="stat">
                  <span>换出</span>
                  <strong>{selected.exits.length}</strong>
                  <small>{selected.held_from ? `持有自 ${selected.held_from}` : '本日为开仓日'}</small>
                </div>
              </div>
              {selected.exits.length > 0 && (
                <p className="eyebrow">
                  换出：
                  {selected.exits
                    .map((x: Row) => `${x.code} ${x.name || ''}`.trim())
                    .join('、')}
                </p>
              )}
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
                    {selected.names.map((item: Row, index: number) => (
                      <tr key={item.code}>
                        <td>{index + 1}</td>
                        <td>{item.code}</td>
                        <td>{item.name || '—'}</td>
                        <td>{item.score === null ? '—' : number(item.score, 4)}</td>
                        <td>{changeName[item.change] || item.change}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )
        })()}
      </section>

      <section className="section">
        <div className="section-heading">
          <h2>S2 影子运行</h2>
          <span>
            {number(shadow?.decided)} 次决策 / {number(shadow?.stale)} 次未推进 · 共{' '}
            {number(shadow?.rows)} 行
          </span>
        </div>
        <div className="definition-list">
          <div>
            <span>当前持有</span>
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
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>记录时间</th>
                <th>面板日期</th>
                <th>状态</th>
                <th>持仓</th>
                <th>入场日</th>
                <th>说明</th>
              </tr>
            </thead>
            <tbody>
              {(shadow?.recent || []).slice().reverse().map((row: Row, index: number) => (
                <tr key={index}>
                  <td>{String(row.recorded_at || '').slice(0, 19).replace('T', ' ')}</td>
                  <td>{row.panel_date || '—'}</td>
                  <td>{statusNames[row.status] || row.status}</td>
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
  )
}

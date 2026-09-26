import { useEffect, useRef, useState } from 'react'
import * as echarts from 'echarts/core'
import { CandlestickChart, LineChart, BarChart } from 'echarts/charts'
import {
  GridComponent,
  TooltipComponent,
  LegendComponent,
  DataZoomComponent,
} from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import { ChartCandlestick, ChartLine } from 'lucide-react'
import { api, number } from './api'

echarts.use([
  CandlestickChart,
  LineChart,
  BarChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  DataZoomComponent,
  CanvasRenderer,
])
type Bar = {
  date: string
  open: number | null
  close: number | null
  low: number | null
  high: number | null
  volume: number
  amount: number
}
type Batch = { id: string; entries: { code: string; basis: string; status: string }[] }

export default function MarketChart({ batch, datasetId }: { batch: Batch; datasetId?: string }) {
  const available = batch.entries.filter((e) => e.status === 'ready')
  const codes = [...new Set(available.map((e) => e.code))]
  const [code, setCode] = useState(codes[0] || '')
  const bases = available.filter((e) => e.code === code).map((e) => e.basis)
  const [preferredBasis, setBasis] = useState('unadjusted')
  const basis = bases.includes(preferredBasis) ? preferredBasis : bases[0]
  const [mode, setMode] = useState('candlestick')
  const [range, setRange] = useState(120)
  const [result, setResult] = useState<{ key: string; rows: Bar[] } | null>(null)
  const [error, setError] = useState('')
  const key = `${datasetId || 'history'}/${batch.id}/${code}/${basis}`
  const rows = result?.key === key ? result.rows : []
  const element = useRef<HTMLDivElement>(null)
  useEffect(() => {
    let active = true
    setError('')
    setResult(null)
    if (code && basis)
      api(
        datasetId
          ? `/market/datasets/${datasetId}/bars?code=${code}`
          : `/market/history/${batch.id}/bars?code=${code}&basis=${basis}`
      )
        .then((r) => {
          if (active) setResult({ key, rows: r.rows })
        })
        .catch((e) => {
          if (active) setError(String(e.message || e))
        })
    return () => {
      active = false
    }
  }, [key])
  useEffect(() => {
    if (!element.current || !rows.length) return
    const chart = echarts.init(element.current)
    const ma = (days: number) =>
      rows.map((_, i) =>
        i < days - 1
          ? null
          : rows.slice(i - days + 1, i + 1).some((r) => r.close === null)
            ? null
            : rows.slice(i - days + 1, i + 1).reduce((s, r) => s + r.close!, 0) / days
      )
    const axis = (gridIndex: number) => ({
      type: 'category',
      gridIndex,
      data: rows.map((r) => r.date),
      axisLabel: { show: gridIndex === 1, hideOverlap: true },
      axisTick: { show: gridIndex === 1 },
      axisLine: { lineStyle: { color: '#abb5ba' } },
    })
    chart.setOption({
      animation: false,
      tooltip: { trigger: 'axis', confine: true, axisPointer: { type: 'cross' } },
      legend: { top: 0, data: [mode === 'candlestick' ? '日K' : '收盘价', 'MA5', 'MA20'] },
      axisPointer: { link: [{ xAxisIndex: 'all' }] },
      grid: [
        { left: 68, right: 22, top: 40, height: '53%' },
        { left: 68, right: 22, top: '70%', height: '15%' },
      ],
      xAxis: [axis(0), axis(1)],
      yAxis: [
        { scale: true, splitLine: { lineStyle: { color: '#edf0f1' } } },
        {
          gridIndex: 1,
          name: '成交量(股)',
          splitNumber: 2,
          axisLabel: {
            formatter: (v: number) =>
              v >= 1e8 ? `${(v / 1e8).toFixed(1)}亿` : `${(v / 1e4).toFixed(0)}万`,
          },
          splitLine: { show: false },
        },
      ],
      dataZoom: [
        {
          type: 'inside',
          xAxisIndex: [0, 1],
          startValue: Math.max(0, rows.length - (range || rows.length)),
          endValue: rows.length - 1,
        },
        { type: 'slider', xAxisIndex: [0, 1], bottom: 0, height: 22 },
      ],
      series: [
        mode === 'candlestick'
          ? {
              name: '日K',
              type: 'candlestick',
              data: rows.map((r) =>
                [r.open, r.close, r.low, r.high].some((v) => v === null)
                  ? '-'
                  : [r.open, r.close, r.low, r.high]
              ),
              itemStyle: {
                color: '#ce424c',
                color0: '#16856b',
                borderColor: '#ce424c',
                borderColor0: '#16856b',
              },
            }
          : {
              name: '收盘价',
              type: 'line',
              showSymbol: false,
              data: rows.map((r) => r.close),
              itemStyle: { color: '#3971c6' },
            },
        ...[5, 20].map((n, i) => ({
          name: `MA${n}`,
          type: 'line',
          showSymbol: false,
          data: ma(n),
          itemStyle: { color: ['#bf852a', '#8764a0'][i] },
          lineStyle: { width: 1.3 },
        })),
        {
          name: '成交量(股)',
          type: 'bar',
          xAxisIndex: 1,
          yAxisIndex: 1,
          data: rows.map((r) => ({
            value: r.volume,
            itemStyle: {
              color:
                r.close === null || r.open === null
                  ? '#899399'
                  : r.close >= r.open
                    ? '#ce424c'
                    : '#16856b',
            },
          })),
        },
      ],
    })
    const observer = new ResizeObserver(() => chart.resize())
    observer.observe(element.current)
    return () => {
      observer.disconnect()
      chart.dispose()
    }
  }, [rows, mode, range])
  const last = rows.at(-1)
  const previous = rows.at(-2)
  const change =
    last?.close != null && previous?.close != null && previous.close > 0
      ? (last.close / previous.close - 1) * 100
      : null
  return (
    <section className="section market-chart-section">
      <div className="section-heading">
        <h2>历史行情</h2>
        <span>
          {datasetId ? '数据版本原始来源' : '本地快照'} · {batch.id}
        </span>
      </div>
      {!codes.length ? (
        <div className="empty">该批次没有已保存的日线</div>
      ) : (
        <>
          <div className="market-chart-controls">
            <label>
              股票
              <select aria-label="图表股票" value={code} onChange={(e) => setCode(e.target.value)}>
                {codes.map((c) => (
                  <option key={c}>{c}</option>
                ))}
              </select>
            </label>
            <label>
              价格口径
              <select
                aria-label="图表价格口径"
                value={basis}
                onChange={(e) => setBasis(e.target.value)}
              >
                {bases.map((b) => (
                  <option key={b} value={b}>
                    {b === 'hfq' ? '后复权' : '未复权'}
                  </option>
                ))}
              </select>
            </label>
            <label>
              时间范围
              <select
                aria-label="图表时间范围"
                value={range}
                onChange={(e) => setRange(Number(e.target.value))}
              >
                {[
                  [30, '近30条'],
                  [120, '近120条'],
                  [250, '近250条'],
                  [0, '全部'],
                ].map(([v, label]) => (
                  <option key={v} value={v}>
                    {label}
                  </option>
                ))}
              </select>
            </label>
            <div className="tabs" aria-label="行情图形">
              <button
                title="日K线"
                aria-label="日K线"
                aria-pressed={mode === 'candlestick'}
                className={mode === 'candlestick' ? 'active' : ''}
                onClick={() => setMode('candlestick')}
              >
                <ChartCandlestick size={18} />
              </button>
              <button
                title="收盘价曲线"
                aria-label="收盘价曲线"
                aria-pressed={mode === 'line'}
                className={mode === 'line' ? 'active' : ''}
                onClick={() => setMode('line')}
              >
                <ChartLine size={18} />
              </button>
            </div>
          </div>
          {last && (
            <div className="market-chart-summary">
              <strong>
                {code} · {number(last.close, 2)}
              </strong>
              <span className={change !== null && change >= 0 ? 'market-up' : 'market-down'}>
                {change === null ? '—' : `${change >= 0 ? '+' : ''}${change.toFixed(2)}%`}
              </span>
              <span>末条日线 {last.date}</span>
              <span>{rows.length.toLocaleString()} 条</span>
              <span>{basis === 'hfq' ? '后复权研究价格 · 非实际报价' : '未复权历史价格'}</span>
            </div>
          )}
          {error ? (
            <div role="alert" className="empty">
              {error}
            </div>
          ) : !result ? (
            <div className="empty">正在读取本地日线…</div>
          ) : !rows.length ? (
            <div className="empty">没有日线记录</div>
          ) : null}
          <div
            ref={element}
            className="market-canvas"
            data-testid="market-chart"
            role="img"
            aria-label={`${code} ${basis === 'hfq' ? '后复权' : '未复权'}历史行情`}
          />
        </>
      )}
    </section>
  )
}

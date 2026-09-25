import { useEffect, useRef } from 'react'
import * as echarts from 'echarts/core'
import { LineChart } from 'echarts/charts'
import { GridComponent, LegendComponent, TooltipComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'

echarts.use([LineChart, GridComponent, LegendComponent, TooltipComponent, CanvasRenderer])

const colors = ['#167b64', '#bd7b29', '#3971c6', '#b45763', '#8764a0', '#708574', '#555d68']
export default function Chart({
  series,
  x,
  height = 270,
  percentage = false,
}: {
  series: { name: string; values: (number | null)[] }[]
  x: (string | number)[]
  height?: number
  percentage?: boolean
}) {
  const element = useRef<HTMLDivElement>(null)
  const instance = useRef<ReturnType<typeof echarts.init> | null>(null)
  const signature = JSON.stringify({ series, x, percentage })
  useEffect(() => {
    if (!element.current) return
    const chart = echarts.init(element.current)
    instance.current = chart
    const observer = new ResizeObserver(() => chart.resize())
    observer.observe(element.current)
    return () => {
      observer.disconnect()
      chart.dispose()
      instance.current = null
    }
  }, [])
  useEffect(() => {
    const chart = instance.current
    if (!chart) return
    chart.setOption({
      animation: false,
      color: colors,
      tooltip: { trigger: 'axis', confine: true },
      legend: { type: 'scroll', top: 0, textStyle: { color: '#5d656b', fontSize: 11 } },
      grid: { left: 58, right: 18, top: 42, bottom: 36 },
      xAxis: {
        type: 'category',
        data: x,
        boundaryGap: false,
        axisLine: { lineStyle: { color: '#d8dddf' } },
        axisLabel: { color: '#6d767c', fontSize: 10, hideOverlap: true },
      },
      yAxis: {
        type: 'value',
        scale: true,
        splitLine: { lineStyle: { color: '#edf0f1' } },
        axisLabel: {
          fontSize: 10,
          color: '#6d767c',
          formatter: (v: number) =>
            percentage ? `${(v * 100).toFixed(1)}%` : Number(v.toPrecision(3)).toString(),
        },
      },
      series: series.map((s) => ({
        name: s.name,
        data: s.values,
        type: 'line',
        showSymbol: false,
        connectNulls: false,
        lineStyle: { width: 2 },
      })),
    })
  }, [signature])
  return <div className="chart" ref={element} style={{ height }} data-testid="chart" />
}

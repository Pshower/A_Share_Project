import { lazy, Suspense, useState } from 'react'
import { number } from './api'
const Chart = lazy(() => import('./Chart'))
type Row = Record<string, any>

export default function TrainingDiagnostics({ metrics }: { metrics: Row[] }) {
  const [view, setView] = useState('optimization')
  const last = metrics.at(-1)?.metrics || {}
  const x = metrics.map((m) => m.timesteps)
  function plot(title: string, fields: [string, string][], percentage = false) {
    return (
      <div key={title}>
        <h3>{title}</h3>
        <Suspense fallback={<div className="empty">读取图表…</div>}>
          <Chart
            x={x}
            percentage={percentage}
            series={fields.map(([key, name]) => ({
              name,
              values: metrics.map((m) => m.metrics[key] ?? null),
            }))}
          />
        </Suspense>
      </div>
    )
  }
  return (
    <>
      <div className="tabs">
        {[
          ['optimization', '优化诊断'],
          ['portfolio', '组合行为'],
          ['resources', '耗时与资源'],
        ].map(([key, label]) => (
          <button key={key} className={view === key ? 'active' : ''} onClick={() => setView(key)}>
            {label}
          </button>
        ))}
      </div>
      <div className="stats-band">
        <div className="stat">
          <span>实际优化器步数</span>
          <strong>{number(last['optimizer/steps_max'])}</strong>
        </div>
        <div className="stat">
          <span>最后一次裁剪后梯度范数</span>
          <strong>{number(last['optimizer/last_post_clip_grad_norm'], 4)}</strong>
        </div>
        <div className="stat">
          <span>当前学习率</span>
          <strong>
            {last['train/learning_rate'] == null
              ? '—'
              : Number(last['train/learning_rate']).toExponential(2)}
          </strong>
        </div>
      </div>
      <div className="two-charts">
        {view === 'optimization' && (
          <>
            {plot('价值损失', [['train/value_loss', 'Value loss']])}
            {plot('策略更新', [
              ['train/approx_kl', 'KL'],
              ['train/clip_fraction', 'Clip fraction'],
            ])}
            {plot('策略损失与价值解释度', [
              ['train/policy_gradient_loss', 'Policy loss'],
              ['train/explained_variance', 'Explained variance'],
            ])}
            {plot('每维潜在动作熵', [
              ['train/entropy_per_dimension', 'Gaussian entropy / dimension'],
            ])}
          </>
        )}
        {view === 'portfolio' && (
          <>
            {plot(
              '目标与实际现金',
              [
                ['portfolio/target_cash', '目标现金'],
                ['portfolio/actual_cash', '实际现金'],
              ],
              true
            )}
            {plot(
              '换手与目标变动',
              [
                ['portfolio/gross_turnover', '实际总换手'],
                ['portfolio/target_change_l1', '目标权重 L1 变化'],
              ],
              true
            )}
            {plot(
              '映射与仓位限制',
              [
                ['portfolio/action_saturation', '潜在动作饱和'],
                ['portfolio/cap_fraction', '达到仓位上限比例'],
              ],
              true
            )}
            {plot('每步平均手续费', [['portfolio/fees', '手续费']])}
          </>
        )}
        {view === 'resources' && (
          <>
            {plot('采样与更新阶段耗时', [
              ['time/rollout_seconds', '采样秒'],
              ['time/update_wall_seconds', '更新墙钟秒'],
            ])}
            {plot('CUDA 已分配显存', [['resource/cuda_allocated_mb', 'MB']])}
            {plot('有效股票数', [['portfolio/eligible_stocks', '平均有效股票']])}
            {plot('累计验证耗时', [['time/validation_seconds_total', '秒']])}
          </>
        )}
      </div>
      {view === 'portfolio' && (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>订单限制原因</th>
                <th>累计次数（原因可重叠）</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(last)
                .filter(([key]) => key.startsWith('orders_total/'))
                .map(([key, value]) => (
                  <tr key={key}>
                    <td>{key.slice(13)}</td>
                    <td>{number(value)}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  )
}

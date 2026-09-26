export const advancedDefaults = {
  learning_rate_schedule: 'constant',
  learning_rate_end_factor: 0.1,
  gamma: 0.99,
  gae_lambda: 0.95,
  clip_range: 0.2,
  target_kl: 0.03,
  ent_coef: 0,
  vf_coef: 0.5,
  max_grad_norm: 0.5,
  hidden_sizes: [128, 64],
  max_validation_drawdown: null as number | null,
  max_validation_turnover: null as number | null,
  min_validation_steps: 20,
}

export default function PpoSettings({
  value,
  onChange,
}: {
  value: Record<string, any>
  onChange: (patch: Record<string, any>) => void
}) {
  const fields: [string, string, number, number, number][] = [
    ['gamma', '折扣系数 γ', 0, 1, 0.005],
    ['gae_lambda', 'GAE λ', 0, 1, 0.01],
    ['clip_range', 'PPO 裁剪范围', 0.001, 0.5, 0.01],
    ['target_kl', '目标 KL', 0.001, 0.2, 0.005],
    ['ent_coef', '熵系数', 0, 0.1, 0.0001],
    ['vf_coef', '价值损失系数', 0, 2, 0.1],
    ['max_grad_norm', '梯度范数上限', 0.01, 10, 0.1],
    ['min_validation_steps', '选模最少验证记录', 2, 2000, 1],
  ]
  return (
    <details className="ppo-settings">
      <summary>PPO 高级参数与选模约束</summary>
      <div className="field-grid">
        <label className="field">
          学习率日程
          <select
            aria-label="学习率日程"
            value={value.learning_rate_schedule}
            onChange={(e) => onChange({ learning_rate_schedule: e.target.value })}
          >
            <option value="constant">恒定</option>
            <option value="linear">线性衰减</option>
          </select>
        </label>
        <label className="field">
          末端学习率比例
          <input
            aria-label="末端学习率比例"
            type="number"
            min={0}
            max={1}
            step={0.05}
            disabled={value.learning_rate_schedule !== 'linear'}
            value={value.learning_rate_end_factor}
            onChange={(e) => onChange({ learning_rate_end_factor: Number(e.target.value) })}
          />
        </label>
        {fields.map(([key, label, min, max, step]) => (
          <label className="field" key={key}>
            {label}
            <input
              aria-label={label}
              type="number"
              min={min}
              max={max}
              step={step}
              value={value[key]}
              onChange={(e) => onChange({ [key]: Number(e.target.value) })}
            />
          </label>
        ))}
        {[0, 1].map((i) => (
          <label className="field" key={i}>
            隐藏层 {i + 1}
            <input
              aria-label={`隐藏层 ${i + 1}`}
              type="number"
              min={8}
              max={512}
              value={value.hidden_sizes[i]}
              onChange={(e) =>
                onChange({
                  hidden_sizes: value.hidden_sizes.map((v: number, j: number) =>
                    i === j ? Number(e.target.value) : v
                  ),
                })
              }
            />
          </label>
        ))}
      </div>
      <div className="field-grid">
        <div>
          <label className="check-label">
            <input
              type="checkbox"
              checked={value.max_validation_drawdown != null}
              onChange={(e) => onChange({ max_validation_drawdown: e.target.checked ? 0.2 : null })}
            />
            限制验证回撤幅度
          </label>
          <input
            aria-label="验证回撤上限"
            type="number"
            min={0.001}
            max={1}
            step={0.01}
            disabled={value.max_validation_drawdown == null}
            value={value.max_validation_drawdown ?? ''}
            onChange={(e) => onChange({ max_validation_drawdown: Number(e.target.value) })}
          />
        </div>
        <div>
          <label className="check-label">
            <input
              type="checkbox"
              checked={value.max_validation_turnover != null}
              onChange={(e) => onChange({ max_validation_turnover: e.target.checked ? 0.5 : null })}
            />
            限制验证日均总换手
          </label>
          <input
            aria-label="验证换手上限"
            type="number"
            min={0.001}
            max={10}
            step={0.01}
            disabled={value.max_validation_turnover == null}
            value={value.max_validation_turnover ?? ''}
            onChange={(e) => onChange({ max_validation_turnover: Number(e.target.value) })}
          />
        </div>
      </div>
    </details>
  )
}

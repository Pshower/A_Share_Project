type Model = Record<string, any>

export const webActionModes = ['target', 'stock_delta', 'stock_trade']
export const predictionActionModes = ['target', 'stock_delta']
export const dailyPlanActionModes = ['target']

export const actionModeNames: Record<string, string> = {
  target: '目标权重',
  stock_delta: '股票增量',
  stock_trade: '整手交易',
  stock_priority: '排序动作（Web 不支持）',
  timing_control: '调仓控制（Web 不支持）',
}

export function actionMode(model: Model | undefined): string {
  return model?.action_mode || model?.metadata?.contract?.action_mode || 'target'
}

export function supportsAction(model: Model | undefined, allowed: string[]): boolean {
  return Boolean(model?.valid && model?.trained && allowed.includes(actionMode(model)))
}

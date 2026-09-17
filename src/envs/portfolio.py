import numpy as np

class Portfolio:
    """
    管理现金、持仓及资产估值
    """
    def __init__(self, initial_capital: float = 1e6, stock_codes: list = None):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.stock_codes = stock_codes or []
        self.n_stocks = len(self.stock_codes)
        # 持仓股数
        self.positions = np.zeros(self.n_stocks, dtype=np.float64)   # 股数
        # 持仓成本（用于计算盈亏，可忽略）
        self.cost_basis = np.zeros(self.n_stocks)
    
    def reset(self):
        self.cash = self.initial_capital
        self.positions = np.zeros(self.n_stocks)
        self.cost_basis = np.zeros(self.n_stocks)
    
    def value(self, prices: np.ndarray) -> float:
        """计算当前组合总资产 = 现金 + 持仓市值"""
        return self.cash + np.sum(self.positions * prices)
    
    def update_positions(self, new_positions: np.ndarray, prices: np.ndarray, 
                         transaction_cost: float = 0.0) -> Tuple[float, float]:
        """
        根据目标股数更新持仓，返回交易成本和交易金额（正为买入，负为卖出）
        假设市价成交，无滑点（可由broker处理）
        """
        # 计算买卖股数差值
        delta = new_positions - self.positions
        # 交易金额（不含费用）
        trade_amount = np.sum(delta * prices)
        # 交易费用（假设按交易金额比例）
        fee = abs(trade_amount) * transaction_cost
        # 更新现金
        self.cash -= (trade_amount + fee)
        # 更新持仓
        self.positions = new_positions.copy()
        # 更新成本（简单起见，加权平均）
        # 此处略，可后续实现
        return fee, trade_amount
    
    def apply_weights(self, weights: np.ndarray, prices: np.ndarray, 
                      transaction_cost: float = 0.0) -> Tuple[float, float]:
        """
        根据目标权重（占总资产比例）调整仓位
        weights: (n_stocks,)，和为1（含现金？这里只处理股票部分）
        假设现金为剩余部分
        """
        total_asset = self.value(prices)
        target_values = weights * total_asset
        target_shares = target_values / prices
        return self.update_positions(target_shares, prices, transaction_cost)
    
    def get_weights(self, prices: np.ndarray) -> np.ndarray:
        """计算当前股票权重（占总资产比例）"""
        total = self.value(prices)
        if total == 0:
            return np.zeros(self.n_stocks)
        return (self.positions * prices) / total
    
    def get_cash_ratio(self, prices: np.ndarray) -> float:
        return self.cash / self.value(prices) if self.value(prices) > 0 else 1.0
# src/data/data_preview.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class DataPreview:
    """
    股票数据预览与质量检查
    - 统计每只股票的数据长度、时间范围、缺失值
    - 计算价格范围、波动率、最大回撤、夏普比率等
    - 生成汇总报告（DataFrame）并保存为CSV
    """

    def __init__(self, raw_dir: Path, stock_list_file: Path):
        self.raw_dir = Path(raw_dir)
        self.stock_list = self._load_stock_list(stock_list_file)

    def _load_stock_list(self, file_path: Path) -> List[str]:
        """从文本文件读取股票代码"""
        with open(file_path, 'r', encoding='utf-8') as f:
            codes = [line.strip() for line in f if line.strip()]
        return codes

    def _find_hfq_file(self, code: str) -> Optional[Path]:
        """查找后复权数据文件"""
        # 匹配模式：{code}_data_*_hfq.csv
        candidates = list(self.raw_dir.glob(f"{code}_data_*_hfq.csv"))
        if not candidates:
            # 尝试无hfq后缀或其它命名，但建议使用后复权
            candidates = list(self.raw_dir.glob(f"{code}_data_*.csv"))
            # 过滤掉可能包含 "hfq" 的已经捕获，所以这里可以保留
        if candidates:
            return candidates[0]  # 取第一个
        return None

    def _read_stock(self, code: str) -> Optional[pd.DataFrame]:
        """读取单只股票数据，返回DataFrame（包含日期、收盘价等）"""
        file_path = self._find_hfq_file(code)
        if file_path is None:
            return None
        df = pd.read_csv(file_path, parse_dates=['日期'])
        # 统一列名
        df.rename(columns={'日期': 'date', '收盘': 'close', '开盘': 'open',
                           '最高': 'high', '最低': 'low', '成交量': 'volume',
                           '成交额': 'amount', '涨跌幅': 'pct_chg'}, inplace=True)
        df = df.sort_values('date').reset_index(drop=True)
        return df

    def calculate_metrics(self, df: pd.DataFrame, code: str) -> Dict:
        """
        计算单只股票的统计指标
        """
        # 确保有收盘价
        close = df['close']
        if close.isnull().all():
            return {}

        # 基础信息
        n_days = len(df)
        start_date = df['date'].min()
        end_date = df['date'].max()
        missing_ratio = close.isnull().mean()

        # 价格范围
        price_min = close.min()
        price_max = close.max()
        price_mean = close.mean()
        price_std = close.std()

        # 收益率（对数收益率更稳定）
        returns = close.pct_change().dropna()
        if len(returns) == 0:
            return {}
        log_returns = np.log(close / close.shift(1)).dropna()

        # 波动率（日度、年化）
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252)

        # 偏度、峰度
        skewness = returns.skew()
        kurtosis = returns.kurtosis()

        # 最大回撤
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # 夏普比率（假设无风险利率0）
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else np.nan

        # 年化收益率
        total_return = (close.iloc[-1] / close.iloc[0]) - 1 if close.iloc[0] != 0 else np.nan
        years = (end_date - start_date).days / 365.25
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else np.nan

        # 换手率均值（如果有）
        turnover_mean = df['换手率'].mean() if '换手率' in df.columns else np.nan

        # 成交量均值
        volume_mean = df['volume'].mean() if 'volume' in df.columns else np.nan

        return {
            'code': code,
            'start_date': start_date,
            'end_date': end_date,
            'n_days': n_days,
            'missing_ratio': missing_ratio,
            'price_min': price_min,
            'price_max': price_max,
            'price_mean': price_mean,
            'price_std': price_std,
            'daily_vol': daily_vol,
            'annual_vol': annual_vol,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'annual_return': annual_return,
            'total_return': total_return,
            'turnover_mean': turnover_mean,
            'volume_mean': volume_mean,
        }

    def generate_report(self) -> pd.DataFrame:
        """
        遍历所有股票，生成汇总报告
        """
        records = []
        for code in self.stock_list:
            df = self._read_stock(code)
            if df is None:
                print(f"⚠️ 未找到股票 {code} 的数据文件，跳过")
                continue
            metrics = self.calculate_metrics(df, code)
            if metrics:
                records.append(metrics)
            else:
                print(f"⚠️ 股票 {code} 数据不足，跳过")
        report_df = pd.DataFrame(records)
        # 排序（按代码）
        report_df = report_df.sort_values('code').reset_index(drop=True)
        return report_df

    def save_report(self, report_df: pd.DataFrame, output_path: Path):
        """保存报告为CSV"""
        report_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"📊 报告已保存至 {output_path}")

    def print_summary(self, report_df: pd.DataFrame):
        """打印简要统计摘要"""
        print("\n" + "="*60)
        print("数据预览报告摘要")
        print("="*60)
        print(f"股票总数: {len(report_df)}")
        print(f"平均交易日数: {report_df['n_days'].mean():.0f}")
        print(f"数据完整率: {(1 - report_df['missing_ratio'].mean())*100:.2f}%")
        print(f"平均年化收益率: {report_df['annual_return'].mean()*100:.2f}%")
        print(f"平均年化波动率: {report_df['annual_vol'].mean()*100:.2f}%")
        print(f"平均夏普比率: {report_df['sharpe_ratio'].mean():.3f}")
        print(f"平均最大回撤: {report_df['max_drawdown'].mean()*100:.2f}%")
        print("-"*60)
        # 显示前5行
        print(report_df.head().to_string(index=False))


if __name__ == "__main__":
    # 路径配置
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    RAW_DIR = PROJECT_ROOT / "data" / "raw"
    STOCK_LIST_FILE = PROJECT_ROOT / "data" / "stock_list" / "hs300_20260629.txt"
    OUTPUT_REPORT = PROJECT_ROOT / "data" / "stock_list" / "data_preview_report.csv"

    preview = DataPreview(raw_dir=RAW_DIR, stock_list_file=STOCK_LIST_FILE)
    report = preview.generate_report()

    if report is not None and not report.empty:
        preview.save_report(report, OUTPUT_REPORT)
        preview.print_summary(report)
    else:
        print("❌ 未生成任何有效报告，请检查数据文件是否存在。")
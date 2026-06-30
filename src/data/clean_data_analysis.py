# src/data/clean_data_analysis.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class CleanDataAnalyzer:
    """
    分析清洗后的特征数据（Parquet格式）
    - 统计每只股票的特征缺失率、异常值
    - 计算特征间的相关性（可选）
    - 检查标准化后特征分布
    - 生成汇总统计报告
    """

    def __init__(self, clean_dir: Path):
        self.clean_dir = Path(clean_dir)

    def load_clean_data(self) -> pd.DataFrame:
        """读取所有股票的清洗后特征并合并"""
        all_dfs = []
        parquet_files = list(self.clean_dir.glob("*_features.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"在 {self.clean_dir} 中未找到任何 Parquet 文件")
        for f in parquet_files:
            df = pd.read_parquet(f)
            all_dfs.append(df)
        combined = pd.concat(all_dfs, ignore_index=True)
        return combined

    def analyze_per_stock(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        按股票统计：
        - 记录数、特征缺失率（标准化后不应有缺失）
        - 检查是否存在无穷值（inf/-inf）
        - 特征均值（应接近0）、标准差（应接近1）——标准化验证
        - 异常值比例（绝对值>3）
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # 排除 code 列（如果是数值型但不应作为特征）
        numeric_cols = [c for c in numeric_cols if c not in ['code']]

        records = []
        for code in df['code'].unique():
            sub = df[df['code'] == code]
            n_rows = len(sub)

            # 检查是否有缺失（标准化后不应有）
            missing_ratio = sub[numeric_cols].isnull().mean().mean()

            # 检查无穷值
            inf_count = sub[numeric_cols].isin([np.inf, -np.inf]).sum().sum()

            # 均值与标准差
            means = sub[numeric_cols].mean()
            stds = sub[numeric_cols].std()

            # 异常值比例（绝对值>3）
            outlier_ratio = (np.abs(sub[numeric_cols]) > 3).mean().mean()

            # 特征波动性：每日特征变化的平均绝对值（衡量稳定性）
            # 只取日期时间特征，排除非时间序列
            if 'date' in sub.columns:
                sub_sorted = sub.sort_values(['code', 'date'])
                # 对所有数值特征计算差分绝对值的均值
                diff_abs = sub_sorted.groupby('code')[numeric_cols].diff().abs().mean().mean()
            else:
                diff_abs = np.nan

            records.append({
                'code': code,
                'rows': n_rows,
                'missing_ratio': missing_ratio,
                'inf_count': inf_count,
                'mean_feature': means.mean(),
                'std_feature': stds.mean(),
                'outlier_ratio': outlier_ratio,
                'avg_abs_diff': diff_abs
            })
        return pd.DataFrame(records)

    def analyze_overall_distribution(self, df: pd.DataFrame) -> Dict:
        """总体统计：所有特征的整体分布"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ['code']]
        all_values = df[numeric_cols].values.flatten()
        all_values = all_values[~np.isnan(all_values)]

        stats = {
            'total_cells': len(all_values),
            'mean': np.mean(all_values),
            'std': np.std(all_values),
            'min': np.min(all_values),
            'max': np.max(all_values),
            'percentiles': {
                1: np.percentile(all_values, 1),
                25: np.percentile(all_values, 25),
                50: np.percentile(all_values, 50),
                75: np.percentile(all_values, 75),
                99: np.percentile(all_values, 99)
            },
            'outlier_ratio': np.mean(np.abs(all_values) > 3)
        }
        return stats

    def generate_report(self, save_path: Path = None) -> pd.DataFrame:
        """生成完整报告并保存为CSV"""
        df = self.load_clean_data()
        print(f"✅ 加载了 {len(df)} 条记录，涵盖 {df['code'].nunique()} 只股票")

        # 按股票分析
        per_stock = self.analyze_per_stock(df)
        overall = self.analyze_overall_distribution(df)

        # 打印摘要
        print("\n" + "="*60)
        print("清洗后数据质量报告")
        print("="*60)
        print(f"总记录数: {len(df)}")
        print(f"股票数: {df['code'].nunique()}")
        print(f"所有特征均值: {overall['mean']:.6f} (应接近0)")
        print(f"所有特征标准差: {overall['std']:.6f} (应接近1)")
        print(f"异常值比例 (>3σ): {overall['outlier_ratio']*100:.2f}%")
        print("分位数: 1%={:.3f}, 25%={:.3f}, 50%={:.3f}, 75%={:.3f}, 99%={:.3f}".format(
            overall['percentiles'][1], overall['percentiles'][25],
            overall['percentiles'][50], overall['percentiles'][75],
            overall['percentiles'][99]
        ))

        # 每只股票的统计摘要
        print("\n每只股票统计摘要（前10行）：")
        print(per_stock.head(10).to_string(index=False))

        # 检查标准化是否合格：均值是否接近0，标准差是否接近1
        mean_means = per_stock['mean_feature'].mean()
        mean_stds = per_stock['std_feature'].mean()
        print(f"\n标准化质量检查：")
        print(f"  所有特征平均均值: {mean_means:.6f} (理想为0)")
        print(f"  所有特征平均标准差: {mean_stds:.6f} (理想为1)")

        # 保存报告
        if save_path:
            per_stock.to_csv(save_path, index=False, encoding='utf-8-sig')
            print(f"📊 报告已保存至 {save_path}")

        return per_stock


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
    REPORT_PATH = PROJECT_ROOT / "data" / "stock_list" / "clean_data_quality_report.csv"

    analyzer = CleanDataAnalyzer(CLEAN_DIR)
    report = analyzer.generate_report(save_path=REPORT_PATH)
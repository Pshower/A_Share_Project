# src/data/clean_data_analysis.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class CleanDataAnalyzer:
    """
    分析清洗后的特征数据（Parquet格式）
    - 优先加载 all_stocks_features.parquet（合并长表）
    - 若不存在则加载单个股票 Parquet 文件并合并
    - 统计每只股票的数据长度、日期范围、缺失率
    - 检查标准化质量（均值、标准差）
    - 检测异常值比例（绝对值>3）
    - 生成汇总报告
    """

    def __init__(self, clean_dir: Path):
        self.clean_dir = Path(clean_dir)
        self.combined_file = self.clean_dir / "all_stocks_features.parquet"
        self.panel_file = self.clean_dir / "panel_data.parquet"

    def load_clean_data(self) -> pd.DataFrame:
        """
        加载清洗后的数据
        优先使用 all_stocks_features.parquet，若不存在则合并单个股票文件
        """
        if self.combined_file.exists():
            df = pd.read_parquet(self.combined_file)
            print(f"✅ 加载合并文件: {self.combined_file.name}, 共 {len(df)} 条记录")
            return df
        else:
            # 回退到单个文件合并
            print("⚠️ 未找到合并文件，尝试加载单个股票特征文件...")
            all_dfs = []
            parquet_files = list(self.clean_dir.glob("*_features.parquet"))
            if not parquet_files:
                raise FileNotFoundError(f"在 {self.clean_dir} 中未找到任何 Parquet 文件")
            for f in parquet_files:
                df = pd.read_parquet(f)
                all_dfs.append(df)
            combined = pd.concat(all_dfs, ignore_index=True)
            print(f"✅ 合并了 {len(all_dfs)} 个股票文件，共 {len(combined)} 条记录")
            return combined

    def analyze_per_stock(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        按股票统计：
        - 记录数、日期范围
        - 特征缺失率（应无缺失）
        - 无穷值数量
        - 特征均值和标准差（验证标准化效果）
        - 异常值比例（|z|>3）
        - 平均绝对差分（波动稳定性）
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # 排除 code 列（如果是数值型但不应作为特征）
        numeric_cols = [c for c in numeric_cols if c not in ['code']]

        records = []
        for code in df['code'].unique():
            sub = df[df['code'] == code]
            n_rows = len(sub)

            # 日期范围
            start_date = sub['date'].min()
            end_date = sub['date'].max()

            # 缺失值（标准化后不应有）
            missing_ratio = sub[numeric_cols].isnull().mean().mean()

            # 无穷值
            inf_count = sub[numeric_cols].isin([np.inf, -np.inf]).sum().sum()

            # 均值和标准差（越接近0和1越好）
            means = sub[numeric_cols].mean()
            stds = sub[numeric_cols].std()

            # 异常值比例 (|z| > 3)
            outlier_ratio = (np.abs(sub[numeric_cols]) > 3).mean().mean()

            # 特征时间序列平均绝对差分（稳定性）
            if 'date' in sub.columns:
                sub_sorted = sub.sort_values(['code', 'date'])
                # 对所有数值特征计算差分绝对值的均值
                diff_abs = sub_sorted.groupby('code')[numeric_cols].diff().abs().mean().mean()
            else:
                diff_abs = np.nan

            records.append({
                'code': code,
                'rows': n_rows,
                'start_date': start_date,
                'end_date': end_date,
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

    def check_panel_data(self) -> Optional[Dict]:
        """如果面板数据存在，输出其形状信息"""
        if self.panel_file.exists():
            panel = pd.read_parquet(self.panel_file)
            return {
                'shape': panel.shape,
                'n_dates': len(panel.index),
                'n_stocks': panel.columns.get_level_values(0).nunique(),
                'n_features': panel.columns.get_level_values(1).nunique()
            }
        return None

    def generate_report(self, save_path: Optional[Path] = None) -> pd.DataFrame:
        """生成完整报告并打印摘要"""
        df = self.load_clean_data()
        print(f"总记录数: {len(df)}, 涵盖 {df['code'].nunique()} 只股票")

        # 按股票统计
        per_stock = self.analyze_per_stock(df)
        overall = self.analyze_overall_distribution(df)

        # 打印总体统计
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

        # 标准化质量检查
        mean_means = per_stock['mean_feature'].mean()
        mean_stds = per_stock['std_feature'].mean()
        print(f"\n标准化质量检查：")
        print(f"  所有特征平均均值: {mean_means:.6f} (理想为0)")
        print(f"  所有特征平均标准差: {mean_stds:.6f} (理想为1)")

        # 检查是否有股票记录数过少（如 < 100）
        short_stocks = per_stock[per_stock['rows'] < 100]
        if not short_stocks.empty:
            print(f"\n⚠️ 以下股票交易日数少于100，建议检查或剔除：")
            print(short_stocks[['code', 'rows']].to_string(index=False))

        # 检查是否有异常股票（标准差偏离1太多）
        std_deviation = per_stock['std_feature']
        abnormal_std = per_stock[(std_deviation < 0.5) | (std_deviation > 1.5)]
        if not abnormal_std.empty:
            print(f"\n⚠️ 以下股票特征标准差异常（<0.5 或 >1.5），可能存在标准化问题：")
            print(abnormal_std[['code', 'std_feature']].to_string(index=False))

        # 面板数据信息（如果有）
        panel_info = self.check_panel_data()
        if panel_info:
            print(f"\n📊 面板数据信息:")
            print(f"  形状: {panel_info['shape']}")
            print(f"  日期数: {panel_info['n_dates']}")
            print(f"  股票数: {panel_info['n_stocks']}")
            print(f"  特征数: {panel_info['n_features']}")

        # 保存报告
        if save_path:
            per_stock.to_csv(save_path, index=False, encoding='utf-8-sig')
            print(f"\n📊 详细报告已保存至 {save_path}")

        return per_stock


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
    REPORT_PATH = PROJECT_ROOT / "data" / "stock_list" / "clean_data_quality_report.csv"

    analyzer = CleanDataAnalyzer(CLEAN_DIR)
    report = analyzer.generate_report(save_path=REPORT_PATH)
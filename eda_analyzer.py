import pandas as pd
import numpy as np
from scipy import stats

class EDAnalyzer:
    def __init__(self, df, target_col='target'):
        self.df = df
        self.target_col = target_col
        self.eda_results = {}
    
    def basic_statistics(self):
        """基本统计描述"""
        # 确保目标列存在
        if self.target_col not in self.df.columns:
            raise ValueError(f"目标列 '{self.target_col}' 不存在于数据中")
        
        # 计算统计信息
        stats_dict = {
            'shape': self.df.shape,
            'dtypes': self.df.dtypes.astype(str).to_dict(),  # 转换为字符串
            'descriptive_stats': self.df.describe().to_dict(),
            'target_distribution': self.df[self.target_col].value_counts().to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'unique_values': {col: self.df[col].nunique() for col in self.df.columns}
        }
        
        # 添加目标分布百分比
        target_counts = self.df[self.target_col].value_counts()
        total = len(self.df)
        stats_dict['target_percentage'] = {
            int(k): (v / total * 100) for k, v in target_counts.items()
        }
        
        self.eda_results['basic_statistics'] = stats_dict
        return self
    
    def correlation_analysis(self):
        """相关性分析"""
        numeric_df = self.df.select_dtypes(include=[np.number])
        correlation_matrix = numeric_df.corr()
        
        top_correlations = []
        corr_pairs = correlation_matrix.abs().unstack()
        corr_pairs = corr_pairs[corr_pairs < 1].sort_values(ascending=False)
        
        for pair, corr_value in corr_pairs.head(10).items():
            if pair[0] != self.target_col and pair[1] != self.target_col:
                continue
            top_correlations.append({
                'feature1': pair[0],
                'feature2': pair[1],
                'correlation': float(corr_value)
            })
        
        self.eda_results['correlation'] = {
            'matrix': correlation_matrix.to_dict(),
            'top_features_with_target': top_correlations
        }
        return self
    
    def distribution_analysis(self):
        """分布分析"""
        distributions = {}
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols[:10]:  # 限制数量避免过多计算
            data = self.df[col].dropna()
            shapiro_test = stats.shapiro(data)
            distributions[col] = {
                'mean': float(data.mean()),
                'std': float(data.std()),
                'skewness': float(data.skew()),
                'kurtosis': float(data.kurtosis()),
                'normality_test': {
                    'statistic': float(shapiro_test.statistic),
                    'p_value': float(shapiro_test.pvalue)
                }
            }
        
        self.eda_results['distributions'] = distributions
        return self
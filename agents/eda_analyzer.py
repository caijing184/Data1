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
            'dtypes': self.df.dtypes.astype(str).to_dict(),
            'descriptive_stats': self._get_descriptive_stats(),
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
    
    def _get_descriptive_stats(self):
        """获取描述性统计信息"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return {}
        
        stats_df = self.df[numeric_cols].describe()
        
        # 添加额外的统计量
        for col in numeric_cols:
            data = self.df[col].dropna()
            if len(data) > 0:
                stats_df.loc['skew', col] = data.skew()
                stats_df.loc['kurtosis', col] = data.kurtosis()
                stats_df.loc['median', col] = data.median()
                stats_df.loc['iqr', col] = data.quantile(0.75) - data.quantile(0.25)
        
        return stats_df.to_dict()
    
    def correlation_analysis(self):
        """相关性分析"""
        print("开始相关性分析...")
        
        # 只选择数值列
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) == 0:
            print("警告: 没有数值列可用于相关性分析")
            self.eda_results['correlation'] = {
                'matrix': {},
                'top_features_with_target': []
            }
            return self
        
        # 检查目标列是否在数值列中
        if self.target_col not in numeric_df.columns:
            print(f"警告: 目标列 '{self.target_col}' 不在数值列中")
            # 将目标列添加到numeric_df中
            if self.target_col in self.df.columns:
                numeric_df = pd.concat([numeric_df, self.df[self.target_col]], axis=1)
            else:
                print(f"错误: 目标列 '{self.target_col}' 不存在")
                self.eda_results['correlation'] = {
                    'matrix': {},
                    'top_features_with_target': []
                }
                return self
        
        # 计算相关性矩阵
        correlation_matrix = numeric_df.corr()
        print(f"相关性矩阵形状: {correlation_matrix.shape}")
        
        # 获取与目标列相关性最高的特征
        if self.target_col in correlation_matrix.columns:
            target_corr = correlation_matrix[self.target_col].drop(self.target_col, errors='ignore')
            
            if len(target_corr) > 0:
                # 按绝对值排序
                top_correlations = target_corr.abs().sort_values(ascending=False).head(10)
                
                # 格式化为字典列表
                top_corr_list = []
                for feature, corr_value in top_correlations.items():
                    top_corr_list.append({
                        'feature1': self.target_col,
                        'feature2': feature,
                        'correlation': float(corr_value)
                    })
                
                print(f"找到 {len(top_corr_list)} 个与目标相关的特征")
            else:
                print("警告: 没有找到与目标列相关的特征")
                top_corr_list = []
        else:
            print(f"警告: 目标列 '{self.target_col}' 不在相关性矩阵中")
            top_corr_list = []
        
        self.eda_results['correlation'] = {
            'matrix': correlation_matrix.to_dict(),
            'top_features_with_target': top_corr_list
        }
        
        print("相关性分析完成")
        return self
    
    def distribution_analysis(self):
        """分布分析"""
        print("开始分布分析...")
        distributions = {}
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        # 只分析前10个数值列，避免太多
        cols_to_analyze = list(numeric_cols)[:10]
        print(f"将分析 {len(cols_to_analyze)} 个特征的分布")
        
        for col in cols_to_analyze:
            data = self.df[col].dropna()
            if len(data) < 3:  # Shapiro检验需要至少3个样本
                continue
            
            try:
                # 基本统计
                distributions[col] = {
                    'mean': float(data.mean()),
                    'std': float(data.std()),
                    'skewness': float(data.skew()),
                    'kurtosis': float(data.kurtosis()),
                    'median': float(data.median()),
                    'min': float(data.min()),
                    'max': float(data.max())
                }
                
                # 正态性检验（Shapiro-Wilk）
                if len(data) <= 5000:  # Shapiro检验最多支持5000个样本
                    try:
                        shapiro_test = stats.shapiro(data)
                        distributions[col]['normality_test'] = {
                            'statistic': float(shapiro_test.statistic),
                            'p_value': float(shapiro_test.pvalue)
                        }
                    except Exception as e:
                        print(f"Shapiro检验失败 ({col}): {str(e)}")
                        distributions[col]['normality_test'] = {
                            'statistic': None,
                            'p_value': None
                        }
                else:
                    # 对于大样本，使用Kolmogorov-Smirnov检验
                    try:
                        ks_test = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
                        distributions[col]['normality_test'] = {
                            'statistic': float(ks_test.statistic),
                            'p_value': float(ks_test.pvalue)
                        }
                    except Exception as e:
                        print(f"KS检验失败 ({col}): {str(e)}")
                        distributions[col]['normality_test'] = {
                            'statistic': None,
                            'p_value': None
                        }
                        
            except Exception as e:
                # 如果检验失败，只记录基本统计量
                print(f"分布分析失败 ({col}): {str(e)}")
                distributions[col] = {
                    'mean': float(data.mean()) if len(data) > 0 else 0,
                    'std': float(data.std()) if len(data) > 0 else 0,
                    'skewness': float(data.skew()) if len(data) > 0 else 0,
                    'kurtosis': float(data.kurtosis()) if len(data) > 0 else 0,
                    'normality_test': {
                        'statistic': None,
                        'p_value': None
                    }
                }
        
        self.eda_results['distributions'] = distributions
        print(f"分布分析完成，分析了 {len(distributions)} 个特征")
        return self
    
    def get_eda_results(self):
        """获取EDA结果"""
        return self.eda_results
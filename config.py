"""
Kaggle乳腺癌数据分析配置
"""
# 数据集配置
DATA_CONFIG = {
    'default_path': 'data/breast_cancer_kaggle.csv',  # 默认数据路径
    'id_column': 'id',  # ID列名（如果有的话）
    'diagnosis_column': 'diagnosis',  # 诊断结果列名
    'diagnosis_mapping': {'B': 0, 'M': 1},  # 诊断结果映射：B=良性(0), M=恶性(1)
    'drop_columns': ['id', 'Unnamed: 32'],  # 需要删除的列
}

# 分析配置
ANALYSIS_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5,
    'feature_scaling_method': 'standard',  # 'standard'或'minmax'
    'feature_selection_k': 10,
    'top_features_count': 15,
}

# 可视化配置
VIZ_CONFIG = {
    'figsize_corr': (14, 12),
    'figsize_dist': (12, 8),
    'figsize_model': (14, 7),
    'dpi': 300,
}
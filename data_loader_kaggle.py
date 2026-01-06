import pandas as pd
import numpy as np
import os
from config import DATA_CONFIG

class KaggleDataLoader:
    def __init__(self, data_path=None):
        """
        初始化Kaggle数据加载器
        
        Args:
            data_path: 数据文件路径，如果为None则使用默认路径
        """
        self.data_path = data_path or DATA_CONFIG['default_path']
        self.raw_data = None
        self.feature_names = None
        self.target_column = 'target'  # 转换后的目标列名
        self._load_config()
    
    def _load_config(self):
        """加载配置"""
        self.id_column = DATA_CONFIG['id_column']
        self.diagnosis_column = DATA_CONFIG['diagnosis_column']
        self.diagnosis_mapping = DATA_CONFIG['diagnosis_mapping']
        self.drop_columns = DATA_CONFIG['drop_columns']
    
    def load_data(self):
        """加载Kaggle乳腺癌数据"""
        try:
            print(f"正在加载数据: {self.data_path}")
            
            # 检查文件是否存在
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"数据文件不存在: {self.data_path}")
            
            # 读取CSV文件
            self.raw_data = pd.read_csv(self.data_path)
            print(f"数据加载成功: {self.raw_data.shape[0]} 行, {self.raw_data.shape[1]} 列")
            
            # 显示前几行数据
            print("\n数据预览:")
            print(self.raw_data.head())
            
            # 显示列名
            print("\n数据列名:")
            print(self.raw_data.columns.tolist())
            
            # 数据预处理
            self._preprocess_data()
            
            return self.raw_data
            
        except Exception as e:
            print(f"数据加载失败: {str(e)}")
            raise
    
    def _preprocess_data(self):
        """预处理数据"""
        print("\n开始数据预处理...")
        
        # 1. 检查诊断结果列是否存在
        if self.diagnosis_column not in self.raw_data.columns:
            raise ValueError(f"诊断结果列 '{self.diagnosis_column}' 不存在于数据中")
        
        # 2. 创建备份
        original_data = self.raw_data.copy()
        
        # 3. 删除不需要的列
        columns_to_drop = []
        for col in self.drop_columns:
            if col in self.raw_data.columns:
                columns_to_drop.append(col)
        
        if columns_to_drop:
            print(f"删除列: {columns_to_drop}")
            self.raw_data = self.raw_data.drop(columns=columns_to_drop)
        
        # 4. 将诊断结果转换为数值
        print(f"转换诊断结果列: {self.diagnosis_column}")
        unique_values = self.raw_data[self.diagnosis_column].unique()
        print(f"诊断结果唯一值: {unique_values}")
        
        # 检查是否所有值都在映射中
        for val in unique_values:
            if pd.isna(val):
                continue
            if val not in self.diagnosis_mapping:
                raise ValueError(f"诊断结果值 '{val}' 不在映射字典中")
        
        # 执行转换
        self.raw_data[self.target_column] = self.raw_data[self.diagnosis_column].map(self.diagnosis_mapping)
        
        # 5. 删除原始诊断列（保留转换后的target列）
        if self.diagnosis_column != self.target_column:
            self.raw_data = self.raw_data.drop(columns=[self.diagnosis_column])
        
        # 6. 检查转换结果
        target_distribution = self.raw_data[self.target_column].value_counts()
        print(f"目标变量分布: {target_distribution.to_dict()}")
        
        # 7. 获取特征名称（排除目标列）
        self.feature_names = [col for col in self.raw_data.columns if col != self.target_column]
        print(f"特征数量: {len(self.feature_names)}")
        print(f"特征示例: {self.feature_names[:5]}...")
        
        # 8. 基本统计信息
        print(f"\n数据形状: {self.raw_data.shape}")
        print(f"数据类型:")
        print(self.raw_data.dtypes.value_counts())
        
        print("数据预处理完成!")
        
    def get_data_info(self):
        """获取数据信息"""
        if self.raw_data is None:
            return None
        
        target_counts = self.raw_data[self.target_column].value_counts()
        total = len(self.raw_data)
        
        info = {
            'original_shape': self.raw_data.shape,
            'feature_count': len(self.feature_names),
            'sample_count': total,
            'benign_count': int(target_counts.get(0, 0)),
            'malignant_count': int(target_counts.get(1, 0)),
            'benign_percentage': (target_counts.get(0, 0) / total) * 100 if total > 0 else 0,
            'malignant_percentage': (target_counts.get(1, 0) / total) * 100 if total > 0 else 0,
            'feature_names': self.feature_names,
            'target_column': self.target_column,
        }
        
        return info
    
    def get_sample_data(self, n=5):
        """获取数据样本"""
        if self.raw_data is None:
            return None
        
        return self.raw_data.head(n)
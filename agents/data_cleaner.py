import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, df):
        self.df = df
        self.cleaning_report = {}
    
    def detect_missing_values(self):
        """检测缺失值"""
        missing_stats = self.df.isnull().sum()
        missing_percentage = (missing_stats / len(self.df)) * 100
        self.cleaning_report['missing_values'] = {
            'counts': missing_stats[missing_stats > 0].to_dict(),
            'percentages': missing_percentage[missing_percentage > 0].to_dict()
        }
        return self
    
    def handle_missing_values(self, strategy='median'):
        """处理缺失值"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if strategy == 'median':
            for col in numeric_cols:
                if self.df[col].isnull().any():
                    self.df[col] = self.df[col].fillna(self.df[col].median())
        elif strategy == 'mean':
            for col in numeric_cols:
                if self.df[col].isnull().any():
                    self.df[col] = self.df[col].fillna(self.df[col].mean())
        
        self.cleaning_report['missing_treatment'] = {
            'strategy': strategy,
            'columns_affected': list(numeric_cols[self.df[numeric_cols].isnull().any()])
        }
        return self
    
    def detect_outliers(self, method='iqr'):
        """检测异常值"""
        outliers_report = {}
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if method == 'iqr':
            for col in numeric_cols:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
                if len(outliers) > 0:
                    outliers_report[col] = {
                        'count': len(outliers),
                        'percentage': (len(outliers) / len(self.df)) * 100,
                        'indices': outliers.index.tolist()
                    }
        
        self.cleaning_report['outliers'] = outliers_report
        return self
    
    def get_cleaning_report(self):
        """获取清洗报告"""
        return self.cleaning_report
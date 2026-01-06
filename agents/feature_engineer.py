import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

class FeatureEngineer:
    def __init__(self, df, target_col='target'):
        self.df = df
        self.target_col = target_col
        self.feature_importance = {}
    
    def scale_features(self, method='standard'):
        """特征标准化"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != self.target_col]
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            return self
        
        scaled_features = scaler.fit_transform(self.df[numeric_cols])
        self.df[numeric_cols] = scaled_features
        return self
    
    def feature_selection_anova(self, k=10):
        """使用ANOVA进行特征选择"""
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        
        selector = SelectKBest(score_func=f_classif, k=min(k, len(X.columns)))
        selector.fit(X, y)
        
        feature_scores = {}
        for i, col in enumerate(X.columns):
            feature_scores[col] = {
                'score': float(selector.scores_[i]) if i < len(selector.scores_) else 0,
                'p_value': float(selector.pvalues_[i]) if i < len(selector.pvalues_) else 1,
                'selected': col in X.columns[selector.get_support()]
            }
        
        self.feature_importance['anova'] = feature_scores
        return self
    
    def feature_selection_rf(self):
        """使用随机森林进行特征选择"""
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        feature_scores = {}
        for i, col in enumerate(X.columns):
            feature_scores[col] = {
                'importance': float(rf.feature_importances_[i]),
                'rank': i + 1
            }
        
        sorted_features = sorted(feature_scores.items(), 
                                key=lambda x: x[1]['importance'], 
                                reverse=True)
        
        self.feature_importance['random_forest'] = dict(sorted_features[:15])
        return self
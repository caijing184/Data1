import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix)

class ModelBuilder:
    def __init__(self, df, target_col='target', test_size=0.2, random_state=42):
        self.df = df
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.models = {}
        self.results = {}
    
    def prepare_data(self):
        """准备训练和测试数据"""
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        return self
    
    def train_models(self):
        """训练多个模型"""
        model_configs = {
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=self.random_state),
            'decision_tree': DecisionTreeClassifier(random_state=self.random_state),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'gradient_boosting': GradientBoostingClassifier(random_state=self.random_state),
            'svm': SVC(probability=True, random_state=self.random_state)
        }
        
        for name, model in model_configs.items():
            model.fit(self.X_train, self.y_train)
            self.models[name] = model
        return self
    
    def evaluate_models(self):
        """评估模型性能"""
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            metrics = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred, average='weighted'),
                'recall': recall_score(self.y_test, y_pred, average='weighted'),
                'f1_score': f1_score(self.y_test, y_pred, average='weighted'),
                'confusion_matrix': confusion_matrix(self.y_test, y_pred).tolist()
            }
            
            if y_pred_proba is not None:
                metrics['roc_auc'] = roc_auc_score(self.y_test, y_pred_proba)
            
            self.results[name] = metrics
        return self
    
    def cross_validation(self, cv=5):
        """交叉验证"""
        cv_results = {}
        for name, model in self.models.items():
            scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring='accuracy')
            cv_results[name] = {
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'all_scores': scores.tolist()
            }
        self.results['cross_validation'] = cv_results
        return self
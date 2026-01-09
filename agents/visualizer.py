import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import BytesIO
import base64

class Visualizer:
    def __init__(self):
        self.figures = {}
        self.base64_images = {}
    
    def create_correlation_heatmap(self, df, figsize=(12, 10)):
        """创建相关性热力图"""
        fig, ax = plt.subplots(figsize=figsize)
        correlation_matrix = df.select_dtypes(include=[np.number]).corr()
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, 
                   square=True, linewidths=.5, cbar_kws={"shrink": .8}, ax=ax)
        ax.set_title('Feature Correlation Heatmap')
        
        self._save_figure_as_base64(fig, 'correlation_heatmap')
        return self
    
    def create_feature_distribution(self, df, feature, target_col='target', figsize=(10, 6)):
        """创建特征分布图"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        df[feature].hist(ax=axes[0], bins=30, edgecolor='black')
        axes[0].set_title(f'Distribution of {feature}')
        axes[0].set_xlabel(feature)
        axes[0].set_ylabel('Frequency')
        
        for target_value in df[target_col].unique():
            subset = df[df[target_col] == target_value]
            axes[1].hist(subset[feature], alpha=0.5, label=f'Target={target_value}', bins=30)
        
        axes[1].set_title(f'{feature} Distribution by Target')
        axes[1].set_xlabel(feature)
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        
        plt.tight_layout()
        
        key = f'distribution_{feature}'
        self._save_figure_as_base64(fig, key)
        return self
    
    def create_model_comparison(self, model_results, figsize=(12, 6)):
        """创建模型比较图"""
        fig, ax = plt.subplots(figsize=figsize)
        
        models = list(model_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        x = np.arange(len(models))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = [model_results[model][metric] for model in models]
            ax.bar(x + i*width, values, width, label=metric)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Scores')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width*1.5)
        ax.set_xticklabels(models, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        self._save_figure_as_base64(fig, 'model_comparison')
        return self
    
    def _save_figure_as_base64(self, fig, key):
        """将图形保存为base64编码"""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        self.base64_images[key] = image_base64
        plt.close(fig)
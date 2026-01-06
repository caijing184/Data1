from jinja2 import Template
import markdown
from datetime import datetime

class ReportGenerator:
    def __init__(self, analysis_results):
        self.results = analysis_results
        self.report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def generate_markdown(self):
        """生成Markdown格式报告"""
        # 准备数据
        data_info = self.results.get('data_info', {})
        benign_count = data_info.get('benign_count', 0)
        malignant_count = data_info.get('malignant_count', 0)
        benign_percentage = data_info.get('benign_percentage', 0)
        malignant_percentage = data_info.get('malignant_percentage', 0)
        
        template_content = """
# Kaggle乳腺癌数据分析报告

**生成时间**: {{ report_date }}

## 1. 数据集概览

### 1.1 基本信息
- 数据来源: Kaggle乳腺癌数据集
- 数据集形状: {{ data_info.shape }}
- 特征数量: {{ data_info.feature_count }}
- 样本数量: {{ data_info.sample_count }}
- 目标变量分布: 
  - 良性 (B): {{ benign_count }} 个样本 ({{ "%.1f"|format(benign_percentage) }}%)
  - 恶性 (M): {{ malignant_count }} 个样本 ({{ "%.1f"|format(malignant_percentage) }}%)

{% if 'feature_names' in data_info %}
### 1.2 特征列表
前10个特征: {{ data_info.feature_names[:10]|join(', ') }}...
{% endif %}

## 2. 数据质量检查

{% if 'cleaning' in results and 'missing_values' in results.cleaning and results.cleaning.missing_values %}
### 2.1 缺失值检测
发现缺失值的特征:
{% for col, count in results.cleaning.missing_values.counts.items() %}
- {{ col }}: {{ count }} 个缺失值 ({{ "%.2f"|format(results.cleaning.missing_values.percentages[col]) }}%)
{% endfor %}
{% else %}
✅ 无缺失值
{% endif %}

{% if 'cleaning' in results and 'outliers' in results.cleaning and results.cleaning.outliers %}
### 2.2 异常值检测
发现异常值的特征:
{% for col, info in results.cleaning.outliers.items() %}
- {{ col }}: {{ info.count }} 个异常值 ({{ "%.2f"|format(info.percentage) }}%)
{% endfor %}
{% else %}
✅ 无明显异常值
{% endif %}

## 3. 探索性数据分析

{% if 'eda' in results and 'correlation' in results.eda and results.eda.correlation.top_features_with_target %}
### 3.1 与诊断结果相关性最强的特征
{% for feature in results.eda.correlation.top_features_with_target[:5] %}
{% set corr_feature = feature.feature1 if feature.feature1 != 'target' else feature.feature2 %}
{{ loop.index }}. **{{ corr_feature }}**: 相关性 = {{ "%.3f"|format(feature.correlation) }}
{% endfor %}
{% endif %}

{% if 'eda' in results and 'basic_statistics' in results.eda %}
### 3.2 数据统计摘要
- 数值型特征数量: {{ results.eda.basic_statistics.dtypes|select("equalto", "float64")|list|length + results.eda.basic_statistics.dtypes|select("equalto", "int64")|list|length }}
- 平均缺失值比例: {{ "%.2f"|format(results.eda.basic_statistics.missing_values.values()|sum / (results.eda.basic_statistics.shape[0] * results.eda.basic_statistics.shape[1]) * 100) }}%
{% endif %}

## 4. 特征重要性分析

{% if 'feature_importance' in results and 'random_forest' in results.feature_importance %}
### 4.1 基于随机森林的特征重要性排名
{% for feature, importance in results.feature_importance.random_forest.items() %}
{{ loop.index }}. **{{ feature }}**: {{ "%.4f"|format(importance.importance) }}
{% endfor %}
{% endif %}

## 5. 模型性能评估

{% if 'modeling' in results %}
### 5.1 模型性能比较
| 模型 | 准确率 | 精确率 | 召回率 | F1分数 | AUC |
|------|--------|--------|--------|--------|-----|
{% for model, metrics in results.modeling.items() if model != 'cross_validation' and model != 'error' %}
| {{ model.replace('_', ' ')|title }} | {{ "%.3f"|format(metrics.accuracy) }} | {{ "%.3f"|format(metrics.precision) }} | {{ "%.3f"|format(metrics.recall) }} | {{ "%.3f"|format(metrics.f1_score) }} | {{ "%.3f"|format(metrics.get('roc_auc', 0)) }} |
{% endfor %}

{% if 'cross_validation' in results.modeling %}
### 5.2 交叉验证结果
{% for model, cv_results in results.modeling.cross_validation.items() %}
- **{{ model.replace('_', ' ')|title }}**: 平均准确率 = {{ "%.3f"|format(cv_results.mean_score) }} (±{{ "%.3f"|format(cv_results.std_score) }})
{% endfor %}
{% endif %}
{% endif %}

## 6. 关键洞见与发现

### 6.1 主要发现
{% for insight in results.insights %}
{{ loop.index }}. {{ insight }}
{% endfor %}

### 6.2 数据特点总结
{% if benign_count > malignant_count %}
1. 数据集存在类别不平衡，良性样本多于恶性样本
2. 这可能会影响模型对恶性样本的识别能力
{% else %}
1. 数据集类别分布相对平衡
{% endif %}
3. 基于特征重要性分析，形态学特征（如半径、周长）对诊断最重要
4. 非线性模型（如随机森林）通常表现优于线性模型

## 7. 建议与后续步骤

{% for recommendation in results.recommendations %}
{{ loop.index }}. {{ recommendation }}
{% endfor %}

### 7.1 临床应用建议
1. **模型部署**: 建议将最佳模型集成到临床决策支持系统中
2. **持续监控**: 定期评估模型在新数据上的性能
3. **特征工程**: 考虑添加临床背景特征以提升模型解释性
4. **多中心验证**: 在不同医疗机构数据上验证模型泛化能力

## 8. 技术细节

### 8.1 分析方法
- 数据清洗: 缺失值填充、异常值检测
- 特征工程: 标准化、特征选择（ANOVA、随机森林）
- 建模算法: 逻辑回归、决策树、随机森林、梯度提升、SVM
- 评估指标: 准确率、精确率、召回率、F1分数、AUC
- 验证方法: 80/20训练测试分割 + 5折交叉验证

### 8.2 生成的可视化图表
{% if 'visualizations' in results %}
- 相关性热力图
- 重要特征分布图
- 模型性能对比图
{% endif %}

---

**报告说明**: 本报告由Kaggle乳腺癌数据分析代理系统自动生成。所有分析基于提供的乳腺癌数据集，结果仅供参考，不能替代专业医疗诊断。

**免责声明**: 本报告中的分析结果基于机器学习模型，可能存在误差。实际医疗决策应结合临床医生专业判断。
"""
        
        template = Template(template_content)
        markdown_report = template.render(
            report_date=self.report_date,
            results=self.results,
            data_info=data_info,
            benign_count=benign_count,
            malignant_count=malignant_count,
            benign_percentage=benign_percentage,
            malignant_percentage=malignant_percentage
        )
        
        return markdown_report
    
    def generate_html(self, markdown_report):
        """将Markdown转换为HTML"""
        try:
            html_report = markdown.markdown(markdown_report, extensions=['tables', 'fenced_code'])
            
            html_template = """
            <!DOCTYPE html>
            <html lang="zh-CN">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>乳腺癌数据分析报告</title>
                <style>
                    body { font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; background-color: #f5f5f5; color: #333; }
                    .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }
                    h1, h2, h3 { color: #2c3e50; }
                    h1 { border-bottom: 3px solid #3498db; padding-bottom: 10px; }
                    h2 { border-left: 5px solid #3498db; padding-left: 15px; margin-top: 30px; }
                    table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                    th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                    th { background-color: #3498db; color: white; }
                    tr:nth-child(even) { background-color: #f8f9fa; }
                    .insight { background-color: #e8f4fc; border-left: 4px solid #3498db; padding: 15px; margin: 15px 0; }
                    .recommendation { background-color: #e8f6f3; border-left: 4px solid #1abc9c; padding: 15px; margin: 15px 0; }
                    .image-container { text-align: center; margin: 20px 0; }
                    .image-container img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }
                    .error { background-color: #f8d7da; border-left: 4px solid #dc3545; padding: 15px; margin: 15px 0; color: #721c24; }
                </style>
            </head>
            <body>
                <div class="container">
                    {{ html_content }}
                    <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; text-align: center; color: #7f8c8d; font-size: 0.9em;">
                        报告生成时间: {{ report_date }}
                    </footer>
                </div>
            </body>
            </html>
            """
            
            final_html = Template(html_template).render(
                html_content=html_report,
                report_date=self.report_date
            )
            
            return final_html
            
        except Exception as e:
            error_html = f"""
            <!DOCTYPE html>
            <html lang="zh-CN">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>乳腺癌数据分析报告 - 错误</title>
                <style>
                    body {{ font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; background-color: #f5f5f5; color: #333; }}
                    .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
                    h1 {{ color: #dc3545; border-bottom: 3px solid #dc3545; padding-bottom: 10px; }}
                    .error {{ background-color: #f8d7da; border-left: 4px solid #dc3545; padding: 15px; margin: 15px 0; color: #721c24; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>乳腺癌数据分析报告生成错误</h1>
                    <div class="error">
                        <h3>错误信息:</h3>
                        <p>{str(e)}</p>
                        <h3>报告生成时间:</h3>
                        <p>{self.report_date}</p>
                    </div>
                    <p>请重新运行分析程序或联系技术支持。</p>
                </div>
            </body>
            </html>
            """
            return error_html
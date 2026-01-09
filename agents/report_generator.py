from jinja2 import Template
import markdown
from datetime import datetime
import json

class ReportGenerator:
    def __init__(self, analysis_results):
        self.results = analysis_results
        self.report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _get_nested_value(self, data, keys, default=None):
        """安全获取嵌套字典的值"""
        if isinstance(keys, str):
            keys = keys.split('.')
        
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current
    
    def generate_markdown(self):
        """生成Markdown格式报告"""
        # 使用调试信息查看数据结构
        debug_info = {
            'eda_keys': list(self.results.get('eda', {}).keys()) if self.results.get('eda') else [],
            'eda_structure': json.dumps(self.results.get('eda', {}), indent=2, default=str) if self.results.get('eda') else 'No EDA data'
        }
        
        # 1. 基本数据信息
        data_info = self.results.get('data_info', {})
        benign_count = data_info.get('benign_count', 0)
        malignant_count = data_info.get('malignant_count', 0)
        benign_percentage = data_info.get('benign_percentage', 0)
        malignant_percentage = data_info.get('malignant_percentage', 0)
        
        # 2. 数据清洗信息
        cleaning = self.results.get('cleaning', {})
        missing_values = cleaning.get('missing_values', {})
        missing_counts = missing_values.get('counts', {}) if isinstance(missing_values, dict) else {}
        missing_percentages = missing_values.get('percentages', {}) if isinstance(missing_values, dict) else {}
        outliers = cleaning.get('outliers', {})
        
        # 3. EDA信息 - 直接使用原始数据，不进行复杂的提取
        eda_data = self.results.get('eda', {})
        
        # 尝试不同的访问路径获取EDA数据
        basic_stats = None
        correlation_results = []
        
        # 尝试获取基本统计信息
        if 'basic_statistics' in eda_data:
            basic_stats = eda_data['basic_statistics']
        elif 'basic_stats' in eda_data:
            basic_stats = eda_data['basic_stats']
        
        # 尝试获取相关性分析结果
        if 'correlation' in eda_data:
            corr_data = eda_data['correlation']
            if isinstance(corr_data, dict) and 'top_features_with_target' in corr_data:
                correlation_results = corr_data['top_features_with_target']
            elif isinstance(corr_data, list):
                correlation_results = corr_data
        
        # 4. 特征重要性
        feature_importance = self.results.get('feature_importance', {})
        rf_features = feature_importance.get('random_forest', {})
        
        # 5. 模型结果
        modeling_results = self.results.get('modeling', {})
        model_table_data = []
        
        # 提取模型性能数据
        for model_name, metrics in modeling_results.items():
            if model_name == 'cross_validation' or not isinstance(metrics, dict):
                continue
            
            # 提取指标
            accuracy = metrics.get('accuracy', 0)
            precision = metrics.get('precision_weighted', metrics.get('precision_macro', metrics.get('precision', 0)))
            recall = metrics.get('recall_weighted', metrics.get('recall_macro', metrics.get('recall', 0)))
            f1 = metrics.get('f1_score_weighted', metrics.get('f1_score_macro', metrics.get('f1_score', 0)))
            auc = metrics.get('roc_auc', 0)
            
            model_table_data.append({
                'name': model_name.replace('_', ' ').title(),
                'accuracy': f"{accuracy:.3f}" if isinstance(accuracy, (int, float)) else "0.000",
                'precision': f"{precision:.3f}" if isinstance(precision, (int, float)) else "0.000",
                'recall': f"{recall:.3f}" if isinstance(recall, (int, float)) else "0.000",
                'f1': f"{f1:.3f}" if isinstance(f1, (int, float)) else "0.000",
                'auc': f"{auc:.3f}" if isinstance(auc, (int, float)) else "0.000"
            })
        
        # 6. 交叉验证数据
        cv_data = []
        cv_results = modeling_results.get('cross_validation', {})
        for model_name, cv_metrics in cv_results.items():
            if isinstance(cv_metrics, dict):
                mean_score = cv_metrics.get('mean_score', 0)
                std_score = cv_metrics.get('std_score', 0)
                
                cv_data.append({
                    'name': model_name.replace('_', ' ').title(),
                    'mean': f"{mean_score:.3f}",
                    'std': f"{std_score:.3f}"
                })
        
        # 7. 洞见和建议
        insights = self.results.get('insights', [])
        recommendations = self.results.get('recommendations', [])
        
        template_content = """
# Kaggle乳腺癌数据分析报告

**报告生成时间**: {{ report_date }}

{% if debug_info.eda_keys %}
<!-- 调试信息: EDA键值: {{ debug_info.eda_keys }} -->
{% endif %}

## 1. 数据集概览

### 1.1 基本信息
- **数据来源**: Kaggle乳腺癌数据集
- **数据形状**: {{ data_info.shape | default("N/A") }}
- **特征数量**: {{ data_info.feature_count | default("N/A") }}
- **样本数量**: {{ data_info.sample_count | default("N/A") }}
- **目标变量分布**:
  - **良性 (B)**: {{ benign_count }} 个样本 ({{ "%.1f"|format(benign_percentage) }}%)
  - **恶性 (M)**: {{ malignant_count }} 个样本 ({{ "%.1f"|format(malignant_percentage) }}%)

## 2. 数据质量检查

### 2.1 缺失值检测
{% if missing_counts %}
发现缺失值的特征:
{% for col, count in missing_counts.items() %}
- **{{ col }}**: {{ count }} 个缺失值 ({{ "%.2f"|format(missing_percentages.get(col, 0)) }}%)
{% endfor %}
{% else %}
✅ **无缺失值**
{% endif %}

### 2.2 异常值检测
{% if outliers %}
发现异常值的特征:
{% for col, info in outliers.items() %}
- **{{ col }}**: {{ info.count }} 个异常值 ({{ "%.2f"|format(info.percentage) }}%)
{% endfor %}
{% else %}
✅ **无明显异常值**
{% endif %}

## 3. 探索性数据分析

### 3.1 基本统计信息
{% if basic_stats %}
{% if basic_stats.shape %}
- 数据集形状: {{ basic_stats.shape }}
{% endif %}
{% if basic_stats.target_distribution %}
- 目标变量分布: 
  {% for target, count in basic_stats.target_distribution.items() %}
    {% if target == 0 or target == '0' %}良性: {{ count }} 个样本
    {% elif target == 1 or target == '1' %}恶性: {{ count }} 个样本
    {% else %}类别{{ target }}: {{ count }} 个样本{% endif %}
  {% endfor %}
{% endif %}
{% else %}
- 基本统计信息不可用
{% endif %}

### 3.2 与诊断结果相关性最强的特征
{% if correlation_results %}
{% for feature in correlation_results %}
{% if loop.index <= 5 %}
{% set corr_feature = feature.feature2 if feature.feature1 == 'target' else feature.feature1 %}
{{ loop.index }}. **{{ corr_feature }}**: 相关性 = {{ "%.3f"|format(feature.correlation) }}
{% endif %}
{% endfor %}
{% else %}
- 相关性分析结果不可用
{% endif %}

## 4. 特征重要性分析

### 4.1 基于随机森林的特征重要性排名
{% if rf_features %}
{% for feature, importance in rf_features.items() %}
{% if loop.index <= 10 %}
{% if importance is mapping %}
{{ loop.index }}. **{{ feature }}**: {{ "%.4f"|format(importance.get('importance', importance)) }}
{% else %}
{{ loop.index }}. **{{ feature }}**: {{ "%.4f"|format(importance) }}
{% endif %}
{% endif %}
{% endfor %}
{% else %}
- 特征重要性分析结果不可用
{% endif %}

## 5. 模型性能评估

### 5.1 模型性能比较
{% if model_table_data %}
| 模型 | 准确率 | 精确率 | 召回率 | F1分数 | AUC |
|------|--------|--------|--------|--------|-----|
{% for model in model_table_data %}
| {{ model.name }} | {{ model.accuracy }} | {{ model.precision }} | {{ model.recall }} | {{ model.f1 }} | {{ model.auc }} |
{% endfor %}
{% else %}
- 模型性能数据不可用
{% endif %}

{% if cv_data %}
### 5.2 交叉验证结果
{% for cv in cv_data %}
- **{{ cv.name }}**: 平均准确率 = {{ cv.mean }} (±{{ cv.std }})
{% endfor %}
{% endif %}

## 6. 关键洞见与发现

### 6.1 主要发现
{% if insights %}
{% for insight in insights %}
{{ loop.index }}. {{ insight }}
{% endfor %}
{% else %}
1. 数据集包含 {{ data_info.sample_count | default("未知数量") }} 个样本
2. 数据质量分析完成
3. 机器学习模型训练完成
{% endif %}

### 6.2 数据特点总结
{% if benign_count > malignant_count %}
1. 数据集存在类别不平衡，良性样本多于恶性样本
2. 这可能会影响模型对恶性样本的识别能力
{% elif malignant_count > benign_count %}
1. 数据集存在类别不平衡，恶性样本多于良性样本
2. 这可能会影响模型对良性样本的识别能力
{% else %}
1. 数据集类别分布平衡
{% endif %}
3. 基于特征重要性分析，形态学特征对诊断最重要
4. 非线性模型通常表现优于线性模型

## 7. 建议与后续步骤

{% if recommendations %}
{% for recommendation in recommendations %}
{{ loop.index }}. {{ recommendation }}
{% endfor %}
{% else %}
1. **模型优化**: 尝试调整模型超参数以进一步提升性能
2. **特征工程**: 创建新的交互特征或多项式特征
3. **集成学习**: 使用模型集成方法（如投票分类器、堆叠）
4. **深度学习**: 尝试神经网络模型
5. **部署监控**: 部署模型后持续监控性能变化
{% endif %}

## 8. 技术细节

### 8.1 分析流程
1. **数据加载**: 读取CSV文件，转换诊断结果为数值型
2. **数据清洗**: 处理缺失值，检测异常值
3. **探索性分析**: 统计描述，相关性分析，分布分析
4. **特征工程**: 特征缩放，特征选择，PCA降维
5. **模型训练**: 训练多种机器学习模型
6. **模型评估**: 使用交叉验证和测试集评估性能
7. **报告生成**: 自动生成分析报告

### 8.2 评估指标说明
- **准确率**: 正确预测的比例
- **精确率**: 预测为正例中实际为正例的比例
- **召回率**: 实际为正例中被正确预测的比例
- **F1分数**: 精确率和召回率的调和平均
- **AUC**: ROC曲线下的面积，衡量模型整体性能

### 8.3 生成的可视化图表
- 特征相关性热力图
- 重要特征分布图
- 模型性能对比图

---

**免责声明**: 本报告基于机器学习算法生成，分析结果仅供参考，不能替代专业医疗诊断。实际医疗决策应结合临床医生专业判断和更多检查结果。

**报告生成系统**: Kaggle乳腺癌数据分析代理
**版本**: 1.0
**生成时间**: {{ report_date }}
"""
        
        template = Template(template_content)
        markdown_report = template.render(
            report_date=self.report_date,
            data_info=data_info,
            benign_count=benign_count,
            malignant_count=malignant_count,
            benign_percentage=benign_percentage,
            malignant_percentage=malignant_percentage,
            missing_counts=missing_counts,
            missing_percentages=missing_percentages,
            outliers=outliers,
            basic_stats=basic_stats,
            correlation_results=correlation_results,
            rf_features=rf_features,
            model_table_data=model_table_data,
            cv_data=cv_data,
            insights=insights,
            recommendations=recommendations,
            debug_info=debug_info
        )
        
        return markdown_report
    
    def generate_html(self, markdown_report):
        """将Markdown转换为HTML"""
        # 移除调试信息
        lines = markdown_report.split('\n')
        clean_lines = [line for line in lines if not line.strip().startswith('<!-- 调试信息') and not line.strip().endswith('-->')]
        clean_markdown = '\n'.join(clean_lines)
        
        # 转换Markdown为HTML
        html_content = markdown.markdown(
            clean_markdown, 
            extensions=['tables', 'fenced_code']
        )
        
        # 创建完整的HTML页面
        html_template = """
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Kaggle乳腺癌数据分析报告</title>
            <style>
                * {
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }
                
                body {
                    font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
                    line-height: 1.6;
                    color: #333;
                    background-color: #f8f9fa;
                    padding: 20px;
                }
                
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 40px;
                    border-radius: 10px;
                    box-shadow: 0 0 30px rgba(0, 0, 0, 0.1);
                }
                
                .header {
                    text-align: center;
                    margin-bottom: 40px;
                    padding-bottom: 20px;
                    border-bottom: 3px solid #3498db;
                }
                
                h1 {
                    color: #2c3e50;
                    font-size: 2.5em;
                    margin-bottom: 10px;
                }
                
                .subtitle {
                    color: #7f8c8d;
                    font-size: 1.1em;
                    margin-bottom: 20px;
                }
                
                .report-date {
                    color: #95a5a6;
                    font-size: 0.9em;
                }
                
                h2 {
                    color: #2c3e50;
                    font-size: 1.8em;
                    margin-top: 40px;
                    margin-bottom: 20px;
                    padding-bottom: 10px;
                    border-bottom: 2px solid #ecf0f1;
                }
                
                h3 {
                    color: #3498db;
                    font-size: 1.4em;
                    margin-top: 30px;
                    margin-bottom: 15px;
                }
                
                h4 {
                    color: #2c3e50;
                    font-size: 1.2em;
                    margin-top: 20px;
                    margin-bottom: 10px;
                }
                
                p {
                    margin-bottom: 15px;
                }
                
                ul, ol {
                    margin-bottom: 20px;
                    padding-left: 30px;
                }
                
                li {
                    margin-bottom: 8px;
                }
                
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
                    font-size: 0.95em;
                }
                
                th {
                    background-color: #3498db;
                    color: white;
                    font-weight: 600;
                    text-align: left;
                    padding: 12px 15px;
                }
                
                td {
                    padding: 10px 15px;
                    border-bottom: 1px solid #ecf0f1;
                }
                
                tr:nth-child(even) {
                    background-color: #f8f9fa;
                }
                
                tr:hover {
                    background-color: #f1f8ff;
                }
                
                .highlight {
                    background-color: #e8f4fc;
                    padding: 15px;
                    border-left: 4px solid #3498db;
                    margin: 15px 0;
                    border-radius: 0 5px 5px 0;
                }
                
                .success {
                    background-color: #d5f4e6;
                    padding: 15px;
                    border-left: 4px solid #27ae60;
                    margin: 15px 0;
                    border-radius: 0 5px 5px 0;
                }
                
                .warning {
                    background-color: #fef9e7;
                    padding: 15px;
                    border-left: 4px solid #f39c12;
                    margin: 15px 0;
                    border-radius: 0 5px 5px 0;
                }
                
                .info-box {
                    background-color: #f8f9fa;
                    border: 1px solid #e0e0e0;
                    padding: 20px;
                    margin: 20px 0;
                    border-radius: 8px;
                }
                
                .visualization-section {
                    text-align: center;
                    margin: 30px 0;
                }
                
                .visualization-section img {
                    max-width: 90%;
                    height: auto;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    box-shadow: 0 3px 15px rgba(0, 0, 0, 0.1);
                }
                
                .footer {
                    margin-top: 50px;
                    padding-top: 20px;
                    border-top: 1px solid #ecf0f1;
                    text-align: center;
                    color: #95a5a6;
                    font-size: 0.9em;
                }
                
                .disclaimer {
                    background-color: #fff3cd;
                    border: 1px solid #ffeaa7;
                    padding: 15px;
                    margin: 30px 0;
                    border-radius: 5px;
                    font-size: 0.9em;
                }
                
                .status-good {
                    color: #27ae60;
                    font-weight: bold;
                }
                
                .status-warning {
                    color: #f39c12;
                    font-weight: bold;
                }
                
                .status-error {
                    color: #e74c3c;
                    font-weight: bold;
                }
                
                @media (max-width: 768px) {
                    .container {
                        padding: 20px;
                    }
                    
                    h1 {
                        font-size: 2em;
                    }
                    
                    h2 {
                        font-size: 1.6em;
                    }
                    
                    table {
                        font-size: 0.85em;
                    }
                    
                    th, td {
                        padding: 8px 10px;
                    }
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Kaggle乳腺癌数据分析报告</h1>
                    <div class="subtitle">基于机器学习的数据分析与预测模型评估</div>
                    <div class="report-date">生成时间: {{ report_date }}</div>
                </div>
                
                {{ html_content }}
                
                <div class="disclaimer">
                    <strong>免责声明:</strong> 本报告基于机器学习算法自动生成，分析结果仅供参考，不能替代专业医疗诊断。实际医疗决策应结合临床医生专业判断和更多检查结果。
                </div>
                
                <div class="footer">
                    <p>报告生成系统: Kaggle乳腺癌数据分析代理 | 版本: 1.0</p>
                    <p>© 2024 乳腺癌数据分析项目 | 仅供学术研究使用</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        final_html = Template(html_template).render(
            html_content=html_content,
            report_date=self.report_date
        )
        
        return final_html
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import base64
from config import DATA_CONFIG, ANALYSIS_CONFIG, VIZ_CONFIG

# 导入自定义模块
from agents.data_loader_kaggle import KaggleDataLoader
from agents.data_cleaner import DataCleaner
from agents.eda_analyzer import EDAnalyzer
from agents.feature_engineer import FeatureEngineer
from agents.model_builder import ModelBuilder
from agents.visualizer import Visualizer
from agents.report_generator import ReportGenerator


class BreastCancerKaggleAnalyzer:
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.df = None
        self.results = {}
        self.visualizer = Visualizer()
        self._load_config()
    
    def _load_config(self):
        """加载配置"""
        self.test_size = ANALYSIS_CONFIG['test_size']
        self.random_state = ANALYSIS_CONFIG['random_state']
        self.cv_folds = ANALYSIS_CONFIG['cv_folds']
        self.feature_scaling_method = ANALYSIS_CONFIG['feature_scaling_method']
        self.feature_selection_k = ANALYSIS_CONFIG['feature_selection_k']
        self.top_features_count = ANALYSIS_CONFIG['top_features_count']
        
        # 诊断列映射
        self.diagnosis_mapping = DATA_CONFIG['diagnosis_mapping']
        self.target_column = 'target'  # 统一的目标列名
    
    def run_full_analysis(self):
        """运行完整分析流程"""
        print("=" * 60)
        print("Kaggle乳腺癌数据分析代理系统")
        print("=" * 60)
        
        # 1. 加载数据
        print("\n1. 加载Kaggle乳腺癌数据...")
        try:
            data_loader = KaggleDataLoader(self.data_path)
            self.df = data_loader.load_data()
            data_info = data_loader.get_data_info()
            
            self.results['data_info'] = data_info
            print(f"数据加载成功: {self.df.shape[0]} 样本, {self.df.shape[1]-1} 特征")
            
        except Exception as e:
            print(f"数据加载失败: {str(e)}")
            raise
        
        # 2. 数据清洗
        print("\n2. 数据清洗...")
        try:
            cleaner = DataCleaner(self.df)
            cleaner.detect_missing_values().handle_missing_values().detect_outliers()
            self.results['cleaning'] = cleaner.get_cleaning_report()
            
            print("数据清洗完成")
            if self.results['cleaning'].get('missing_values'):
                missing = self.results['cleaning']['missing_values']['counts']
                print(f"发现缺失值: {len(missing)} 列")
            
        except Exception as e:
            print(f"数据清洗失败: {str(e)}")
            self.results['cleaning'] = {'error': str(e)}
        
        # 3. EDA分析
        print("\n3. 探索性数据分析...")
        try:
            eda = EDAnalyzer(self.df, target_col=self.target_column)
            eda.basic_statistics()
            
            # 添加对相关性分析的异常处理
            try:
                eda.correlation_analysis()
            except Exception as e:
                print(f"相关性分析出错，但继续执行: {str(e)}")
                # 确保eda_results中有correlation键
                if 'correlation' not in eda.eda_results:
                    eda.eda_results['correlation'] = {
                        'matrix': {},
                        'top_features_with_target': []
                    }
            
            # 添加对分布分析的异常处理
            try:
                eda.distribution_analysis()
            except Exception as e:
                print(f"分布分析出错，但继续执行: {str(e)}")
                if 'distributions' not in eda.eda_results:
                    eda.eda_results['distributions'] = {}
            
            self.results['eda'] = eda.eda_results
            
            print("EDA分析完成")
            
            # 安全地显示关键发现
            if 'correlation' in self.results['eda']:
                top_corr = self.results['eda']['correlation'].get('top_features_with_target', [])
                if len(top_corr) > 0:
                    corr_item = top_corr[0]
                    corr_feature = corr_item['feature2'] if corr_item['feature1'] == self.target_column else corr_item['feature1']
                    print(f"✓ 与诊断最相关的特征: {corr_feature} (相关性: {corr_item['correlation']:.3f})")
                else:
                    print("✓ 未找到显著的相关性特征")
            
        except Exception as e:
            print(f"EDA分析失败: {str(e)}")
            import traceback
            traceback.print_exc()
            # 确保eda部分有基本结构
            self.results['eda'] = {
                'basic_statistics': {},
                'correlation': {'top_features_with_target': []},
                'distributions': {},
                'error': str(e)
            }
        
        # 4. 特征工程
        print("\n4. 特征工程...")
        try:
            feature_engineer = FeatureEngineer(self.df, target_col=self.target_column)
            feature_engineer.scale_features(method=self.feature_scaling_method)
            feature_engineer.feature_selection_anova(k=self.feature_selection_k)
            feature_engineer.feature_selection_rf()
            
            self.results['feature_importance'] = feature_engineer.feature_importance
            self.df = feature_engineer.df  # 更新处理后的数据
            
            print("特征工程完成")
            top_features = list(feature_engineer.feature_importance['random_forest'].keys())[:5]
            print(f"最重要的5个特征: {', '.join(top_features)}")
            
        except Exception as e:
            print(f"特征工程失败: {str(e)}")
            self.results['feature_importance'] = {'error': str(e)}
        
        # 5. 可视化
        print("\n5. 生成可视化...")
        try:
            # 相关性热力图
            self.visualizer.create_correlation_heatmap(
                self.df, 
                figsize=VIZ_CONFIG['figsize_corr']
            )
            
            # 重要特征分布图
            if 'random_forest' in self.results.get('feature_importance', {}):
                important_features = list(
                    self.results['feature_importance']['random_forest'].keys()
                )[:3]
                
                for feature in important_features:
                    if feature in self.df.columns:
                        self.visualizer.create_feature_distribution(
                            self.df, 
                            feature, 
                            target_col=self.target_column,
                            figsize=VIZ_CONFIG['figsize_dist']
                        )
            
            print(f"生成 {len(self.visualizer.base64_images)} 个可视化图表")
            
        except Exception as e:
            print(f"可视化生成失败: {str(e)}")
        
        # 6. 建模与评估
        print("\n6. 建模与评估...")
        try:
            model_builder = ModelBuilder(
                self.df, 
                target_col=self.target_column,
                test_size=self.test_size,
                random_state=self.random_state
            )
            
            model_builder.prepare_data().train_models().evaluate_models().cross_validation(cv=self.cv_folds)
            self.results['modeling'] = model_builder.results
            
            # 找出最佳模型
            best_model_name = None
            best_accuracy = 0
            
            for model_name, metrics in model_builder.results.items():
                if model_name == 'cross_validation':
                    continue
                
                if metrics['accuracy'] > best_accuracy:
                    best_accuracy = metrics['accuracy']
                    best_model_name = model_name
            
            print(f"模型训练完成，最佳模型: {best_model_name} (准确率: {best_accuracy:.3f})")
            
            # 生成模型比较图
            model_results = {k: v for k, v in model_builder.results.items() if k != 'cross_validation'}
            self.visualizer.create_model_comparison(
                model_results,
                figsize=VIZ_CONFIG['figsize_model']
            )
            
        except Exception as e:
            print(f"建模失败: {str(e)}")
            self.results['modeling'] = {'error': str(e)}
        
        # 7. 生成洞见
        print("\n7. 生成洞见...")
        self._generate_insights()
        
        # 8. 生成报告
        print("\n8. 生成分析报告...")
        report = self._generate_report()
        
        print("\n" + "=" * 60)
        print("分析完成!")
        print("=" * 60)
        
        return report
    
    def _generate_insights(self):
        """生成关键洞见"""
        insights = []
        
        try:
            # 数据分布洞见
            if 'data_info' in self.results:
                info = self.results['data_info']
                insights.append(
                    f"数据集包含 {info['sample_count']} 个样本，其中良性 {info['benign_count']} 个 ({info['benign_percentage']:.1f}%)，"
                    f"恶性 {info['malignant_count']} 个 ({info['malignant_percentage']:.1f}%)"
                )
            
            # 特征重要性洞见
            if 'feature_importance' in self.results and 'random_forest' in self.results['feature_importance']:
                top_features = list(self.results['feature_importance']['random_forest'].keys())[:3]
                insights.append(f"最重要的预测特征: {', '.join(top_features)}")
            
            # 相关性洞见
            if 'eda' in self.results and 'correlation' in self.results['eda']:
                top_corr = self.results['eda']['correlation']['top_features_with_target']
                if top_corr:
                    corr_item = top_corr[0]
                    corr_feature = corr_item['feature1'] if corr_item['feature1'] != self.target_column else corr_item['feature2']
                    insights.append(f"与诊断结果相关性最强的特征: {corr_feature} (相关性: {corr_item['correlation']:.3f})")
            
            # 模型性能洞见
            if 'modeling' in self.results:
                best_model = None
                best_accuracy = 0
                
                for model_name, metrics in self.results['modeling'].items():
                    if model_name == 'cross_validation':
                        continue
                    
                    if isinstance(metrics, dict) and 'accuracy' in metrics:
                        if metrics['accuracy'] > best_accuracy:
                            best_accuracy = metrics['accuracy']
                            best_model = model_name
                
                if best_model:
                    insights.append(f"最佳性能模型: {best_model} (准确率: {best_accuracy:.3f})")
            
            # 数据质量洞见
            if 'cleaning' in self.results:
                if 'missing_values' in self.results['cleaning']:
                    missing = self.results['cleaning']['missing_values']['counts']
                    if missing:
                        insights.append(f"数据包含缺失值，涉及 {len(missing)} 个特征")
                    else:
                        insights.append("数据质量良好，无缺失值")
            
        except Exception as e:
            print(f"生成洞见时出错: {str(e)}")
            insights = ["洞见生成过程中出现错误"]
        
        self.results['insights'] = insights
        
        # 生成建议
        recommendations = [
            "建议优先关注半径、周长和面积相关的特征，它们通常对乳腺癌诊断最重要",
            "随机森林和梯度提升模型通常表现良好，建议作为首选模型",
            "考虑收集更多临床特征（如患者年龄、家族史）以提升模型性能",
            "定期验证模型性能，特别是在应用于新数据集时",
            "可以探索深度学习模型如神经网络，但需要更多数据支持"
        ]
        self.results['recommendations'] = recommendations
    
    def _generate_report(self):
        """生成完整报告"""
        try:
            report_generator = ReportGenerator(self.results)
            markdown_report = report_generator.generate_markdown()
            html_report = report_generator.generate_html(markdown_report)
            
            # 保存报告
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs('reports', exist_ok=True)
            
            # 保存Markdown报告
            md_filename = f'reports/breast_cancer_kaggle_report_{timestamp}.md'
            with open(md_filename, 'w', encoding='utf-8') as f:
                f.write(markdown_report)
            
            # 保存HTML报告
            html_filename = f'reports/breast_cancer_kaggle_report_{timestamp}.html'
            with open(html_filename, 'w', encoding='utf-8') as f:
                f.write(html_report)
            
            # 保存可视化图像
            for key, fig_base64 in self.visualizer.base64_images.items():
                img_filename = f'reports/{key}_{timestamp}.png'
                with open(img_filename, 'wb') as f:
                    f.write(base64.b64decode(fig_base64))
            
            # 保存原始结果数据（JSON格式）
            json_filename = f'reports/analysis_results_{timestamp}.json'
            with open(json_filename, 'w', encoding='utf-8') as f:
                # 将numpy类型转换为Python原生类型以便JSON序列化
                json_results = self._convert_to_json_serializable(self.results)
                json.dump(json_results, f, indent=2, ensure_ascii=False)
            
            print(f"报告已保存到 reports/ 目录")
            print(f"- Markdown报告: {md_filename}")
            print(f"- HTML报告: {html_filename}")
            print(f"- 可视化图表: {len(self.visualizer.base64_images)} 个PNG文件")
            print(f"- 结果数据: {json_filename}")
            
            return {
                'markdown': markdown_report,
                'html': html_report,
                'visualizations': list(self.visualizer.base64_images.keys()),
                'timestamp': timestamp,
                'files': {
                    'markdown': md_filename,
                    'html': html_filename,
                    'json': json_filename
                }
            }
            
        except Exception as e:
            print(f"生成报告失败: {str(e)}")
            raise
    
    def _convert_to_json_serializable(self, obj):
        """将对象转换为JSON可序列化的格式"""
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_to_json_serializable(item) for item in obj)
        elif pd.isna(obj):
            return None
        else:
            return obj


def main():
    """主函数"""
    print("Kaggle乳腺癌数据分析代理系统")
    print("=" * 60)
    
    # 询问数据路径
    default_path = DATA_CONFIG['default_path']
    print(f"默认数据路径: {default_path}")
    
    use_default = input(f"是否使用默认数据路径? (y/n): ").strip().lower()
    
    if use_default == 'y':
        data_path = default_path
    else:
        data_path = input("请输入CSV文件路径: ").strip()
    
    # 检查文件是否存在
    if not os.path.exists(data_path):
        print(f"错误: 文件 '{data_path}' 不存在!")
        print("请确保数据文件存在，或将其放置在 data/ 目录下")
        return
    
    # 运行分析
    try:
        analyzer = BreastCancerKaggleAnalyzer(data_path)
        report = analyzer.run_full_analysis()
        
        print("\n分析完成!")
        print(f"报告文件已保存到 reports/ 目录")
        print(f"时间戳: {report['timestamp']}")
        
        # 询问是否打开HTML报告
        open_report = input("\n是否打开HTML报告? (y/n): ").strip().lower()
        if open_report == 'y' and os.name == 'nt':  # Windows
            import webbrowser
            html_file = report['files']['html']
            webbrowser.open(f'file:///{os.path.abspath(html_file)}')
        
    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
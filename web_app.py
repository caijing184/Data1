import matplotlib
# 设置matplotlib为非交互式后端，避免使用Tkinter
matplotlib.use('Agg')  # Agg是一个非交互式后端，不会使用Tkinter

import warnings
warnings.filterwarnings('ignore')

from flask import Flask, request, jsonify, render_template, send_file
import os
import uuid
import sys

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import BreastCancerKaggleAnalyzer

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
REPORT_FOLDER = 'reports'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB限制

@app.route('/')
def index():
    """首页"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_data():
    """分析数据并生成报告"""
    try:
        # 检查是否有文件上传
        if 'file' not in request.files:
            # 如果没有文件，使用默认数据
            default_path = 'data/breast_cancer_kaggle.csv'
            if not os.path.exists(default_path):
                return jsonify({
                    'success': False,
                    'error': '没有上传文件且默认数据文件不存在'
                })
            
            analyzer = BreastCancerKaggleAnalyzer(default_path)
            report = analyzer.run_full_analysis()
            
            return jsonify({
                'success': True,
                'message': '使用默认数据集分析完成',
                'report_files': report['visualizations'],
                'timestamp': report['timestamp']
            })
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': '没有选择文件'}), 400
        
        if file and file.filename.endswith('.csv'):
            # 保存上传的文件
            file_id = str(uuid.uuid4())[:8]
            filename = f"{file_id}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                # 运行分析
                analyzer = BreastCancerKaggleAnalyzer(filepath)
                report = analyzer.run_full_analysis()
                
                return jsonify({
                    'success': True,
                    'message': '文件分析完成',
                    'report_files': report['visualizations'],
                    'timestamp': report['timestamp']
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
            finally:
                # 清理上传的文件
                if os.path.exists(filepath):
                    os.remove(filepath)
        else:
            return jsonify({'success': False, 'error': '请上传CSV文件'}), 400
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """下载报告文件"""
    filepath = os.path.join(REPORT_FOLDER, filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    else:
        return jsonify({'success': False, 'error': '文件不存在'}), 404

@app.route('/reports')
def list_reports():
    """列出所有报告"""
    reports = []
    for file in os.listdir(REPORT_FOLDER):
        if file.endswith('.html'):
            filepath = os.path.join(REPORT_FOLDER, file)
            reports.append({
                'name': file,
                'path': f'/download/{file}',
                'size': os.path.getsize(filepath),
                'created': os.path.getctime(filepath)
            })
    
    # 按创建时间排序
    reports.sort(key=lambda x: x['created'], reverse=True)
    
    return jsonify({'success': True, 'reports': reports})

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API接口：直接分析数据"""
    try:
        # 获取数据（可以是文件或JSON数据）
        if 'file' in request.files:
            file = request.files['file']
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{uuid.uuid4()}.csv")
            file.save(filepath)
            
            analyzer = BreastCancerKaggleAnalyzer(filepath)
            report = analyzer.run_full_analysis()
            
            # 清理临时文件
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'data': report
            })
        else:
            return jsonify({
                'success': False,
                'error': '请提供CSV文件'
            })
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("启动Kaggle乳腺癌数据分析Web应用...")
    print("访问 http://localhost:5000 使用Web界面")
    print("按 Ctrl+C 停止服务器")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
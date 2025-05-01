
from flask import Flask, render_template, request, send_file
from predictor import load_model, predict
import pandas as pd
import os
from werkzeug.utils import secure_filename
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 历史记录文件路径
HISTORY_FILE = "history.csv"

# 加载模型
model, scaler, feature_columns = load_model()

# 初始化历史记录文件（首次创建）
if not os.path.exists(HISTORY_FILE):
    pd.DataFrame(columns=["Time", "Alloy", "Dₘₐₓ (mm)", "Mode"]).to_csv(HISTORY_FILE, index=False)

# 首页：单个预测
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    error = None
    if request.method == 'POST':
        composition = request.form.get('composition')
        try:
            result = predict(model, scaler, feature_columns, composition)

            # 写入历史记录
            pd.DataFrame([{
                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Alloy": composition,
                "Dₘₐₓ (mm)": round(result, 4),
                "Mode": "single"
            }]).to_csv(HISTORY_FILE, mode='a', header=False, index=False)

        except Exception as e:
            error = str(e)
    return render_template('index.html', result=result, error=error)

# 批量预测
@app.route('/batch', methods=['GET', 'POST'])
def batch():
    results_df = None
    error = None
    chart_data = []

    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            error = "请上传一个文件。"
        else:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            try:
                # 读取上传文件
                if filename.endswith('.xlsx'):
                    df = pd.read_excel(file_path)
                elif filename.endswith('.csv'):
                    df = pd.read_csv(file_path)
                else:
                    raise ValueError("仅支持 .csv 或 .xlsx 文件")

                if 'Alloy' not in df.columns:
                    raise ValueError("文件必须包含名为 'Alloy' 的列")

                # 执行预测
                df['Dₘₐₓ (mm)'] = df['Alloy'].apply(
                    lambda s: predict(model, scaler, feature_columns, s)
                )
                results_df = df
                chart_data = df['Dₘₐₓ (mm)'].tolist()
                df.to_excel("预测结果.xlsx", index=False)

                # 写入历史记录（批量）
                batch_records = pd.DataFrame({
                    "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Alloy": df['Alloy'],
                    "Dₘₐₓ (mm)": df['Dₘₐₓ (mm)'].round(4),
                    "Mode": "batch"
                })
                batch_records.to_csv(HISTORY_FILE, mode='a', header=False, index=False)

            except Exception as e:
                error = f"处理文件失败：{str(e)}"

    return render_template("batch.html", table=results_df, error=error, chart_data=chart_data)

# 下载批量预测结果
@app.route('/download')
def download():
    return send_file("预测结果.xlsx", as_attachment=True)

# 下载历史记录文件（CSV）
@app.route('/history')
def download_history():
    return send_file(HISTORY_FILE, as_attachment=True)

# 历史记录页面（HTML 表格展示）
@app.route('/history/view')
def view_history():
    try:
        df = pd.read_csv(HISTORY_FILE)
    except Exception:
        df = pd.DataFrame(columns=["Time", "Alloy", "Dₘₐₓ (mm)", "Mode"])
    return render_template("history.html", records=df)

# 启动 Flask 服务（Render 平台专用端口支持）
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

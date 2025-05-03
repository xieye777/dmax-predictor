
from flask import Flask, render_template, request, send_file, redirect, url_for, session, g
from predictor import load_model, predict
import pandas as pd
import os
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

USER_DB = "/tmp/users.db"

def init_db():
    conn = sqlite3.connect(USER_DB)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

def get_db():
    conn = sqlite3.connect(USER_DB)
    conn.row_factory = sqlite3.Row
    return conn

def get_user_by_email(email):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    conn.close()
    return user

def create_user(email, password):
    password_hash = generate_password_hash(password)
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO users (email, password_hash) VALUES (?, ?)", (email, password_hash))
    conn.commit()
    user_id = cursor.lastrowid
    conn.close()
    return user_id

def verify_user(email, password):
    user = get_user_by_email(email)
    if user and check_password_hash(user["password_hash"], password):
        return user
    return None

def load_logged_in_user():
    user_id = session.get("user_id")
    if user_id is None:
        g.user = None
    else:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        g.user = cursor.fetchone()
        conn.close()

def login_required(view_func):
    def wrapped_view(*args, **kwargs):
        if g.user is None:
            return redirect(url_for('login'))
        return view_func(*args, **kwargs)
    wrapped_view.__name__ = view_func.__name__
    return wrapped_view

@app.before_request
def before_request():
    load_logged_in_user()

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if get_user_by_email(email):
            error = '邮箱已被注册'
        else:
            create_user(email, password)
            return redirect(url_for('login'))
    return render_template('register.html', error=error)

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = verify_user(email, password)
        if user:
            session.clear()
            session['user_id'] = user['id']
            return redirect(url_for('index'))
        else:
            error = '邮箱或密码错误'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

model, scaler, feature_columns = load_model()

@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    result = None
    error = None
    user_id = g.user['id']
    HISTORY_FILE = f"history_user_{user_id}.csv"

    if request.method == 'POST':
        composition = request.form.get('composition')
        try:
            result = predict(model, scaler, feature_columns, composition)
            pd.DataFrame([{
                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Alloy": composition,
                "Dₘₐₓ (mm)": round(result, 4),
                "Mode": "single"
            }]).to_csv(HISTORY_FILE, mode='a', header=not os.path.exists(HISTORY_FILE), index=False)
        except Exception as e:
            error = str(e)
    return render_template('index.html', result=result, error=error)

@app.route('/batch', methods=['GET', 'POST'])
@login_required
def batch():
    results_df = None
    error = None
    chart_data = []
    user_id = g.user['id']
    HISTORY_FILE = f"history_user_{user_id}.csv"

    failed_rows = []

    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            error = "请上传一个文件。"
        else:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            try:
                if filename.endswith('.xlsx'):
                    df = pd.read_excel(file_path)
                elif filename.endswith('.csv'):
                    df = pd.read_csv(file_path)
                else:
                    raise ValueError("仅支持 .csv 或 .xlsx 文件")

                if 'Alloy' not in df.columns:
                    raise ValueError("文件必须包含名为 'Alloy' 的列")

                predictions = []
                for i, row in df.iterrows():
                    try:
                        val = predict(model, scaler, feature_columns, row['Alloy'])
                        predictions.append(val)
                    except Exception as e:
                        failed_rows.append((i, row['Alloy'], str(e)))
                        predictions.append(None)

                df["Dₘₐₓ (mm)"] = predictions
                results_df = df[df["Dₘₐₓ (mm)"].notna()]
                chart_data = results_df["Dₘₐₓ (mm)"].tolist()
                results_df.to_excel("预测结果.xlsx", index=False)

                batch_records = pd.DataFrame({
                    "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Alloy": results_df['Alloy'],
                    "Dₘₐₓ (mm)": results_df['Dₘₐₓ (mm)'].round(4),
                    "Mode": "batch"
                })
                batch_records.to_csv(HISTORY_FILE, mode='a', header=not os.path.exists(HISTORY_FILE), index=False)

                if failed_rows:
                    error = f"部分合金无法预测，共 {len(failed_rows)} 项，例如：{failed_rows[0][:2]}"

            except Exception as e:
                error = f"处理文件失败：{str(e)}"

    return render_template("batch.html", table=results_df, error=error, chart_data=chart_data)

@app.route('/download')
@login_required
def download():
    user_id = g.user['id']
    return send_file("预测结果.xlsx", as_attachment=True)

@app.route('/history')
@login_required
def download_history():
    user_id = g.user['id']
    HISTORY_FILE = f"history_user_{user_id}.csv"
    return send_file(HISTORY_FILE, as_attachment=True)

@app.route('/history/view')
@login_required
def view_history():
    user_id = g.user['id']
    HISTORY_FILE = f"history_user_{user_id}.csv"
    try:
        df = pd.read_csv(HISTORY_FILE)
    except Exception:
        df = pd.DataFrame(columns=["Time", "Alloy", "Dₘₐₓ (mm)", "Mode"])
    return render_template("history.html", records=df)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

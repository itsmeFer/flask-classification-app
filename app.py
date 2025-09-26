"""
Flask Classification/Regression App - Random Forest
Fitur:
- Login admin
- Upload dataset CSV (bisa "," atau ";" sebagai pemisah)
- Pilih kolom target
- Training RandomForest (Classifier atau Regressor otomatis)
- Tampilkan hasil akurasi / R2 / MAE / MSE, classification report, confusion matrix

Cara menjalankan:
1. pip install flask pandas scikit-learn
2. python app.py
3. Buka http://127.0.0.1:5000
Login: admin / password123
"""

import os
import traceback
import pandas as pd
from flask import Flask, request, redirect, url_for, session, render_template_string, flash
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

app = Flask(__name__)
app.secret_key = "replace_this_with_a_secure_random_key"

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ADMIN_USER = "admin"
ADMIN_PASS = "password123"

# ================= HTML Template =================
LOGIN_HTML = '''
<!doctype html>
<title>Login</title>
<h2>Login - Admin</h2>
{% with messages = get_flashed_messages() %}
  {% if messages %}
    <ul style="color: red;">{% for m in messages %}<li>{{m}}</li>{% endfor %}</ul>
  {% endif %}
{% endwith %}
<form action="{{ url_for('login') }}" method="post">
  <label>Username: <input name="username"></label><br>
  <label>Password: <input name="password" type="password"></label><br>
  <button type="submit">Login</button>
</form>
<p>Or <a href="{{ url_for('dashboard') }}?sample=1">continue with sample dataset (Breast Cancer)</a></p>
'''

DASHBOARD_HTML = '''
<!doctype html>
<title>Dashboard</title>
<h2>Dashboard</h2>
<p>Logged in as <strong>{{ session.get('user') }}</strong> - <a href="{{ url_for('logout') }}">Logout</a></p>

<h3>1) Upload CSV dataset (or use sample)</h3>
<form method="post" action="{{ url_for('upload') }}" enctype="multipart/form-data">
  <input type="file" name="file" accept=".csv">
  <button type="submit">Upload CSV</button>
</form>
<hr>

{% if df is not none %}
  <h3>Dataset preview (first 5 rows)</h3>
  {{ df.head().to_html(classes='data', header=True, index=False) | safe }}
  <h4>Columns detected:</h4>
  <form method="post" action="{{ url_for('process') }}">
    <label for="target">Select target column (label):</label>
    <select name="target">
      {% for c in columns %}
        <option value="{{c}}">{{c}}</option>
      {% endfor %}
    </select>
    <p>
      <label>Test size (0-0.9): <input name="test_size" value="0.2"></label>
    </p>
    <p>
      <label>Random state: <input name="random_state" value="42"></label>
    </p>
    <button type="submit">Run Model</button>
  </form>
{% else %}
  <p>No dataset loaded. You can upload a CSV or <a href="{{ url_for('dashboard') }}?sample=1">use sample dataset</a>.</p>
{% endif %}

{% if result %}
  <hr>
  <h3>Results</h3>
  {% if result.type == 'classification' %}
    <p><strong>Accuracy:</strong> {{ result['accuracy']*100 | round(2) }}%</p>
    <h4>Classification Report</h4>
    <pre>{{ result['report'] }}</pre>
    <h4>Confusion Matrix</h4>
    <pre>{{ result['confusion'] }}</pre>
  {% else %}
    <p><strong>R2 Score:</strong> {{ result['r2'] | round(4) }}</p>
    <p><strong>MAE:</strong> {{ result['mae'] | round(4) }}</p>
    <p><strong>MSE:</strong> {{ result['mse'] | round(4) }}</p>
  {% endif %}
{% endif %}

<hr>
<p><small>Note: Demo only. In production, secure sessions and passwords properly.</small></p>
'''

# ================= Utils =================
def load_sample_dataframe():
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    df = pd.concat([X, y], axis=1)
    return df

def read_csv_flexible(filepath):
    """Baca CSV dengan fleksibel: deteksi ',' atau ';' sebagai separator."""
    with open(filepath, 'r', encoding='utf-8') as f:
        first_line = f.readline()
    sep = ';' if ';' in first_line else ','
    return pd.read_csv(filepath, sep=sep)

# ================= Routes =================
@app.route('/', methods=['GET'])
def index():
    if session.get('logged_in'):
        return redirect(url_for('dashboard'))
    return render_template_string(LOGIN_HTML)

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username', '')
    password = request.form.get('password', '')
    if username == ADMIN_USER and password == ADMIN_PASS:
        session['logged_in'] = True
        session['user'] = username
        session.pop('csv_file', None)
        return redirect(url_for('dashboard'))
    else:
        flash('Invalid credentials')
        return redirect(url_for('index'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/dashboard', methods=['GET'])
def dashboard():
    if not session.get('logged_in'):
        if request.args.get('sample') == '1':
            df = load_sample_dataframe()
            session['csv_file'] = None
            columns = list(df.columns)
            return render_template_string(DASHBOARD_HTML, df=df, columns=columns, result=None)
        return redirect(url_for('index'))

    csv_file = session.get('csv_file')
    if csv_file and os.path.exists(csv_file):
        try:
            df = read_csv_flexible(csv_file)
        except Exception as e:
            df = None
            flash(f'Error reading stored dataset: {e}')
    else:
        df = None

    columns = list(df.columns) if df is not None else []
    return render_template_string(DASHBOARD_HTML, df=df, columns=columns, result=None)

@app.route('/upload', methods=['POST'])
def upload():
    if not session.get('logged_in'):
        return redirect(url_for('index'))
    f = request.files.get('file')
    if not f:
        flash('No file uploaded')
        return redirect(url_for('dashboard'))
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        f.save(filepath)
        session['csv_file'] = filepath
        flash(f'Uploaded {f.filename} successfully')
    except Exception as e:
        flash(f'Error saving CSV: {e}')
    return redirect(url_for('dashboard'))

@app.route('/process', methods=['POST'])
def process():
    if not session.get('logged_in'):
        return redirect(url_for('index'))

    csv_file = session.get('csv_file')
    if csv_file and os.path.exists(csv_file):
        df = read_csv_flexible(csv_file)
    else:
        flash('No dataset loaded. Upload a CSV or use sample.')
        return redirect(url_for('dashboard'))

    try:
        target = request.form.get('target')
        test_size = float(request.form.get('test_size', 0.2))
        random_state = int(request.form.get('random_state', 42))

        if target not in df.columns:
            flash('Selected target column not in dataset')
            return redirect(url_for('dashboard'))

        y = df[target]
        X = df.drop(columns=[target])

        # Handle missing values
        for col in X.columns:
            if X[col].dtype.kind in 'biufc':
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else 'missing')

        # Encode categorical
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

        # Hapus kolom yang hanya punya 1 nilai unik
        X = X.loc[:, X.apply(pd.Series.nunique) > 1]

        if X.shape[1] == 0:
            flash('Error: After preprocessing, no features left for training.')
            return redirect(url_for('dashboard'))

        # Tentukan tipe: classification atau regression
        is_classification = y.dtype.kind not in 'fc'

        if is_classification:
            # Encode target jika kategori
            if y.dtype.kind not in 'biufc':
                y = pd.factorize(y)[0]

            # Stratify hanya jika semua kelas punya >=2 sampel
            class_counts = pd.Series(y).value_counts()
            stratify_param = y if class_counts.min() >= 2 else None

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state,
                stratify=stratify_param
            )

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=random_state)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, zero_division=0)
            cm = confusion_matrix(y_test, y_pred)

            result = {
                'type': 'classification',
                'accuracy': acc,
                'report': report,
                'confusion': cm.tolist()
            }
        else:
            # Regression
            y = pd.to_numeric(y, errors='coerce')
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=random_state)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            result = {
                'type': 'regression',
                'r2': r2_score(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'mse': mean_squared_error(y_test, y_pred)
            }

        columns = list(df.columns)
        return render_template_string(DASHBOARD_HTML, df=df, columns=columns, result=result)

    except Exception as e:
        tb = traceback.format_exc()
        flash(f'Error during processing: {e}')
        print(tb)
        return redirect(url_for('dashboard'))

# ================= Run =================
if __name__ == '__main__':
    app.run(debug=True)

"""
Flask Classification/Regression App - Random Forest (Elegant UI)
Fitur tetap sama, tapi menggunakan Bootstrap untuk tampilan elegan.
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

# ================= HTML Template with Bootstrap =================
LOGIN_HTML = '''
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Login - Admin</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="bg-light">
<div class="container py-5">
  <div class="row justify-content-center">
    <div class="col-md-5">
      <div class="card shadow-sm">
        <div class="card-body">
          <h3 class="card-title text-center mb-4">Login Admin</h3>
          {% with messages = get_flashed_messages() %}
            {% if messages %}
              <div class="alert alert-danger">
                {% for m in messages %}{{m}}<br>{% endfor %}
              </div>
            {% endif %}
          {% endwith %}
          <form action="{{ url_for('login') }}" method="post">
            <div class="mb-3">
              <label class="form-label">Username</label>
              <input class="form-control" name="username" required>
            </div>
            <div class="mb-3">
              <label class="form-label">Password</label>
              <input class="form-control" name="password" type="password" required>
            </div>
            <button type="submit" class="btn btn-primary w-100">Login</button>
          </form>
          <p class="mt-3 text-center">Or <a href="{{ url_for('dashboard') }}?sample=1">use sample dataset</a></p>
        </div>
      </div>
    </div>
  </div>
</div>
</body>
</html>
'''

DASHBOARD_HTML = '''
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dashboard</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
<style>
.dataframe { width: 100%; }
pre { background: #f8f9fa; padding: 10px; border-radius: 5px; }
</style>
</head>
<body class="bg-light">
<nav class="navbar navbar-expand-lg navbar-dark bg-primary mb-4">
  <div class="container">
    <a class="navbar-brand" href="#">RandomForest App</a>
    <div class="collapse navbar-collapse">
      <ul class="navbar-nav ms-auto">
        <li class="nav-item"><span class="nav-link text-white">Logged in as {{ session.get('user') }}</span></li>
        <li class="nav-item"><a class="nav-link text-white" href="{{ url_for('logout') }}">Logout</a></li>
      </ul>
    </div>
  </div>
</nav>

<div class="container">

  {% with messages = get_flashed_messages() %}
    {% if messages %}
      <div class="alert alert-warning">
        {% for m in messages %}{{m}}<br>{% endfor %}
      </div>
    {% endif %}
  {% endwith %}

  <div class="card mb-4 shadow-sm">
    <div class="card-body">
      <h5 class="card-title">1) Upload CSV dataset (or use sample)</h5>
      <form method="post" action="{{ url_for('upload') }}" enctype="multipart/form-data" class="d-flex gap-2">
        <input type="file" name="file" accept=".csv" class="form-control">
        <button type="submit" class="btn btn-success">Upload CSV</button>
      </form>
    </div>
  </div>

  {% if df is not none %}
    <div class="card mb-4 shadow-sm">
      <div class="card-body">
        <h5 class="card-title">Dataset preview (first 5 rows)</h5>
        {{ df.head().to_html(classes='table table-striped table-bordered', header=True, index=False) | safe }}

        <h5 class="mt-3">Columns detected</h5>
        <form method="post" action="{{ url_for('process') }}">
          <div class="row mb-3">
            <div class="col-md-4">
              <label class="form-label">Target column</label>
              <select name="target" class="form-select">
                {% for c in columns %}
                  <option value="{{c}}">{{c}}</option>
                {% endfor %}
              </select>
            </div>
            <div class="col-md-2">
              <label class="form-label">Test size</label>
              <input name="test_size" value="0.2" class="form-control">
            </div>
            <div class="col-md-2">
              <label class="form-label">Random state</label>
              <input name="random_state" value="42" class="form-control">
            </div>
          </div>
          <button type="submit" class="btn btn-primary">Run Model</button>
        </form>
      </div>
    </div>
  {% endif %}

  {% if result %}
    <div class="card mb-4 shadow-sm">
      <div class="card-body">
        <h5 class="card-title">Results</h5>
        {% if result.type == 'classification' %}
          <p><strong>Accuracy:</strong> {{ result['accuracy']*100 | round(2) }}%</p>
          <h6>Classification Report</h6>
          <pre>{{ result['report'] }}</pre>
          <h6>Confusion Matrix</h6>
          <pre>{{ result['confusion'] }}</pre>
        {% else %}
          <p><strong>R2 Score:</strong> {{ result['r2'] | round(4) }}</p>
          <p><strong>MAE:</strong> {{ result['mae'] | round(4) }}</p>
          <p><strong>MSE:</strong> {{ result['mse'] | round(4) }}</p>
        {% endif %}
      </div>
    </div>
  {% endif %}

  <p class="text-center text-muted"><small>Note: Demo only. In production, secure sessions and passwords properly.</small></p>
</div>
</body>
</html>
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

        for col in X.columns:
            if X[col].dtype.kind in 'biufc':
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else 'missing')

        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        X = X.loc[:, X.apply(pd.Series.nunique) > 1]
        if X.shape[1] == 0:
            flash('Error: After preprocessing, no features left for training.')
            return redirect(url_for('dashboard'))

        is_classification = y.dtype.kind not in 'fc'
        if is_classification:
            if y.dtype.kind not in 'biufc':
                y = pd.factorize(y)[0]
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
            result = {
                'type': 'classification',
                'accuracy': accuracy_score(y_test, y_pred),
                'report': classification_report(y_test, y_pred, zero_division=0),
                'confusion': confusion_matrix(y_test, y_pred).tolist()
            }
        else:
            y = pd.to_numeric(y, errors='coerce')
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

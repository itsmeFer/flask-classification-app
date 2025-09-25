"""
Aplikasi Flask untuk Klasifikasi dengan RandomForest
- Login admin
- Upload dataset (CSV / Excel)
- Pilih kolom target
- Training RandomForest + tampilkan hasil
"""

from flask import Flask, request, redirect, url_for, session, render_template_string, flash
import pandas as pd
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import traceback

app = Flask(__name__)
app.secret_key = "replace_this_with_a_secure_random_key"

# Hardcoded admin credentials
ADMIN_USER = "admin"
ADMIN_PASS = "password123"

# ------------------------------------------------------------------
# HTML Templates
# ------------------------------------------------------------------

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
<p>Or <a href="{{ url_for('dashboard') }}?sample=1">use sample dataset (Breast Cancer)</a></p>
'''

DASHBOARD_HTML = '''
<!doctype html>
<title>Dashboard</title>
<h2>Dashboard</h2>
<p>Logged in as <strong>{{ session.get('user') }}</strong> - <a href="{{ url_for('logout') }}">Logout</a></p>

<h3>1) Upload dataset (CSV or Excel)</h3>
<form method="post" action="{{ url_for('upload') }}" enctype="multipart/form-data">
  <input type="file" name="file" accept=".csv,.xls,.xlsx">
  <button type="submit">Upload</button>
</form>

<hr>

{% if df is not none %}
  <h3>Dataset preview (first 5 rows)</h3>
  {{ df.head().to_html(classes='data', header=True, index=False) | safe }}
  <h4>Columns detected:</h4>
  <form method="post" action="{{ url_for('process') }}">
    <label for="target">Select target column:</label>
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
  <p>No dataset loaded. You can upload a file or <a href="{{ url_for('dashboard') }}?sample=1">use sample dataset</a>.</p>
{% endif %}

{% if result %}
  <hr>
  <h3>Results</h3>
  <p><strong>Accuracy:</strong> {{ result['accuracy']*100 | round(2) }}%</p>
  <h4>Classification Report</h4>
  <pre>{{ result['report'] }}</pre>
  <h4>Confusion Matrix</h4>
  <pre>{{ result['confusion'] }}</pre>
{% endif %}
'''

# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def load_sample_dataframe():
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    return pd.concat([X, y], axis=1)

def read_file(content, filename):
    """Auto-detect file type: CSV (comma/semicolon/tab) or Excel"""
    if filename.endswith(".csv"):
        return pd.read_csv(io.BytesIO(content), sep=None, engine="python"), "csv"
    elif filename.endswith((".xls", ".xlsx")):
        return pd.read_excel(io.BytesIO(content)), "excel"
    else:
        raise ValueError("Unsupported file type. Please upload CSV or Excel.")

# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.route('/')
def index():
    if session.get('logged_in'):
        return redirect(url_for('dashboard'))
    return render_template_string(LOGIN_HTML)

@app.route('/login', methods=['POST'])
def login():
    if request.form.get('username') == ADMIN_USER and request.form.get('password') == ADMIN_PASS:
        session['logged_in'] = True
        session['user'] = ADMIN_USER
        session.pop('csv_bytes', None)
        return redirect(url_for('dashboard'))
    flash("Invalid credentials")
    return redirect(url_for('index'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        if request.args.get("sample") == "1":
            df = load_sample_dataframe()
            session['csv_bytes'] = df.to_csv(index=False).encode("utf-8")
            session['filetype'] = "csv"
            columns = list(df.columns)
            return render_template_string(DASHBOARD_HTML, df=df, columns=columns, result=None)
        return redirect(url_for("index"))

    df = None
    csv_bytes = session.get("csv_bytes")
    if csv_bytes:
        try:
            if session.get("filetype") == "excel":
                df = pd.read_excel(io.BytesIO(csv_bytes))
            else:
                df = pd.read_csv(io.BytesIO(csv_bytes), sep=None, engine="python")
        except Exception as e:
            flash(f"Error reading dataset: {e}")
    columns = list(df.columns) if df is not None else []
    return render_template_string(DASHBOARD_HTML, df=df, columns=columns, result=None)

@app.route('/upload', methods=['POST'])
def upload():
    if not session.get("logged_in"):
        return redirect(url_for("index"))
    f = request.files.get("file")
    if not f:
        flash("No file uploaded")
        return redirect(url_for("dashboard"))
    try:
        content = f.read()
        df, ftype = read_file(content, f.filename.lower())
        session['csv_bytes'] = content
        session['filetype'] = ftype
        flash(f"Uploaded {f.filename} successfully")
    except Exception as e:
        flash(f"Error reading file: {e}")
    return redirect(url_for("dashboard"))

@app.route('/process', methods=['POST'])
def process():
    if not session.get("logged_in"):
        return redirect(url_for("index"))
    csv_bytes = session.get("csv_bytes")
    if not csv_bytes:
        flash("No dataset loaded.")
        return redirect(url_for("dashboard"))
    try:
        # reload dataframe
        if session.get("filetype") == "excel":
            df = pd.read_excel(io.BytesIO(csv_bytes))
        else:
            df = pd.read_csv(io.BytesIO(csv_bytes), sep=None, engine="python")

        target = request.form.get("target")
        test_size = float(request.form.get("test_size", 0.2))
        random_state = int(request.form.get("random_state", 42))

        if target not in df.columns:
            flash("Target column not found")
            return redirect(url_for("dashboard"))

        y = df[target]
        X = df.drop(columns=[target])

        # Fill missing values
        for col in X.columns:
            if X[col].dtype.kind in "biufc":
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else "missing")

        # One-hot encode categoricals
        X = pd.get_dummies(X, drop_first=True)
        X = X.loc[:, X.apply(pd.Series.nunique) > 1]

        if y.dtype.kind not in "biufc":
            y = pd.factorize(y)[0]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if len(set(y)) > 1 else None
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=random_state)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)

        result = {
            "accuracy": accuracy_score(y_test, y_pred),
            "report": classification_report(y_test, y_pred),
            "confusion": confusion_matrix(y_test, y_pred).tolist()
        }

        columns = list(df.columns)
        return render_template_string(DASHBOARD_HTML, df=df, columns=columns, result=result)
    except Exception as e:
        tb = traceback.format_exc()
        flash(f"Error: {e}")
        print(tb)
        return redirect(url_for("dashboard"))

# ------------------------------------------------------------------
# Run (Replit will expose via web)
# ------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

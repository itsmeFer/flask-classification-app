from flask import Flask, render_template_string, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Template HTML sederhana
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Decision Helper</title>
</head>
<body>
    <h2>Upload Dataset (CSV)</h2>
    <form method="post" enctype="multipart/form-data">
        <input type="file" name="dataset" required>
        <input type="text" name="target" placeholder="Target Column" required>
        <button type="submit">Train Model</button>
    </form>

    {% if image %}
        <h3>Feature Importance</h3>
        <img src="data:image/png;base64,{{ image }}" alt="Feature Importance">
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    image = None
    if request.method == "POST":
        file = request.files['dataset']
        target = request.form['target']

        # Baca dataset
        df = pd.read_csv(file)
        X = df.drop(columns=[target])
        y = df[target]

        # Train RandomForest
        model = RandomForestClassifier()
        model.fit(X, y)

        # Plot feature importance
        importance = model.feature_importances_
        plt.figure(figsize=(8, 5))
        plt.bar(X.columns, importance)
        plt.xticks(rotation=45)
        plt.title("Feature Importance")

        # Simpan ke buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        image = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()

    return render_template_string(html_template, image=image)


if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, request, jsonify
from sklearn.linear_model import LogisticRegression
import numpy as np

app = Flask(__name__)

# --- Dummy ML model: predict 0/1 from two numeric features --- #
X_train = np.array([
    [0.1, 0.2],
    [1.0, 1.2],
    [0.2, 0.1],
    [1.3, 1.1],
])
y_train = np.array([0, 1, 0, 1])

model = LogisticRegression()
model.fit(X_train, y_train)


@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Flask ML app is running on Azure!"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expected JSON:
    {
      "feature1": float,
      "feature2": float
    }
    """
    data = request.get_json()
    if data is None:
        return jsonify({"error": "No JSON body provided"}), 400

    try:
        f1 = float(data["feature1"])
        f2 = float(data["feature2"])
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "feature1 and feature2 must be provided as numbers"}), 400

    X = np.array([[f1, f2]])
    pred = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][pred])

    return jsonify({
        "prediction": pred,
        "probability": proba
    })


if __name__ == "__main__":
    # For local dev; Azure will use gunicorn
    app.run(host="0.0.0.0", port=5000, debug=True)

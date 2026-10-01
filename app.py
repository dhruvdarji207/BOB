"""
app.py
------
Flask backend for the Car Price Prediction web application.

Routes:
  GET  /           → Home page
  GET  /predict    → Prediction form
  GET  /insights   → Model insights dashboard
  GET  /about      → About page
  GET  /api/meta   → Dataset meta (dropdowns, ranges)
  POST /predict    → JSON prediction endpoint
"""

import os
import json
import traceback

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

# ── App setup ──────────────────────────────────────────────────────────────
app = Flask(__name__)
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "car_price_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "models", "model_metrics.json")

# Cached objects (loaded once at startup)
_model   = None
_metrics = None


def load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "Model file not found. Please run: python train_model.py"
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def load_metrics():
    global _metrics
    if _metrics is None:
        if not os.path.exists(METRICS_PATH):
            raise FileNotFoundError(
                "Metrics file not found. Please run: python train_model.py"
            )
        with open(METRICS_PATH) as f:
            _metrics = json.load(f)
    return _metrics


# ── Feature definitions (must match train_model.py) ────────────────────────
NUMERICAL_FEATURES = [
    "symboling", "wheelbase", "carlength", "carwidth", "carheight",
    "curbweight", "enginesize", "boreratio", "stroke", "compressionratio",
    "horsepower", "peakrpm", "citympg", "highwaympg",
]

CATEGORICAL_FEATURES = [
    "manufacturer", "fueltype", "aspiration", "doornumber",
    "carbody", "drivewheel", "enginelocation", "enginetype",
    "cylindernumber", "fuelsystem",
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Typo normalisation (must match train_model.py)
MANUFACTURER_TYPOS = {
    "maxda":     "mazda",
    "toyouta":   "toyota",
    "porcshce":  "porsche",
    "vokswagen": "volkswagen",
    "vw":        "volkswagen",
}


# ── Input validation ────────────────────────────────────────────────────────
def validate_and_parse(data: dict):
    """
    Validates and coerces incoming JSON payload.
    Returns (feature_df, error_message).
    feature_df is None if validation fails.
    """
    errors = []
    parsed = {}

    # Numerical fields
    for field in NUMERICAL_FEATURES:
        val = data.get(field)
        if val is None or str(val).strip() == "":
            errors.append(f"Missing required field: {field}")
            continue
        try:
            parsed[field] = float(val)
        except (ValueError, TypeError):
            errors.append(f"Field '{field}' must be a number. Got: {val!r}")

    # Categorical fields
    for field in CATEGORICAL_FEATURES:
        val = data.get(field)
        if val is None or str(val).strip() == "":
            errors.append(f"Missing required field: {field}")
            continue
        parsed[field] = str(val).strip().lower()

    # Normalise manufacturer typos
    if "manufacturer" in parsed:
        parsed["manufacturer"] = MANUFACTURER_TYPOS.get(
            parsed["manufacturer"], parsed["manufacturer"]
        )

    if errors:
        return None, "; ".join(errors)

    df = pd.DataFrame([parsed], columns=ALL_FEATURES)
    return df, None


# ── Page routes ────────────────────────────────────────────────────────────
@app.route("/")
def home():
    try:
        metrics = load_metrics()
        dataset_info = metrics.get("dataset_info", {})
    except Exception:
        dataset_info = {}
    return render_template("index.html", dataset_info=dataset_info)


@app.route("/predict", methods=["GET"])
def predict_page():
    try:
        metrics = load_metrics()
        dropdown_values  = metrics.get("dropdown_values", {})
        numerical_ranges = metrics.get("numerical_ranges", {})
    except Exception:
        dropdown_values  = {}
        numerical_ranges = {}
    return render_template(
        "predict.html",
        dropdown_values=dropdown_values,
        numerical_ranges=numerical_ranges,
    )


@app.route("/insights")
def insights():
    try:
        metrics = load_metrics()
    except FileNotFoundError as e:
        return render_template("insights.html", error=str(e), metrics=None)
    return render_template("insights.html", metrics=metrics, error=None)


@app.route("/about")
def about():
    return render_template("about.html")


# ── API routes ─────────────────────────────────────────────────────────────
@app.route("/api/meta")
def api_meta():
    """Returns dropdown options and numerical ranges for the frontend."""
    try:
        metrics = load_metrics()
        return jsonify({
            "success": True,
            "dropdown_values":  metrics.get("dropdown_values", {}),
            "numerical_ranges": metrics.get("numerical_ranges", {}),
            "dataset_info":     metrics.get("dataset_info", {}),
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """Accepts JSON input and returns a predicted price."""
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"success": False, "error": "No JSON data received."}), 400

        # Validate
        feature_df, error = validate_and_parse(data)
        if error:
            return jsonify({"success": False, "error": error}), 400

        # Load model
        try:
            model = load_model()
        except FileNotFoundError as e:
            return jsonify({"success": False, "error": str(e)}), 503

        # Predict
        prediction = model.predict(feature_df)[0]
        predicted_price = round(float(prediction), 2)

        # Return result
        metrics = load_metrics()
        return jsonify({
            "success": True,
            "predicted_price": predicted_price,
            "model_used": metrics.get("best_model", "ML Model"),
            "model_r2":   metrics.get("best_metrics", {}).get("r2", None),
            "currency": "USD",
        })

    except Exception:
        # Never expose raw traceback to client
        app.logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": "An internal server error occurred. Please try again.",
        }), 500


# ── Run ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Pre-load to catch missing model early
    try:
        load_model()
        load_metrics()
        print("[✓] Model and metrics loaded successfully.")
    except FileNotFoundError as e:
        print(f"[!] Warning: {e}")
        print("[!] Run 'python train_model.py' before starting the server.\n")

    app.run(debug=True, host="0.0.0.0", port=5000)

"""Flask REST API for the predictive-maintenance model.

Run from the repository root with: python -m src.api
"""
from math import isfinite

from flask import Flask, jsonify, request

from src.config import FEATURES, MODEL_PATH
from src.predict import predict_machine

NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


def create_app():
    """Create the API application (also used by automated tests)."""
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify({
            "status": "ok",
            "model_ready": MODEL_PATH.exists(),
            "model_path": str(MODEL_PATH.name),
        }), 200

    @app.post("/predict")
    def predict():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({
                "error": "Request body must be a JSON object containing machine features."
            }), 400

        missing = [field for field in FEATURES if field not in payload]
        if missing:
            return jsonify({
                "error": "Missing required fields.",
                "missing_fields": missing,
                "required_fields": FEATURES,
            }), 400

        machine_type = payload["Type"]
        if not isinstance(machine_type, str) or machine_type not in {"L", "M", "H"}:
            return jsonify({
                "error": "Invalid machine type.",
                "field": "Type",
                "allowed_values": ["L", "M", "H"],
            }), 400

        clean_values = {"Type": machine_type}
        for field in NUMERIC_FEATURES:
            value = payload[field]
            # bool is a subclass of int in Python, but isn't a valid sensor reading.
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return jsonify({
                    "error": "Field must be a number.",
                    "field": field,
                }), 400
            if not isfinite(float(value)):
                return jsonify({
                    "error": "Field must be a finite number.",
                    "field": field,
                }), 400
            if value < 0:
                return jsonify({
                    "error": "Field cannot be negative.",
                    "field": field,
                }), 400
            clean_values[field] = value

        try:
            result = predict_machine(clean_values)
        except FileNotFoundError:
            return jsonify({
                "error": "The trained model was not found. Run 'python -m src.train' first."
            }), 503
        except (KeyError, ValueError) as exc:
            return jsonify({"error": f"Prediction could not be completed: {exc}"}), 400

        return jsonify(result), 200

    @app.get("/")
    def index():
        return jsonify({
            "name": "AI-Powered Predictive Maintenance API",
            "endpoints": {
                "health": "GET /health",
                "predict": "POST /predict",
            },
            "required_features": FEATURES,
        }), 200

    return app


app = create_app()

if __name__ == "__main__":
    # Local development server only; use a production WSGI server for deployment.
    app.run(host="127.0.0.1", port=5000, debug=False)

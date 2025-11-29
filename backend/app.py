from flask import Flask, request, jsonify
from inference import SurgePredictor
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
# Load models once at startup
predictor = SurgePredictor()


@app.route("/", methods=["GET"])
def index():
    return jsonify(
        {
            "message": "Backend is running",
            "endpoints": ["/health", "/predict"],
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    JSON body: { "horizon": "1h" | "1d" | "2d" }
    """
    data = request.get_json(silent=True) or {}
    horizon = data.get("horizon", "1h")
    try:
        result = predictor.predict_horizon(horizon=horizon)
        return jsonify({"ok": True, "result": result})
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:
        # for debug only
        return jsonify({"ok": False, "error": repr(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

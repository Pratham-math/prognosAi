from flask import Flask, request,jsonify
from agent_llm import run_surge_agent
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
@app.get("/")
def home():
    return {
        "message" : "Agent is alive!!!"
    }

@app.route("/agent", methods=["POST"])
def agent_endpoint():
    """
    JSON body:
      {
        "horizon": "1h" | "1d" | "2d",
        "location": "Mumbai"   # optional
      }
    Returns:
      {
        "ok": true,
        "agent_plan": { ... JSON schema from agent ... }
      }
    """
    data = request.get_json(silent=True) or {}
    horizon = data.get("horizon", "1d")
    location = data.get("location")

    try:
        agent_result = run_surge_agent(horizon=horizon, location=location)
        return jsonify({"ok": True, "agent_plan": agent_result})
    except Exception as e:
        return jsonify({"ok": False, "error": repr(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6969, debug=True)
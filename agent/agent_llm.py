import os
import json
from typing import Optional

import requests
import numpy as np

from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun


# --------- TOOL 1: call the local surge model backend --------- #

@tool
def call_surge_model(horizon: str = "1d") -> str:
    """
    Call the internal hospital surge forecasting API.

    horizon: one of "1h", "1d", or "2d".
    Returns a JSON string with keys:
      - horizon
      - total_predicted_admissions
      - forecast (list of {timestamp, predicted_admissions, xgb_only, lstm_only})
      - supply_plan (oxygen_cylinders, nebulizer_sets, burn_dressing_kits, emergency_beds_to_reserve)
    """
    if horizon not in ("1h", "1d", "2d"):
        raise ValueError("Invalid horizon. Use '1h', '1d', or '2d'.")

    url = "http://127.0.0.1:5000/predict"
    payload = {"horizon": horizon}
    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("ok"):
        raise RuntimeError(f"Model backend error: {data.get('error')}")
    return json.dumps(data["result"], default=float)


# --------- TOOL 2: web/news search to infer reasons --------- #

_search = DuckDuckGoSearchRun()

@tool
def news_search_for_surge(query: str) -> str:
    """
    Search web/news for events that could explain patient surges.

    Use this after you have a prediction, to check for:
    - festivals, holidays, mass gatherings
    - air pollution spikes, weather extremes
    - outbreaks / epidemics / public health alerts
    """
    return _search.run(query)


# --------- TOOL 3: deterministic load stats from forecast --------- #

@tool
def compute_load_stats(forecast_json: str) -> str:
    """
    Given a forecast JSON string (result of call_surge_model),
    compute key statistics: peak load, average load, and peak time window.

    Returns a short JSON string with:
      - peak_admissions
      - peak_timestamp
      - avg_admissions
    """
    data = json.loads(forecast_json)
    if "forecast" in data:
        forecast = data["forecast"]
    elif "result" in data and "forecast" in data["result"]:
        forecast = data["result"]["forecast"]
    else:
        raise ValueError("Invalid forecast_json format; 'forecast' key not found.")

    vals = [float(item["predicted_admissions"]) for item in forecast]
    ts = [item["timestamp"] for item in forecast]
    if not vals:
        raise ValueError("Empty forecast.")

    peak_idx = int(np.argmax(vals))
    stats = {
        "peak_admissions": vals[peak_idx],
        "peak_timestamp": ts[peak_idx],
        "avg_admissions": float(np.mean(vals)),
    }
    return json.dumps(stats, default=float)


# --------- MANUAL TOOL-CALLING ORCHESTRATION --------- #

SYSTEM_PROMPT = """
You are an autonomous hospital operations assistant for a single urban Indian hospital.
You have access to tools:
- call_surge_model(horizon): get predicted ED arrivals and baseline supply plan.
- news_search_for_surge(query): search web/news for likely causes (festivals, air quality, outbreaks).
- compute_load_stats(forecast_json): compute peak and average patient loads.

WORKFLOW:
1. ALWAYS call call_surge_model FIRST with the horizon requested by the user.
2. THEN call news_search_for_surge with a query that includes the city and date range you infer from the forecast timestamps.
3. OPTIONALLY call compute_load_stats if you need numeric metrics (peak load, mean load).
4. Finally, synthesize a playbook with concrete actions.

OUTPUT FORMAT:
Return your final answer as a SINGLE JSON object and NOTHING ELSE.
Do NOT include backticks, markdown, or commentary.

The JSON MUST have this exact top-level schema:
{
  "reason_for_surge": "<short explanation of likely drivers (festivals, pollution, outbreaks, etc.)>",
  "raw_prediction_summary": {
    "horizon": "<string, e.g. 1d>",
    "total_predicted_admissions": <float>,
    "peak_admissions": <float>,
    "peak_timestamp": "<ISO8601 timestamp string>",
    "avg_hourly_admissions": <float>
  },
  "staffing": [
    {
      "id": <int, unique within staffing array>,
      "title": "<short action title, e.g. 'Increase Pulmonology Staff'>",
      "description": "crete action description>",
      "priority": "<'high' | 'medium' | 'low'>",
      "status": "pending",
      "impact": "<'Critical' | 'High' | 'Medium' | 'Low'>",
      "effort": "<'High' | 'Medium' | 'Low'>"
    }
  ],
  "supplies": [
    {
      "id": <int, unique within supplies array>,
      "title": "<short action title, e.g. 'Stock Oxygen Cylinders'>",
      "description": "crete action with quantities>",
      "priority": "<'high' | 'medium' | 'low'>",
      "status": "pending",
      "impact": "<'Critical' | 'High' | 'Medium' | 'Low'>",
      "effort": "<'High' | 'Medium' | 'Low'>"
    }
  ],
  "advisories": [
    {
      "id": <int, unique within advisories array>,
      "title": "<short advisory title, e.g. 'Public Health Advisory'>",
      "description": "<plain-language message and target segment, e.g. 'Send SMS alerts to asthma patients about AQI spike'>",
      "priority": "<'high' | 'medium' | 'low'>",
      "status": "pending",
      "impact": "<'High' | 'Medium' | 'Low'>",
      "effort": "<'High' | 'Medium' | 'Low'>"
    }
  ]
}

CONSTRAINTS:
- Each action list (staffing, supplies, advisories) should have 2-4 items sorted by priority descending.
- All status fields should default to "pending" unless an action is already completed (use your judgment).
- Use realistic numbers for staffing (additional doctors/nurses) and supplies (oxygen cylinders, nebulizers, etc.).
- Assume hospital baseline is ~120 ED visits per day; treat forecasts significantly above this as surge.
- priority: use 'high' for critical/urgent actions, 'medium' for important but not immediate, 'low' for nice-to-have.
- impact: rate the consequence if action is NOT taken.
- effort: rate the resources/time needed to execute the action.
"""


def run_surge_agent(
    horizon: str,
    location: Optional[str] = None,
) -> dict:
    """
    High-level wrapper used by Flask.

    horizon: "1h", "1d", or "2d"
    location: optional city/region string to bias news search.
    """
    if location:
        loc_text = f" The hospital is located in {location}."
    else:
        loc_text = " The hospital is in a large Indian city."

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY environment variable not set. "
            "Get your key from https://aistudio.google.com/app/apikey"
        )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.2,
        google_api_key=api_key,
    )

    # Step 1: call surge model
    try:
        forecast_json = call_surge_model.run(horizon)
    except Exception as e:
        return {"error": f"Failed to call surge model: {e}"}

    # Step 2: extract timestamps for news query context
    try:
        forecast_data = json.loads(forecast_json)
        first_ts = forecast_data.get("forecast", [{}])[0].get("timestamp", "")
        date_hint = first_ts[:10] if first_ts else "recent"
        news_query = f"{location or 'India'} health pollution festival {date_hint}"
        news_result = news_search_for_surge.run(news_query)
    except Exception as e:
        news_result = f"News search failed: {e}"

    # Step 3: optionally compute load stats
    try:
        stats_json = compute_load_stats.run(forecast_json)
    except Exception as e:
        stats_json = f'{{"error": "Stats computation failed: {e}"}}'

    # Step 4: ask LLM to synthesize playbook-format JSON
    user_instruction = (
        SYSTEM_PROMPT
        + f"\n\nTASK:\n"
        f"For forecast horizon '{horizon}', you have the following data:\n\n"
        f"1. Surge model forecast:\n{forecast_json}\n\n"
        f"2. News/web search results:\n{news_result}\n\n"
        f"3. Load statistics:\n{stats_json}\n\n"
        f"Now produce a playbook with staffing actions, supply actions, and patient advisories.\n"
        f"Return output strictly following the required JSON schema (with staffing, supplies, advisories arrays).\n"
        f"{loc_text}"
    )

    # Invoke LLM to get final answer
    response = llm.invoke(user_instruction)
    output_text = response.content if hasattr(response, "content") else str(response)

    # Parse JSON
    try:
        parsed = json.loads(output_text)
    except json.JSONDecodeError:
        parsed = {"error": "Agent did not return valid JSON", "raw": output_text}

    return parsed

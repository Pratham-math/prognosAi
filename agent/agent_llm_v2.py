"""Enhanced agent with multi-source context aggregation and structured reasoning."""

import os
import json
from typing import Optional, Dict, List
from datetime import datetime, timedelta

import requests
import numpy as np

from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun


# ==================== TOOLS ====================

@tool
def call_surge_model(horizon: str = "1d") -> str:
    """
    Call hospital surge forecasting API.
    
    Args:
        horizon: "1h", "1d", or "2d"
    
    Returns:
        JSON with forecast, supply plan, and confidence metrics
    """
    if horizon not in ("1h", "1d", "2d"):
        raise ValueError("Invalid horizon. Use '1h', '1d', or '2d'.")
    
    url = "http://127.0.0.1:5000/predict"
    try:
        resp = requests.post(url, json={"horizon": horizon}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok"):
            raise RuntimeError(f"Model error: {data.get('error')}")
        return json.dumps(data["result"], default=float)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def search_health_events(location: str, date_range: str, keywords: str) -> str:
    """
    Search for health-related events, outbreaks, pollution alerts.
    
    Args:
        location: City/region (e.g., "Mumbai", "Delhi NCR")
        date_range: Date context (e.g., "December 2025", "next week")
        keywords: Search terms (e.g., "air pollution", "dengue outbreak", "festival")
    
    Returns:
        Web search results with recent news and alerts
    """
    search = DuckDuckGoSearchRun()
    query = f"{location} {date_range} {keywords} health hospital"
    try:
        return search.run(query)
    except Exception as e:
        return f"Search failed: {str(e)}"


@tool
def search_pollution_data(location: str, date_hint: str) -> str:
    """
    Search for air quality index (AQI) and pollution forecasts.
    
    Args:
        location: City/region
        date_hint: Time context for forecast
    
    Returns:
        AQI levels, PM2.5 data, pollution forecasts
    """
    search = DuckDuckGoSearchRun()
    query = f"{location} AQI air quality PM2.5 forecast {date_hint}"
    try:
        return search.run(query)
    except Exception as e:
        return f"AQI search failed: {str(e)}"


@tool
def search_local_events(location: str, date_range: str) -> str:
    """
    Search for local festivals, gatherings, sports events.
    
    Args:
        location: City/region
        date_range: Time window
    
    Returns:
        Information about mass gatherings that could increase ED visits
    """
    search = DuckDuckGoSearchRun()
    query = f"{location} {date_range} festival event gathering schedule"
    try:
        return search.run(query)
    except Exception as e:
        return f"Event search failed: {str(e)}"


@tool
def compute_surge_statistics(forecast_json: str) -> str:
    """
    Compute detailed statistics from forecast.
    
    Args:
        forecast_json: Raw forecast output from surge model
    
    Returns:
        Peak load, average, variance, surge windows, percentile thresholds
    """
    try:
        data = json.loads(forecast_json)
        
        # Handle nested structure
        if "forecast" in data:
            forecast = data["forecast"]
        elif "result" in data and "forecast" in data["result"]:
            forecast = data["result"]["forecast"]
        else:
            return json.dumps({"error": "Invalid forecast format"})
        
        if not forecast:
            return json.dumps({"error": "Empty forecast"})
        
        vals = [float(item["predicted_admissions"]) for item in forecast]
        ts = [item["timestamp"] for item in forecast]
        
        # Statistics
        vals_arr = np.array(vals)
        peak_idx = int(np.argmax(vals_arr))
        
        # Identify surge windows (> 90th percentile)
        p90 = np.percentile(vals_arr, 90)
        surge_hours = [ts[i] for i, v in enumerate(vals) if v > p90]
        
        # Hour-over-hour changes
        deltas = np.diff(vals_arr)
        max_increase_idx = int(np.argmax(deltas))
        
        stats = {
            "peak_admissions": float(vals[peak_idx]),
            "peak_timestamp": ts[peak_idx],
            "avg_admissions": float(np.mean(vals_arr)),
            "std_admissions": float(np.std(vals_arr)),
            "p50_admissions": float(np.percentile(vals_arr, 50)),
            "p90_admissions": float(np.percentile(vals_arr, 90)),
            "p95_admissions": float(np.percentile(vals_arr, 95)),
            "surge_window_count": len(surge_hours),
            "surge_windows": surge_hours[:10],  # First 10 surge hours
            "max_hourly_increase": float(deltas[max_increase_idx]) if len(deltas) > 0 else 0.0,
            "max_increase_timestamp": ts[max_increase_idx] if len(deltas) > 0 else ""
        }
        
        return json.dumps(stats, default=float)
    
    except Exception as e:
        return json.dumps({"error": f"Stats computation failed: {str(e)}"})


@tool
def estimate_resource_needs(peak_admissions: float, avg_admissions: float, surge_duration_hours: int) -> str:
    """
    Calculate resource requirements based on predicted load.
    
    Args:
        peak_admissions: Peak hourly admissions
        avg_admissions: Average hourly admissions
        surge_duration_hours: Duration of surge period
    
    Returns:
        Staffing, supplies, and bed allocation recommendations
    """
    # Baseline assumptions (adjust per hospital)
    BASELINE_HOURLY = 5.0  # Normal hourly admissions
    NURSE_PER_5_PATIENTS = 1
    DOCTOR_PER_10_PATIENTS = 1
    
    surge_factor = peak_admissions / BASELINE_HOURLY
    
    # Staffing
    extra_nurses = max(0, int((peak_admissions - BASELINE_HOURLY) / 5))
    extra_doctors = max(0, int((peak_admissions - BASELINE_HOURLY) / 10))
    extra_support_staff = max(0, int(extra_nurses / 3))
    
    # Supplies (per surge duration)
    total_surge_patients = int(avg_admissions * surge_duration_hours)
    oxygen_cylinders = max(10, int(total_surge_patients * 0.25))  # 25% need oxygen
    nebulizers = max(5, int(total_surge_patients * 0.15))  # 15% respiratory
    iv_sets = max(20, int(total_surge_patients * 0.60))  # 60% need IV
    emergency_beds = max(5, int(peak_admissions * 1.5))  # 1.5x peak for buffer
    
    # Medications (estimate based on common ED cases)
    saline_bags = iv_sets
    analgesics = int(total_surge_patients * 0.70)
    antibiotics = int(total_surge_patients * 0.30)
    
    resources = {
        "staffing": {
            "additional_nurses": extra_nurses,
            "additional_doctors": extra_doctors,
            "additional_support_staff": extra_support_staff,
            "justification": f"Surge factor: {surge_factor:.2f}x baseline"
        },
        "supplies": {
            "oxygen_cylinders": oxygen_cylinders,
            "nebulizer_sets": nebulizers,
            "iv_sets": iv_sets,
            "saline_bags": saline_bags,
            "analgesics_doses": analgesics,
            "antibiotic_courses": antibiotics
        },
        "infrastructure": {
            "emergency_beds_to_ready": emergency_beds,
            "isolation_rooms_if_outbreak": max(3, int(emergency_beds * 0.2))
        }
    }
    
    return json.dumps(resources, default=float)


# ==================== AGENT ORCHESTRATION ====================

SYSTEM_PROMPT = """
You are an autonomous hospital operations AI for a large urban Indian hospital ED.

You have access to these tools:
1. call_surge_model(horizon): Get ML forecast for ED admissions
2. search_health_events(location, date_range, keywords): Search health news/outbreaks
3. search_pollution_data(location, date_hint): Get AQI and pollution forecasts
4. search_local_events(location, date_range): Find festivals/gatherings
5. compute_surge_statistics(forecast_json): Calculate detailed surge metrics
6. estimate_resource_needs(peak, avg, duration): Calculate staffing/supply needs

WORKFLOW:
1. Call call_surge_model() with user-requested horizon
2. Extract timestamps from forecast to determine date range
3. Call ALL context tools in parallel:
   - search_health_events() for disease outbreaks, health alerts
   - search_pollution_data() for AQI levels
   - search_local_events() for festivals/mass gatherings
4. Call compute_surge_statistics() to get peak/avg metrics
5. Call estimate_resource_needs() with statistics
6. Synthesize findings into structured playbook JSON

OUTPUT FORMAT:
Return ONLY a single valid JSON object (no markdown, no commentary).

Schema:
{
  "reason_for_surge": "<detailed 2-3 sentence explanation citing specific causes>",
  "confidence": "<high|medium|low>",
  "raw_prediction_summary": {
    "horizon": "<string>",
    "total_predicted_admissions": <float>,
    "peak_admissions": <float>,
    "peak_timestamp": "<ISO8601>",
    "avg_hourly_admissions": <float>,
    "surge_factor": <float>
  },
  "staffing": [
    {
      "id": <int>,
      "title": "<concise action title>",
      "description": "<specific details with numbers>",
      "priority": "<high|medium|low>",
      "status": "pending",
      "impact": "<Critical|High|Medium|Low>",
      "effort": "<High|Medium|Low>",
      "timeline": "<e.g., 'Immediate', 'Within 24h', '2-3 days'>"
    }
  ],
  "supplies": [
    {
      "id": <int>,
      "title": "<action title>",
      "description": "<specific quantities and justification>",
      "priority": "<high|medium|low>",
      "status": "pending",
      "impact": "<Critical|High|Medium|Low>",
      "effort": "<High|Medium|Low>",
      "timeline": "<timeframe>"
    }
  ],
  "advisories": [
    {
      "id": <int>,
      "title": "<advisory title>",
      "description": "<patient communication strategy>",
      "priority": "<high|medium|low>",
      "status": "pending",
      "impact": "<High|Medium|Low>",
      "effort": "<High|Medium|Low>",
      "target_audience": "<e.g., 'Asthma patients', 'General public'>"
    }
  ]
}

CONSTRAINTS:
- Each array should have 3-5 items sorted by priority (high first)
- All numbers must be realistic and justified by the data
- Use specific diseases, pollutants, events found in search results
- Baseline: ~120 ED visits/day. Treat >150/day as surge.
- Timeline: Immediate = <6h, Within 24h = 6-24h, 2-3 days = 24-72h
- If search tools fail, state "limited external data" in reason_for_surge
"""


def run_enhanced_agent(horizon: str, location: Optional[str] = None) -> Dict:
    """Main agent orchestration with parallel context gathering."""
    
    location = location or "Mumbai"  # Default to Mumbai
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return {"error": "GOOGLE_API_KEY not set"}
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",  # Latest Gemini
        temperature=0.1,  # Lower temp for more factual outputs
        google_api_key=api_key,
    )
    
    # Step 1: Get forecast
    try:
        forecast_json = call_surge_model.run(horizon)
        forecast_data = json.loads(forecast_json)
    except Exception as e:
        return {"error": f"Forecast failed: {str(e)}"}
    
    # Step 2: Extract date context
    try:
        first_ts = forecast_data.get("forecast", [{}])[0].get("timestamp", "")
        if first_ts:
            date_obj = datetime.fromisoformat(first_ts.replace("Z", "+00:00"))
            date_range = date_obj.strftime("%B %d-%d, %Y")  # e.g., "December 04-05, 2025"
            date_hint = date_obj.strftime("%Y-%m-%d")
        else:
            date_range = "upcoming days"
            date_hint = "recent"
    except:
        date_range = "upcoming days"
        date_hint = "recent"
    
    # Step 3: Gather context (run searches)
    health_events = search_health_events.run(location, date_range, "outbreak disease epidemic")
    pollution_data = search_pollution_data.run(location, date_hint)
    local_events = search_local_events.run(location, date_range)
    
    # Step 4: Compute statistics
    stats_json = compute_surge_statistics.run(forecast_json)
    stats = json.loads(stats_json)
    
    # Step 5: Estimate resources
    if "peak_admissions" in stats and "avg_admissions" in stats:
        horizon_hours = {"1h": 1, "1d": 24, "2d": 48}[horizon]
        resources_json = estimate_resource_needs.run(
            stats["peak_admissions"],
            stats["avg_admissions"],
            horizon_hours
        )
    else:
        resources_json = json.dumps({"error": "Could not compute resources"})
    
    # Step 6: Build prompt with all context
    context = f"""
TASK: Generate operations playbook for {location} hospital ED.

HORIZON: {horizon}

=== FORECAST DATA ===
{forecast_json}

=== SURGE STATISTICS ===
{stats_json}

=== RESOURCE ESTIMATES ===
{resources_json}

=== HEALTH EVENTS & OUTBREAKS ===
{health_events}

=== POLLUTION DATA ===
{pollution_data}

=== LOCAL EVENTS & GATHERINGS ===
{local_events}

=== INSTRUCTIONS ===
{SYSTEM_PROMPT}

Now synthesize this into the required JSON playbook format.
    """
    
    # Step 7: LLM synthesis
    try:
        response = llm.invoke(context)
        output_text = response.content if hasattr(response, "content") else str(response)
        
        # Clean markdown if present
        if "```json" in output_text:
            output_text = output_text.split("```json")[1].split("```")[0]
        elif "```" in output_text:
            output_text = output_text.split("```")[1].split("```")[0]
        
        parsed = json.loads(output_text.strip())
        return parsed
    
    except json.JSONDecodeError as e:
        return {
            "error": "Agent returned invalid JSON",
            "raw_output": output_text[:500],
            "parse_error": str(e)
        }
    except Exception as e:
        return {"error": f"Agent execution failed: {str(e)}"}


if __name__ == "__main__":
    # Test agent
    result = run_enhanced_agent("1d", "Mumbai")
    print(json.dumps(result, indent=2))

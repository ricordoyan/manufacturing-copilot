"""
Prompt assembly and NVIDIA NIM LLM invocation.

This is the orchestration layer of the RAG pipeline:
  1. Pull defect summary + sensor context from SQLite.
  2. Retrieve relevant document chunks from FAISS.
  3. Assemble a structured prompt with system instructions.
  4. Call the NVIDIA NIM LLM (OpenAI-compatible chat endpoint).
  5. Return the answer plus source citations, latency metrics,
     and supporting sensor metrics for the UI correlation panel.
"""

import pandas as pd
from openai import OpenAI

from config import (
    NVIDIA_API_KEY,
    NVIDIA_BASE_URL,
    NVIDIA_LLM_MODEL,
    SENSOR_DATA_PATH,
    TEMP_WARNING,
)
from db.database import get_defect_summary, get_sensor_context
from rag.ingest import load_faiss_index
from rag.retriever import retrieve_relevant_docs
from utils.metrics import LatencyTracker

# ── NVIDIA NIM client ───────────────────────────────────────────────────────
_client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)

# ── System prompt ───────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are an expert manufacturing copilot assistant for a metal forming production facility.
You help operators understand defect patterns by correlating real-time sensor data with historical incidents and maintenance records.

ANALYSIS RULES — follow these strictly:

1. ROOT CAUSE ANALYSIS: Always trace the causal chain to the deepest root cause.
   - If temperature is high, explain WHY (e.g., coolant flow dropped because valve V-17 drifted).
   - If defect rate is high, explain the chain: equipment issue → sensor change → defect pattern.
   - Never stop at symptoms. Always dig one level deeper.

2. HISTORICAL CORRELATION: Always check if similar incidents occurred before.
   - If you find a matching QA incident report, cite it with the report ID and date.
   - Compare the current sensor pattern to the historical pattern (e.g., "Similar to IR-2023-0914-L3 where coolant flow dropped to 71%").

3. SPECIFIC REFERENCES: Always cite specific equipment IDs, valve numbers, work orders, and SOP procedure numbers.
   - Example: "Inspect cooling valve V-17 per SOP-002 Section 6"
   - Example: "Reduce line speed per SOP-003 Section 4"
   - Never give generic advice like "adjust heating elements."

4. CITATIONS: For every claim from a historical document, include the source filename in brackets, e.g., [QA-Report-2023-09-14.md].

5. RESPONSE FORMAT:
   - Start with a 1-2 sentence executive summary
   - Then provide: What happened (with timestamps), Root cause chain, Historical precedent, Recommended actions
   - Keep the total response under 300 words
   - Use bold for key values like temperatures, valve IDs, and timestamps"""


# ── Trend helper ────────────────────────────────────────────────────────────

def _get_trend_description(values: list[float], label: str) -> str:
    """Describe whether a metric is rising, falling, or stable over the window."""
    if len(values) < 2:
        return "insufficient data"
    half = len(values) // 2
    first_avg = sum(values[:half]) / half
    second_avg = sum(values[half:]) / max(len(values) - half, 1)
    diff = second_avg - first_avg
    if diff > 3:
        return f"rising (increased by {diff:.1f} over the window)"
    elif diff < -3:
        return f"falling (decreased by {abs(diff):.1f} over the window)"
    return "stable"


def _load_trend_series(hours: float, line_id: str) -> dict:
    """Load raw sensor time-series from CSV for trend calculation."""
    try:
        df = pd.read_csv(SENSOR_DATA_PATH, parse_dates=["timestamp"])
        if line_id:
            df = df[df["line_id"] == line_id]
        # Use the last N rows proportional to requested hours (5-min intervals)
        n_rows = int(hours * 12)  # 12 rows per hour at 5-min spacing
        window = df.tail(n_rows) if n_rows < len(df) else df
        return {
            "temp_values": window["forming_zone_temp_c"].dropna().tolist(),
            "flow_values": window["coolant_flow_pct"].dropna().tolist(),
            "speed_values": window["line_speed_mpm"].dropna().tolist(),
        }
    except Exception:
        return {"temp_values": [], "flow_values": [], "speed_values": []}


# ── Prompt builder ──────────────────────────────────────────────────────────

def build_rag_prompt(
    user_question: str,
    defect_summary: dict,
    sensor_context: dict,
    relevant_docs: list[dict],
    time_window_hours: float = 1.0,
    line_id: str = "LINE-3",
) -> str:
    """Assemble the full prompt that the LLM will receive.

    The prompt has clearly labelled sections so the model can distinguish
    factual context from the operator's question.
    """
    # ── Defect summary ──────────────────────────────────────────────────
    ds = defect_summary
    defect_count = ds.get("total_defects", 0)
    avg_rate = ds.get("defect_rate_avg", 0)
    by_type = ds.get("by_type", {})
    type_breakdown = ", ".join(f"{t}: {c}" for t, c in by_type.items()) if by_type else "none"

    # ── Sensor context ──────────────────────────────────────────────────
    sc = sensor_context
    ft = sc.get("forming_zone_temp", {})
    peak_temp = ft.get("max", "N/A")
    peak_temp_time = sc.get("peak_temp_timestamp", "N/A")

    cl = sc.get("coolant_flow", {})
    current_flow = cl.get("min", "N/A")  # worst case in window

    ls = sc.get("line_speed", {})
    current_speed = ls.get("avg", "N/A")

    # ── Trends ──────────────────────────────────────────────────────────
    trends = _load_trend_series(time_window_hours, line_id)
    temp_trend = _get_trend_description(trends["temp_values"], "temperature")
    flow_trend = _get_trend_description(trends["flow_values"], "coolant flow")

    # ── Documents ───────────────────────────────────────────────────────
    if relevant_docs:
        doc_parts: list[str] = []
        for i, doc in enumerate(relevant_docs, 1):
            doc_parts.append(
                f"### [{doc['source']}] (relevance score: {doc['score']})\n{doc['content']}"
            )
        formatted_docs = "\n\n".join(doc_parts)
    else:
        formatted_docs = (
            "No documents available. Answer based on sensor data "
            "and general manufacturing knowledge."
        )

    prompt = f"""## Current Sensor Data (Last {time_window_hours}h on {line_id})
Defect Summary: {defect_count} defects detected, average defect rate: {avg_rate}%
Defect Types: {type_breakdown}
Peak Forming Zone Temperature: {peak_temp}°C at {peak_temp_time}
Min Coolant Flow in Window: {current_flow}% (nominal: 97-100%)
Average Line Speed: {current_speed} m/min (nominal: 45 m/min)
Temperature Trend: {temp_trend}
Coolant Flow Trend: {flow_trend}

## Relevant Historical Documents
{formatted_docs}

## Operator Question
{user_question}

Provide your analysis following the rules above."""

    return prompt


# ── LLM call ────────────────────────────────────────────────────────────────

def call_nvidia_llm(prompt: str) -> str:
    """Call the NVIDIA NIM LLM endpoint and return the text response.

    Uses ``temperature=0.3`` for deterministic, fact-grounded answers —
    manufacturing operators need reliability over creativity.
    """
    try:
        response = _client.chat.completions.create(
            model=NVIDIA_LLM_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        return f"❌ LLM API error: {exc}"


# ── Main orchestration ──────────────────────────────────────────────────────

def query_copilot(
    user_question: str,
    time_window_hours: float = 1.0,
    line_id: str = "LINE-3",
) -> dict:
    """End-to-end RAG pipeline: DB → FAISS → prompt → LLM → response.

    Returns
    -------
    dict
        ``answer``           – the LLM response text
        ``sources``          – list of source document filenames referenced
        ``latency_breakdown``– per-step latency dict
        ``metrics``          – supporting sensor metrics for the UI panel
    """
    tracker = LatencyTracker()

    # Step 1 — Defect summary
    tracker.start("db_defect_summary")
    defect_summary = get_defect_summary(hours=time_window_hours, line_id=line_id)
    tracker.stop("db_defect_summary")

    # Step 2 — Sensor context
    tracker.start("db_sensor_context")
    sensor_context = get_sensor_context(hours=time_window_hours, line_id=line_id)
    tracker.stop("db_sensor_context")

    # Step 3 — RAG retrieval
    tracker.start("rag_retrieval")
    index, chunks = load_faiss_index()
    relevant_docs: list[dict] = []
    if index is not None and chunks is not None:
        relevant_docs = retrieve_relevant_docs(user_question, index, chunks, top_k=8)
    tracker.stop("rag_retrieval")

    # Step 4 — Build prompt
    tracker.start("prompt_assembly")
    prompt = build_rag_prompt(
        user_question, defect_summary, sensor_context, relevant_docs,
        time_window_hours=time_window_hours, line_id=line_id,
    )
    tracker.stop("prompt_assembly")

    # Step 5 — LLM call
    tracker.start("llm_call")
    answer = call_nvidia_llm(prompt)
    tracker.stop("llm_call")

    sources = list({doc["source"] for doc in relevant_docs})
    latency = tracker.get_metrics()

    # ── Supporting metrics for the UI correlation panel ──────────────────
    ft = sensor_context.get("forming_zone_temp", {})
    cl = sensor_context.get("coolant_flow", {})
    max_temp = ft.get("max", 0)
    min_flow = cl.get("min", 0)

    return {
        "answer": answer,
        "sources": sources,
        "latency_breakdown": latency,
        "metrics": {
            "peak_temp": max_temp,
            "temp_above_threshold": round(max(0, max_temp - TEMP_WARNING), 1),
            "min_flow": min_flow,
            "flow_below_nominal": round(max(0, 98 - min_flow), 1),
            "defect_count": defect_summary.get("total_defects", 0),
            "rate_vs_baseline": f"+{max(0, defect_summary.get('defect_rate_avg', 0) - 2.0):.1f}%",
            "total_latency": latency.get("total", 0),
        },
    }

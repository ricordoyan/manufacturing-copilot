"""
SQLite helpers for defect-event logging.

All defect events detected by the simulator or the video processor are
persisted here so that the RAG generator and the Streamlit dashboard can
query them with time-window filters.
"""

import os
import sqlite3
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from config import SQLITE_DB_PATH, SENSOR_DATA_PATH


# ── Helpers ─────────────────────────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    """Return a connection with row-factory enabled for dict-like access."""
    os.makedirs(os.path.dirname(SQLITE_DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _latest_timestamp() -> str:
    """Return the most recent timestamp in the defect_events table.

    Time-window queries are relative to the *data* clock, not wall-clock,
    because the dataset is from 2024-01-15.
    """
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT MAX(timestamp) AS ts FROM defect_events"
        ).fetchone()
        return row["ts"] if row and row["ts"] else datetime.utcnow().isoformat()
    finally:
        conn.close()


# ── Public API ──────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create the defect_events table if it does not already exist."""
    conn = _connect()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS defect_events (
                id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp             TEXT    NOT NULL,
                line_id               TEXT    NOT NULL,
                defect_type           TEXT    NOT NULL,
                confidence            REAL    NOT NULL,
                forming_zone_temp_c   REAL,
                cooling_zone_temp_c   REAL,
                line_speed_mpm        REAL,
                hydraulic_pressure_bar REAL,
                coolant_flow_pct      REAL,
                defect_rate_pct       REAL,
                source_image          TEXT
            )
        """)
        conn.commit()
    finally:
        conn.close()


def log_defect_event(event: dict) -> None:
    """Insert a single defect event dict into the database."""
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO defect_events
                (timestamp, line_id, defect_type, confidence,
                 forming_zone_temp_c, cooling_zone_temp_c,
                 line_speed_mpm, hydraulic_pressure_bar,
                 coolant_flow_pct, defect_rate_pct, source_image)
            VALUES
                (:timestamp, :line_id, :defect_type, :confidence,
                 :forming_zone_temp_c, :cooling_zone_temp_c,
                 :line_speed_mpm, :hydraulic_pressure_bar,
                 :coolant_flow_pct, :defect_rate_pct, :source_image)
            """,
            {
                "timestamp": event.get("timestamp", datetime.utcnow().isoformat()),
                "line_id": event.get("line_id", "LINE-3"),
                "defect_type": event.get("defect_type", "unknown"),
                "confidence": event.get("confidence", 0.0),
                "forming_zone_temp_c": event.get("forming_zone_temp_c"),
                "cooling_zone_temp_c": event.get("cooling_zone_temp_c"),
                "line_speed_mpm": event.get("line_speed_mpm"),
                "hydraulic_pressure_bar": event.get("hydraulic_pressure_bar"),
                "coolant_flow_pct": event.get("coolant_flow_pct"),
                "defect_rate_pct": event.get("defect_rate_pct"),
                "source_image": event.get("source_image"),
            },
        )
        conn.commit()
    finally:
        conn.close()


def get_recent_defects(hours: float = 1, line_id: Optional[str] = None) -> list[dict]:
    """Return defect events from the last *hours* of **data** time.

    Because the dataset is fixed (2024-01-15), the window is measured
    backwards from the newest timestamp in the table, not from wall-clock.
    """
    latest = _latest_timestamp()
    cutoff = (
        datetime.fromisoformat(latest) - timedelta(hours=hours)
    ).isoformat(sep=" ")

    query = "SELECT * FROM defect_events WHERE timestamp >= ?"
    params: list = [cutoff]

    if line_id:
        query += " AND line_id = ?"
        params.append(line_id)

    query += " ORDER BY timestamp DESC"

    conn = _connect()
    try:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_defect_summary(hours: float = 1, line_id: Optional[str] = None) -> dict:
    """Aggregated statistics over the requested time window.

    Returns:
        {
            "total_defects": int,
            "defect_rate_avg": float,
            "by_type": {"surface_crack": int, ...},
            "avg_temp_during_defects": float,
            "time_window_hours": float,
            "latest_timestamp": str,
        }
    """
    defects = get_recent_defects(hours, line_id)
    total = len(defects)

    by_type: dict[str, int] = {}
    temp_sum = 0.0
    rate_sum = 0.0
    for d in defects:
        dt = d.get("defect_type", "unknown")
        by_type[dt] = by_type.get(dt, 0) + 1
        temp_sum += d.get("forming_zone_temp_c", 0) or 0
        rate_sum += d.get("defect_rate_pct", 0) or 0

    return {
        "total_defects": total,
        "defect_rate_avg": round(rate_sum / total, 2) if total else 0.0,
        "by_type": by_type,
        "avg_temp_during_defects": round(temp_sum / total, 1) if total else 0.0,
        "time_window_hours": hours,
        "latest_timestamp": _latest_timestamp(),
    }


def get_sensor_context(hours: float = 1, line_id: Optional[str] = None) -> dict:
    """Sensor reading statistics over the requested time window.

    Queries *all* rows in the defect_events table (not only defect=true),
    but since we only store rows where defect_detected is true, we also
    fall back to the sensor CSV for a fuller picture.

    Returns min/max/avg of key sensors and the timestamp of peak temperature.
    """
    latest = _latest_timestamp()
    cutoff = (
        datetime.fromisoformat(latest) - timedelta(hours=hours)
    ).isoformat(sep=" ")

    # Try to read from the sensor CSV for full context (including non-defect rows)
    try:
        df = pd.read_csv(SENSOR_DATA_PATH, parse_dates=["timestamp"])
        df["timestamp"] = df["timestamp"].astype(str)
        mask = df["timestamp"] >= cutoff
        if line_id:
            mask &= df["line_id"] == line_id
        window = df[mask]
    except Exception:
        # Fallback: use only DB rows
        defects = get_recent_defects(hours, line_id)
        window = pd.DataFrame(defects)

    if window.empty:
        return {
            "forming_zone_temp": {"min": 0, "max": 0, "avg": 0},
            "cooling_zone_temp": {"min": 0, "max": 0, "avg": 0},
            "line_speed": {"min": 0, "max": 0, "avg": 0},
            "hydraulic_pressure": {"min": 0, "max": 0, "avg": 0},
            "coolant_flow": {"min": 0, "max": 0, "avg": 0},
            "peak_temp_timestamp": None,
            "time_window_hours": hours,
        }

    def _stats(col: str) -> dict:
        s = window[col].dropna()
        return {
            "min": round(float(s.min()), 2) if len(s) else 0,
            "max": round(float(s.max()), 2) if len(s) else 0,
            "avg": round(float(s.mean()), 2) if len(s) else 0,
        }

    peak_idx = window["forming_zone_temp_c"].idxmax()
    peak_ts = str(window.loc[peak_idx, "timestamp"]) if pd.notna(peak_idx) else None

    return {
        "forming_zone_temp": _stats("forming_zone_temp_c"),
        "cooling_zone_temp": _stats("cooling_zone_temp_c"),
        "line_speed": _stats("line_speed_mpm"),
        "hydraulic_pressure": _stats("hydraulic_pressure_bar"),
        "coolant_flow": _stats("coolant_flow_pct"),
        "peak_temp_timestamp": peak_ts,
        "time_window_hours": hours,
    }


def populate_from_sensor_csv() -> int:
    """Read sensor_data.csv and insert rows where defect_detected == True.

    Returns the number of events inserted.
    """
    df = pd.read_csv(SENSOR_DATA_PATH)
    defective = df[df["defect_detected"] == True].copy()  # noqa: E712

    count = 0
    for _, row in defective.iterrows():
        # Use the confidence value from the CSV if present, otherwise derive one
        csv_conf = row.get("confidence", "")
        if csv_conf != "" and pd.notna(csv_conf):
            conf = round(float(csv_conf), 2)
        else:
            conf = round(0.70 + 0.25 * (row.get("defect_rate_pct", 3) / 10), 2)

        event = {
            "timestamp": row["timestamp"],
            "line_id": row["line_id"],
            "defect_type": row["defect_type"],
            "confidence": conf,
            "forming_zone_temp_c": row.get("forming_zone_temp_c"),
            "cooling_zone_temp_c": row.get("cooling_zone_temp_c"),
            "line_speed_mpm": row.get("line_speed_mpm"),
            "hydraulic_pressure_bar": row.get("hydraulic_pressure_bar"),
            "coolant_flow_pct": row.get("coolant_flow_pct"),
            "defect_rate_pct": row.get("defect_rate_pct"),
            "source_image": "sensor_simulation",
        }
        log_defect_event(event)
        count += 1

    return count


def get_all_sensor_data(line_id: Optional[str] = "LINE-3") -> pd.DataFrame:
    """Return all sensor readings from sensor_data.csv for charting.

    Loads the full CSV — not just defect events — so dashboard charts show
    continuous sensor trends including normal operating periods.
    """
    try:
        df = pd.read_csv(SENSOR_DATA_PATH, parse_dates=["timestamp"])
        if line_id:
            df = df[df["line_id"] == line_id]
        return df
    except FileNotFoundError:
        return pd.DataFrame()

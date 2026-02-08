"""
Defect simulator — replays the sensor CSV as a time-compressed event stream.

This is the **primary demo path**.  It guarantees a consistent, repeatable
narrative (cooling-valve drift → temperature rise → defect spike) without
depending on image-detection accuracy.  The video processor is a bonus
feature; this simulator is what the copilot queries in the default mode.
"""

import time
from datetime import datetime
from typing import Generator

import pandas as pd

from config import SENSOR_DATA_PATH


class DefectSimulator:
    """Replay sensor_data.csv rows as defect events in compressed time."""

    def __init__(self, sensor_data_path: str = SENSOR_DATA_PATH) -> None:
        self.df = pd.read_csv(sensor_data_path)
        # Ensure timestamp column is string for consistency
        self.df["timestamp"] = self.df["timestamp"].astype(str)

    def run_simulation(
        self, speed_multiplier: float = 10.0
    ) -> Generator[dict, None, None]:
        """Iterate through sensor rows, yielding defect events as they "occur".

        Parameters
        ----------
        speed_multiplier : float
            How much to compress time.  The CSV has 5-minute intervals;
            ``speed_multiplier=10`` makes each interval pass in 30 seconds.

        Yields
        ------
        dict
            One dict per row where ``defect_detected`` is True, with keys
            matching the ``defect_events`` table schema.
        """
        real_interval_sec = (5 * 60) / speed_multiplier  # seconds between rows

        prev_ts: datetime | None = None

        for _, row in self.df.iterrows():
            current_ts = datetime.fromisoformat(str(row["timestamp"]))

            # Sleep to simulate passage of time (skip sleep for the first row)
            if prev_ts is not None:
                time.sleep(real_interval_sec)
            prev_ts = current_ts

            # Only yield rows where a defect was flagged
            if not row.get("defect_detected", False):
                continue

            yield {
                "timestamp": row["timestamp"],
                "line_id": row.get("line_id", "LINE-3"),
                "defect_type": row.get("defect_type", "unknown"),
                # Confidence derived from defect rate — higher rate → higher confidence
                "confidence": round(
                    min(0.99, 0.70 + 0.25 * (row.get("defect_rate_pct", 3) / 10)),
                    2,
                ),
                "forming_zone_temp_c": row.get("forming_zone_temp_c"),
                "cooling_zone_temp_c": row.get("cooling_zone_temp_c"),
                "line_speed_mpm": row.get("line_speed_mpm"),
                "hydraulic_pressure_bar": row.get("hydraulic_pressure_bar"),
                "coolant_flow_pct": row.get("coolant_flow_pct"),
                "defect_rate_pct": row.get("defect_rate_pct"),
                "source_image": "sensor_simulation",
            }

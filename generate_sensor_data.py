"""
generate_sensor_data.py

Generates synthetic sensor data for Production Line 3 on 2024-01-15.
The data tells a coherent story that aligns with all 8 manufacturing documents:

Timeline narrative:
  06:00–09:00  Normal operations, stable baseline
  09:22–09:40  Minor conveyor jam (brief speed dip, no defects)
  09:40–12:00  Normal operations resume
  12:00–13:00  Coolant flow begins slow decline (V-17 calibration drift starting)
               Temperature starts creeping up: 172→178°C
  13:00–13:30  Temperature visibly trending up: 178→182°C, first sporadic defects
  13:30–14:00  Operator reduces speed by 10% (45→40.5). Temp hits 184°C.
               Defect rate climbs to ~5%. Day shift ends, handover to evening shift.
  14:00–14:32  Temperature crosses Warning Threshold (185°C) at ~14:32
  14:32–15:00  Temperature peaks at ~191°C. Defect rate peaks at ~8%.
               Line speed further reduced to 38 m/min.
  15:00–15:30  Maintenance called. V-17 identified and recalibrated.
  15:30–16:00  Coolant flow restored. Temperature dropping. Speed held at 38.
  16:00–17:00  Temperature back to normal. Speed gradually recovers to 45.
  17:00–22:00  Normal operations for remainder of evening shift.

Cross-references to documents:
  - SOP-001: Warning=185°C, Critical=195°C thresholds
  - SOP-002: V-17 is one of three cooling valves on Line 3, known for calibration drift
  - QA-Report-2023-09-14: Similar V-17 incident (drift→temp rise→surface cracks)
  - QA-Report-2024-01-10: Different failure mode (hydraulic pressure) — NOT repeated here
  - Maintenance-Log: V-17 deviation was 1.8% at last check on 2024-01-06
  - Defect-Classification-Guide: Surface cracks + pitting correlate with temp >185°C
  - SOP-003: Speed reduction rules (15% first, then further if defects persist)
  - Shift-Handover-2024-01-15: Day shift reports rising temp and 5.1% defect rate in last hour
"""

import csv
import random
import os
from datetime import datetime, timedelta

random.seed(42)  # Reproducible output


def noise(base, amplitude=1.0):
    """Add Gaussian noise to a base value."""
    return base + random.gauss(0, amplitude)


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def generate_sensor_data():
    rows = []

    start = datetime(2024, 1, 15, 6, 0, 0)
    end = datetime(2024, 1, 15, 22, 0, 0)
    interval = timedelta(minutes=5)

    current = start
    while current <= end:
        t = current
        minutes_since_start = (t - start).total_seconds() / 60  # 0 to 960
        hour = t.hour + t.minute / 60  # decimal hour (6.0 to 22.0)

        # ============================================================
        # FORMING ZONE TEMPERATURE (°C)
        # Normal: 170–175. Drift starts ~12:00, peaks ~15:00, recovers by ~16:30
        # ============================================================
        if hour < 12.0:
            # Normal operations
            forming_temp = noise(172, 1.5)
        elif hour < 13.0:
            # Slow creep: 172 → 178 over 1 hour
            progress = (hour - 12.0) / 1.0
            forming_temp = noise(172 + progress * 6, 1.0)
        elif hour < 13.5:
            # Faster rise: 178 → 182
            progress = (hour - 13.0) / 0.5
            forming_temp = noise(178 + progress * 4, 0.8)
        elif hour < 14.0:
            # Continuing rise: 182 → 184 (day shift handover zone)
            progress = (hour - 13.5) / 0.5
            forming_temp = noise(182 + progress * 2, 0.8)
        elif hour < 14.5:
            # Crosses warning threshold: 184 → 187
            progress = (hour - 14.0) / 0.5
            forming_temp = noise(184 + progress * 3, 0.7)
        elif hour < 15.0:
            # Peak zone: 187 → 191
            progress = (hour - 14.5) / 0.5
            forming_temp = noise(187 + progress * 4, 0.6)
        elif hour < 15.5:
            # Maintenance intervention at 15:00-15:30, temp plateaus then starts dropping
            progress = (hour - 15.0) / 0.5
            forming_temp = noise(191 - progress * 5, 1.0)
        elif hour < 16.0:
            # Dropping: 186 → 180
            progress = (hour - 15.5) / 0.5
            forming_temp = noise(186 - progress * 6, 1.0)
        elif hour < 16.5:
            # Final recovery: 180 → 175
            progress = (hour - 16.0) / 0.5
            forming_temp = noise(180 - progress * 5, 1.2)
        else:
            # Normal for rest of evening shift
            forming_temp = noise(173, 1.5)

        forming_temp = round(clamp(forming_temp, 165, 198), 1)

        # ============================================================
        # COOLING ZONE TEMPERATURE (°C)
        # Normal: 45–55. Slight rise when forming zone is hot.
        # ============================================================
        if forming_temp > 185:
            cooling_temp = noise(58 + (forming_temp - 185) * 0.5, 2.0)
        elif forming_temp > 178:
            cooling_temp = noise(52 + (forming_temp - 178) * 0.8, 2.0)
        else:
            cooling_temp = noise(50, 2.5)

        cooling_temp = round(clamp(cooling_temp, 40, 78), 1)

        # ============================================================
        # LINE SPEED (m/min)
        # Nominal: 45. Reduced at 13:30 to 40.5, further to 38 at ~14:30
        # Recovers after 16:00
        # ============================================================
        if 9.35 <= hour <= 9.67:
            # Conveyor jam 09:22-09:40 — brief speed dip
            line_speed = noise(30, 2.0)
        elif hour < 13.5:
            line_speed = noise(45, 0.3)
        elif hour < 14.5:
            # First reduction: 10% → 40.5
            line_speed = noise(40.5, 0.3)
        elif hour < 15.5:
            # Second reduction: down to 38
            line_speed = noise(38, 0.3)
        elif hour < 16.0:
            # Holding reduced speed during recovery
            line_speed = noise(38, 0.3)
        elif hour < 16.5:
            # Ramping back up: 38 → 41
            progress = (hour - 16.0) / 0.5
            line_speed = noise(38 + progress * 3, 0.3)
        elif hour < 17.0:
            # Ramping: 41 → 43
            progress = (hour - 16.5) / 0.5
            line_speed = noise(41 + progress * 2, 0.3)
        elif hour < 17.5:
            # Final ramp: 43 → 45
            progress = (hour - 17.0) / 0.5
            line_speed = noise(43 + progress * 2, 0.3)
        else:
            line_speed = noise(45, 0.3)

        line_speed = round(clamp(line_speed, 20, 50), 1)

        # ============================================================
        # HYDRAULIC PRESSURE (bar)
        # Stable at ~3.2 throughout — this is NOT the failure mode today
        # (The pressure failure was the Jan 10 incident, already resolved)
        # ============================================================
        hydraulic_pressure = round(noise(3.2, 0.05), 2)
        hydraulic_pressure = clamp(hydraulic_pressure, 2.9, 3.5)

        # ============================================================
        # COOLANT FLOW % (V-17 calibration drift)
        # Normal: ~97-100%. Drifts down from ~12:00, bottoms at ~71% around 15:00
        # Restored after maintenance at 15:30
        # ============================================================
        if hour < 12.0:
            coolant_flow = noise(98, 1.0)
        elif hour < 13.0:
            # Slow drift: 98 → 90
            progress = (hour - 12.0) / 1.0
            coolant_flow = noise(98 - progress * 8, 0.8)
        elif hour < 14.0:
            # Accelerating drift: 90 → 80
            progress = (hour - 13.0) / 1.0
            coolant_flow = noise(90 - progress * 10, 0.8)
        elif hour < 15.0:
            # Bottoming out: 80 → 71
            progress = (hour - 14.0) / 1.0
            coolant_flow = noise(80 - progress * 9, 0.6)
        elif hour < 15.5:
            # Maintenance recalibrating V-17: rapid recovery 71 → 95
            progress = (hour - 15.0) / 0.5
            coolant_flow = noise(71 + progress * 24, 1.5)
        else:
            # Restored
            coolant_flow = noise(97, 1.0)

        coolant_flow = round(clamp(coolant_flow, 65, 100), 1)

        # ============================================================
        # DEFECT DETECTION & CLASSIFICATION
        # Baseline defect probability: ~2%. Rises with temperature.
        # Defect types weighted per Defect Classification Guide:
        #   surface_crack (60%), pitting (30%), edge_deformation (10%)
        # ============================================================
        if forming_temp < 180:
            defect_prob = 0.02 + random.uniform(0, 0.01)
        elif forming_temp < 185:
            defect_prob = 0.05 + (forming_temp - 180) * 0.02 + random.uniform(0, 0.02)
        elif forming_temp < 190:
            defect_prob = 0.15 + (forming_temp - 185) * 0.04 + random.uniform(0, 0.03)
        else:
            defect_prob = 0.30 + (forming_temp - 190) * 0.05 + random.uniform(0, 0.03)

        # Conveyor jam can cause edge deformation
        if 9.35 <= hour <= 9.67:
            defect_prob = 0.04

        defect_detected = random.random() < defect_prob

        if defect_detected:
            roll = random.random()
            if 9.35 <= hour <= 9.67:
                defect_type = "edge_deformation"
            elif roll < 0.60:
                defect_type = "surface_crack"
            elif roll < 0.90:
                defect_type = "pitting"
            else:
                defect_type = "edge_deformation"
            confidence = round(random.uniform(0.72, 0.98), 2)
        else:
            defect_type = ""
            confidence = None

        # ============================================================
        # DEFECT RATE % (rolling 15-minute window approximation)
        # Modeled from defect_prob scaled to rate
        # ============================================================
        defect_rate = round(clamp(defect_prob * 100 * random.uniform(0.7, 1.3), 0.5, 12.0), 1)

        # ============================================================
        # BUILD ROW
        # ============================================================
        rows.append({
            "timestamp": t.strftime("%Y-%m-%d %H:%M:%S"),
            "line_id": "LINE-3",
            "forming_zone_temp_c": forming_temp,
            "cooling_zone_temp_c": cooling_temp,
            "line_speed_mpm": line_speed,
            "hydraulic_pressure_bar": hydraulic_pressure,
            "coolant_flow_pct": coolant_flow,
            "defect_detected": defect_detected,
            "defect_type": defect_type,
            "confidence": confidence if confidence else "",
            "defect_rate_pct": defect_rate,
        })

        current += interval

    return rows


def write_csv(rows, output_path="data/sensor_data.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = [
        "timestamp", "line_id",
        "forming_zone_temp_c", "cooling_zone_temp_c",
        "line_speed_mpm", "hydraulic_pressure_bar", "coolant_flow_pct",
        "defect_detected", "defect_type", "confidence", "defect_rate_pct",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def print_summary(rows):
    total = len(rows)
    defects = sum(1 for r in rows if r["defect_detected"])
    temps = [r["forming_zone_temp_c"] for r in rows]
    flows = [r["coolant_flow_pct"] for r in rows]

    print("=" * 60)
    print("SENSOR DATA GENERATION SUMMARY")
    print("=" * 60)
    print(f"Time range:    {rows[0]['timestamp']}  →  {rows[-1]['timestamp']}")
    print(f"Total rows:    {total}")
    print(f"Interval:      5 minutes")
    print(f"Defect events: {defects} ({defects/total*100:.1f}% of readings)")
    print(f"Temperature:   min={min(temps):.1f}°C  max={max(temps):.1f}°C")
    print(f"Coolant flow:  min={min(flows):.1f}%   max={max(flows):.1f}%")
    print()

    # Show key moments
    print("KEY TIMELINE EVENTS IN DATA:")
    print("-" * 60)
    for r in rows:
        t = r["timestamp"]
        temp = r["forming_zone_temp_c"]
        flow = r["coolant_flow_pct"]
        rate = r["defect_rate_pct"]
        speed = r["line_speed_mpm"]

        # Flag important moments
        flags = []
        if "09:20" <= t[11:16] <= "09:40":
            flags.append("CONVEYOR JAM")
        if abs(temp - 185) < 1.5 and 14.0 <= float(t[11:13]) + float(t[14:16])/60 <= 15.0:
            flags.append("⚠️ WARNING THRESHOLD")
        if temp >= 190:
            flags.append("🔴 PEAK TEMPERATURE")
        if flow < 75:
            flags.append("🔧 MIN COOLANT FLOW")
        if rate > 5.0 and r["defect_detected"]:
            flags.append("📈 HIGH DEFECT RATE")

        if flags:
            print(f"  {t}  |  temp={temp:>6.1f}°C  flow={flow:>5.1f}%  "
                  f"speed={speed:>5.1f}  rate={rate:>4.1f}%  |  {', '.join(flags)}")

    print()
    print("=" * 60)


if __name__ == "__main__":
    print("Generating sensor data for Production Line 3 (2024-01-15)...")
    rows = generate_sensor_data()
    output = write_csv(rows)
    print(f"✅ Saved {len(rows)} rows to {output}\n")
    print_summary(rows)

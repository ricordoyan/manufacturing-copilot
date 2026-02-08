# SOP-001: Temperature Control Procedure

**Department:** Production Engineering
**Effective Date:** 2023-01-15
**Revision:** 3.2
**Applies To:** Production Lines 1–5

## 1. Purpose

This procedure defines the acceptable operating temperature ranges for all extrusion and forming lines, and the required operator response when thresholds are breached.

## 2. Temperature Thresholds

| Zone | Normal Range (°C) | Warning Threshold (°C) | Critical Threshold (°C) |
|------|-------------------|------------------------|-------------------------|
| Pre-heat Zone | 140–155 | 160 | 170 |
| Forming Zone | 165–180 | 185 | 195 |
| Cooling Zone | 40–60 | 70 | 85 |

## 3. Monitoring Requirements

- Operators must verify temperature readings at the start of each shift and every 30 minutes thereafter.
- The SCADA system generates automatic alerts when any zone exceeds the Warning Threshold for more than 5 consecutive minutes.
- All temperature exceedances must be logged in the Shift Event Log.

## 4. Response Procedure — Warning Threshold Exceeded

1. Acknowledge the alert on the SCADA console.
2. Verify the reading with a secondary thermocouple (located at panel J-4 on each line).
3. If confirmed, reduce line speed by 15% and notify the shift supervisor.
4. Check cooling system flow rates (see SOP-002).
5. If the temperature does not return to Normal Range within 15 minutes, escalate to Critical response.

## 5. Response Procedure — Critical Threshold Exceeded

1. Immediately reduce line speed to 50% of nominal.
2. Notify shift supervisor and on-call maintenance engineer.
3. Inspect cooling valves (V-15 through V-20 depending on line) for flow obstruction or calibration drift.
4. If temperature does not stabilize within 10 minutes, initiate controlled line shutdown per SOP-007.
5. Do not restart the line until maintenance has cleared the root cause and temperature has returned to Normal Range for at least 20 minutes.

## 6. Documentation

All temperature exceedances, actions taken, and resolution times must be recorded in the Production Event Management System (PEMS) within 2 hours of occurrence.

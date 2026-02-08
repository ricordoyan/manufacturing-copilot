# SOP-003: Line Speed Adjustment Procedure

**Department:** Production Engineering
**Effective Date:** 2023-02-01
**Revision:** 2.0
**Applies To:** Production Lines 1–5

## 1. Purpose

This procedure defines the rules and authorization levels for adjusting production line speed, and the required coordination with quality and maintenance teams.

## 2. Nominal Line Speeds

| Production Line | Nominal Speed (m/min) | Minimum Safe Speed (m/min) | Maximum Speed (m/min) |
|-----------------|-----------------------|----------------------------|-----------------------|
| Line 1 | 40 | 20 | 50 |
| Line 2 | 40 | 20 | 50 |
| Line 3 | 45 | 22 | 55 |
| Line 4 | 38 | 18 | 48 |
| Line 5 | 42 | 20 | 52 |

## 3. Authorized Speed Adjustments

### Operator-Authorized (no supervisor approval needed)
- Speed reduction of up to 15% from nominal for temperature management (per SOP-001).
- Speed reduction of up to 10% for product changeover ramp-down.

### Supervisor-Authorized
- Speed reduction greater than 15% from nominal.
- Any speed increase above nominal.
- Speed changes during quality hold periods.

## 4. Speed Reduction During Defect Events

When a defect event is detected (defect rate exceeds 3% over a 15-minute rolling window):

1. Immediately reduce line speed by 15%.
2. Notify QA inspector for inline assessment.
3. If defect rate does not decrease below 3% within 15 minutes of speed reduction, reduce speed by an additional 15% (to 70% of nominal) and notify the shift supervisor.
4. If defect rate remains above 3% after 30 minutes at 70% speed, initiate controlled line stop and escalate to maintenance.

## 5. Speed Recovery After Resolution

After the root cause has been resolved and cleared by maintenance:

1. Increase speed in increments of 5 m/min every 10 minutes.
2. QA inspector must verify defect rate remains below 2% at each speed increment.
3. Full nominal speed can be resumed only after 20 minutes of stable operation at 90% speed with defect rate below 2%.

## 6. Documentation

All speed adjustments must be logged in PEMS with timestamp, reason, authorizing person, and the production data (defect rate, temperature, pressure) at the time of adjustment.

# SOP-002: Cooling System Maintenance

**Department:** Maintenance Engineering
**Effective Date:** 2023-03-01
**Revision:** 2.1
**Applies To:** Cooling Subsystems, Production Lines 1–5

## 1. Purpose

This procedure covers the preventive maintenance schedule, calibration requirements, and troubleshooting steps for the cooling valve assemblies on all production lines.

## 2. System Overview

Each production line has a set of cooling valves that regulate coolant flow to the Forming and Cooling Zones. The valve assignments are:

| Production Line | Cooling Valves |
|-----------------|----------------|
| Line 1 | V-11, V-12 |
| Line 2 | V-13, V-14 |
| Line 3 | V-15, V-16, V-17 |
| Line 4 | V-18 |
| Line 5 | V-19, V-20 |

Line 3 has three valves due to its higher throughput capacity and dual cooling loop design.

## 3. Preventive Maintenance Schedule

| Task | Frequency | Responsible |
|------|-----------|-------------|
| Visual inspection of valve body and seals | Weekly | Line Operator |
| Flow rate calibration check | Every 30 days | Maintenance Technician |
| Full valve disassembly and cleaning | Every 90 days | Maintenance Engineer |
| Coolant fluid replacement | Every 180 days | Maintenance Engineer |
| Actuator response time test | Every 90 days | Instrumentation Tech |

## 4. Calibration Procedure

1. Isolate the valve from the production loop using the upstream manual shutoff.
2. Connect the portable flow meter (Model FM-220) to the test port downstream of the valve.
3. Command the valve to 25%, 50%, 75%, and 100% open positions via the SCADA interface.
4. Record the measured flow rate at each position and compare against the baseline values in Appendix A.
5. If any position deviates by more than ±8% from baseline, perform actuator recalibration using the vendor tool (ValveCal Pro v4.1).
6. If recalibration fails, replace the valve actuator assembly and submit a maintenance order for the removed unit.

## 5. Known Failure Modes

- **Calibration Drift:** Gradual loss of positioning accuracy, typically 1–2% per month. Most common on V-15 and V-17 due to higher cycle counts on Line 3. Causes slow temperature creep that may not trigger immediate alarms but leads to elevated defect rates over a shift.
- **Seal Degradation:** Coolant leaks at the valve body. Identified by visible moisture or coolant puddles. Requires immediate valve isolation and seal replacement.
- **Actuator Stall:** Valve fails to respond to position commands. Usually caused by electrical connector corrosion. Clean connectors and retest before replacing the actuator.

## 6. Troubleshooting — Elevated Forming Zone Temperature

If the Forming Zone temperature exceeds the Warning Threshold (185°C):

1. Check the SCADA flow rate reading for the relevant cooling valves.
2. If flow rate is below 80% of nominal, suspect calibration drift or partial obstruction.
3. Manually command the valve to 100% open. If temperature drops, the valve was not responding correctly to automatic control — schedule immediate recalibration.
4. If manually opening the valve has no effect, check for upstream coolant supply issues (pump pressure, reservoir level).

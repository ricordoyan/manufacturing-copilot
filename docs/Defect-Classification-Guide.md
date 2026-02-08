# Defect Classification Guide — Formed Metal Products

**Department:** Quality Assurance
**Revision:** 4.0
**Effective Date:** 2023-04-01

## 1. Purpose

This guide provides standardized definitions, visual references, and probable root causes for common defect types observed on formed metal products across all production lines.

## 2. Defect Types

### 2.1 Surface Cracks

- **Description:** Visible fractures on the product surface, typically hairline width (<0.1mm). May be oriented along or perpendicular to the forming direction.
- **Severity:** Major (structural integrity risk)
- **Common Root Causes:**
  - Forming Zone temperature exceeding 185°C (most common)
  - Raw material brittleness (batch variation)
  - Excessive line speed during temperature exceedance
- **Correlation:** Strong correlation with Forming Zone temperature. Surface cracks almost always appear when temperature exceeds 185°C for more than 10 minutes. Historical data shows 94% of surface crack events are temperature-related.

### 2.2 Pitting

- **Description:** Small, shallow depressions on the product surface, typically 0.5–2mm in diameter. Often appear in clusters.
- **Severity:** Moderate to Major (depending on depth and density)
- **Common Root Causes:**
  - Forming Zone temperature exceeding 185°C (often co-occurs with surface cracks)
  - Coolant contamination (particulate in coolant fluid)
  - Die surface wear
- **Correlation:** Pitting often appears alongside surface cracks during temperature exceedances but can also occur independently due to coolant contamination. When pitting occurs without cracks, check coolant quality first.

### 2.3 Edge Deformation

- **Description:** Curling, warping, or uneven thickness at product edges. Deviation from spec typically 0.2–1.0mm.
- **Severity:** Minor to Moderate
- **Common Root Causes:**
  - Hydraulic pressure drop in the forming press (below 2.8 bar)
  - Uneven die wear
  - Conveyor belt misalignment causing asymmetric feed
- **Correlation:** Strong correlation with forming press hydraulic pressure. Edge deformation events are associated with pressure readings below 2.8 bar in 87% of cases.

### 2.4 Discoloration

- **Description:** Uneven color or surface oxidation on the finished product. May appear as dark spots, streaks, or general color shift.
- **Severity:** Minor (cosmetic)
- **Common Root Causes:**
  - Cooling Zone temperature too high (above 70°C), causing uneven cooling
  - Coolant chemical degradation (beyond 180-day replacement interval)
  - Ambient humidity above 75% during cooling phase
- **Correlation:** Most commonly associated with Cooling Zone temperature elevation. Check coolant age if Cooling Zone temperature is normal.

### 2.5 Dimensional Deviation

- **Description:** Product dimensions outside of specified tolerances (typically ±0.1mm on thickness, ±0.5mm on length/width).
- **Severity:** Major
- **Common Root Causes:**
  - Die wear beyond service limits
  - Forming press pressure inconsistency
  - Raw material thickness variation outside incoming inspection limits
- **Correlation:** If dimensional deviation is consistent across all products, suspect die wear. If intermittent, suspect pressure fluctuation or material variation.

## 3. Defect-to-Signal Quick Reference

| Defect Type | Primary Signal to Check | Secondary Signal |
|-------------|------------------------|------------------|
| Surface Cracks | Forming Zone Temperature | Line Speed |
| Pitting | Forming Zone Temperature | Coolant Quality |
| Edge Deformation | Hydraulic Pressure | Conveyor Alignment |
| Discoloration | Cooling Zone Temperature | Coolant Age |
| Dimensional Deviation | Die Wear Log | Press Pressure |

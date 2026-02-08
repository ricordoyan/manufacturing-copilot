# NEU Surface Defect Database (NEU-DET)

## Overview

The NEU Surface Defect Database is a benchmark dataset for surface defect detection on hot-rolled steel strips, created by Northeastern University (NEU), China.

The dataset contains **1,800 grayscale images** (200×200 pixels) across **6 defect categories**, with 300 images per category.

## Defect Categories

| Category | Description | Visual Characteristics |
|---|---|---|
| **Crazing** | Fine network of surface cracks | Web-like pattern of thin lines across the surface |
| **Inclusion** | Foreign material embedded in steel | Localized dark or bright spots, irregular shapes |
| **Patches** | Surface patches or stains | Irregular areas with different texture/brightness |
| **Pitted Surface** | Pitting corrosion marks | Multiple small, scattered dark indentations |
| **Rolled-in Scale** | Oxide scale pressed into surface | Elongated dark marks aligned with rolling direction |
| **Scratches** | Linear surface scratches | Long, thin lines or grooves on the surface |

## Causes and Corrective Actions

### Crazing
- **Root Cause:** Thermal fatigue, rapid cooling, excessive surface stress during rolling.
- **Corrective Action:** Reduce cooling rate, inspect roll surfaces for thermal damage, verify forming zone temperature stays below 185°C.

### Inclusion
- **Root Cause:** Contaminated raw material, slag carryover from steelmaking, refractory erosion.
- **Corrective Action:** Improve upstream cleanliness, inspect ladle and tundish linings, increase filter effectiveness.

### Patches
- **Root Cause:** Uneven descaling, roll surface degradation, lubrication failure.
- **Corrective Action:** Check descaling nozzle pressure, inspect work rolls, verify lubricant flow rate.

### Pitted Surface
- **Root Cause:** Acid pickling over-exposure, corrosive environmental conditions, condensation on surface.
- **Corrective Action:** Optimize pickling line speed and acid concentration, improve storage environment humidity control.

### Rolled-in Scale
- **Root Cause:** Inadequate descaling before rolling, low descaling water pressure, oxide buildup.
- **Corrective Action:** Verify descaling header pressure ≥ 150 bar, inspect spray nozzles, increase descaling passes.

### Scratches
- **Root Cause:** Mechanical contact with guides, rolls, or conveyor components; foreign objects on line.
- **Corrective Action:** Inspect guide rails and conveyor rollers, check clearances, remove debris from production line.

## Dataset Usage in This Application

The NEU-DET dataset is integrated into the Manufacturing Defect Detection Copilot in the following ways:

1. **Video Feed Tab:** NEU-DET images appear in the simulated camera feed. When a NEU-DET image is displayed, the system uses the ground-truth defect type from the filename and overlays bounding boxes from the XML annotations.

2. **NEU-DET Dataset Browser Tab:** A dedicated tab allows browsing images by defect category, viewing annotations, and inspecting bounding box details.

3. **RAG Knowledge Base:** This document is indexed so the copilot can answer questions about steel surface defect types, their causes, and corrective actions.

## Annotations

Each image has a corresponding Pascal-VOC-format XML annotation file containing:
- Image filename and dimensions
- One or more bounding boxes with defect class labels

## Reference

K. Song and Y. Yan, "A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects," *Applied Surface Science*, vol. 285, pp. 858-864, 2013.

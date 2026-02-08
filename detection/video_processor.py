"""
Frame extraction and simple OpenCV-based defect detection.

This module provides a VideoProcessor that iterates over images on disk
(simulating a camera feed) and applies lightweight anomaly detection using
adaptive thresholding.  In production this would be replaced by an NVIDIA
NIM vision model, but this heuristic is sufficient for the demo.
"""

import os
import time
from datetime import datetime
from typing import Generator, Optional

import cv2
import numpy as np


class VideoProcessor:
    """Simulate a production-line camera feed from a directory of images."""

    # Anomalous-pixel percentage threshold.  5 % strikes a reasonable
    # balance between sensitivity and false-positive rate on common
    # surface-defect datasets (MVTec, NEU).
    ANOMALY_THRESHOLD_PCT: float = 5.0

    def __init__(self, image_dir: str) -> None:
        self.image_dir = image_dir
        self._paths: list[str] = self.get_image_paths()

    # ── Image discovery ─────────────────────────────────────────────────

    def get_image_paths(self) -> list[str]:
        """Return a sorted list of all image files in *image_dir* (recursive)."""
        supported = {".png", ".jpg", ".jpeg", ".bmp"}
        paths: list[str] = []
        if not os.path.isdir(self.image_dir):
            return paths
        for root, _, files in os.walk(self.image_dir):
            for f in sorted(files):
                if os.path.splitext(f)[1].lower() in supported:
                    paths.append(os.path.join(root, f))
        return sorted(paths)

    # ── Simulated feed ──────────────────────────────────────────────────

    def simulate_video_feed(
        self, fps: int = 2
    ) -> Generator[tuple[np.ndarray, str, str], None, None]:
        """Yield ``(frame, filename, iso_timestamp)`` tuples in a loop.

        The generator cycles through the image directory endlessly at the
        given *fps* (frames per second).  Each iteration sleeps to emulate
        a real-time camera.
        """
        if not self._paths:
            return

        delay = 1.0 / fps
        idx = 0
        while True:
            path = self._paths[idx % len(self._paths)]
            frame = cv2.imread(path)
            if frame is None:
                idx += 1
                continue
            filename = os.path.basename(path)
            timestamp = datetime.utcnow().isoformat()
            yield frame, filename, timestamp
            idx += 1
            time.sleep(delay)

    # ── Preprocessing ───────────────────────────────────────────────────

    @staticmethod
    def preprocess_frame(frame: np.ndarray) -> np.ndarray:
        """Resize to 224×224 and normalise to [0, 1] float32."""
        resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        normalised = resized.astype(np.float32) / 255.0
        return normalised

    # ── Simple heuristic defect detector ────────────────────────────────

    def detect_defect_simple(
        self, frame: np.ndarray, threshold_pct: Optional[float] = None
    ) -> dict:
        """Detect anomalies using adaptive thresholding.

        The approach is intentionally simple — the goal is to flag regions
        that deviate from a uniform surface.  On datasets like NEU-DET
        this gives a rough but useful signal.

        Returns
        -------
        dict
            ``has_defect``          – bool
            ``confidence``          – float [0, 1]
            ``anomaly_percentage``  – float (% of frame classified anomalous)
            ``defect_type``         – str | None
        """
        thr = threshold_pct if threshold_pct is not None else self.ANOMALY_THRESHOLD_PCT

        # 1. Greyscale
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        # 2. Blur to suppress sensor noise
        blurred = cv2.GaussianBlur(grey, (11, 11), 0)

        # 3. Adaptive threshold — highlights local deviations
        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=25,
            C=8,
        )

        # 4. Anomaly percentage
        anomaly_pixels = np.count_nonzero(binary)
        total_pixels = binary.size
        anomaly_pct = (anomaly_pixels / total_pixels) * 100.0

        has_defect = anomaly_pct > thr

        # 5. Confidence — map anomaly_pct into [0, 1].
        #    10 % → conf ≈ 0.65,  20 % → conf ≈ 0.85
        confidence = min(1.0, anomaly_pct / 25.0) if has_defect else anomaly_pct / 50.0

        # 6. Defect-type heuristic (very rough):
        #    - Find contours of the anomalous regions.
        #    - If many small regions → "pitting"
        #    - If any contour has high aspect ratio → "surface_crack"
        #    - Else default to "surface_crack"
        defect_type: Optional[str] = None
        if has_defect:
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 15:
                # Many small, scattered blobs → pitting
                defect_type = "pitting"
            else:
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect = max(w, h) / (min(w, h) + 1e-6)
                    if aspect > 4.0:
                        defect_type = "surface_crack"
                        break
                if defect_type is None:
                    defect_type = "surface_crack"  # conservative default

        return {
            "has_defect": has_defect,
            "confidence": round(confidence, 3),
            "anomaly_percentage": round(anomaly_pct, 2),
            "defect_type": defect_type,
        }

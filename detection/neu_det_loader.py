"""
NEU Surface Defect Database loader and annotation parser.

Parses Pascal-VOC-style XML annotations from the NEU-DET dataset and
provides utilities for browsing images by defect category, retrieving
bounding boxes, and computing dataset statistics.

Dataset structure expected:
    NEU-DET/
        train/
            images/<category>/<category>_<id>.jpg
            annotations/<category>_<id>.xml
        validation/
            images/<category>/<category>_<id>.jpg
            annotations/<category>_<id>.xml

Categories: crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches
"""

import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from config import (
    NEU_DET_TRAIN_ANNOTATIONS,
    NEU_DET_TRAIN_IMAGES,
    NEU_DET_VAL_ANNOTATIONS,
    NEU_DET_VAL_IMAGES,
    NEU_DEFECT_TYPES,
)


@dataclass
class BoundingBox:
    """A single bounding box annotation."""
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    label: str


@dataclass
class NEUAnnotation:
    """Parsed annotation for one NEU-DET image."""
    filename: str
    width: int
    height: int
    defect_type: str
    boxes: list[BoundingBox] = field(default_factory=list)

    @property
    def image_stem(self) -> str:
        return os.path.splitext(self.filename)[0]


def parse_annotation(xml_path: str) -> Optional[NEUAnnotation]:
    """Parse a Pascal-VOC XML annotation file.

    Returns None if the file cannot be parsed.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename = root.findtext("filename", "")
        size_el = root.find("size")
        width = int(size_el.findtext("width", "200"))
        height = int(size_el.findtext("height", "200"))

        # Derive defect type from filename  (e.g. "crazing_1.jpg" → "crazing")
        defect_type = _defect_type_from_filename(filename)

        boxes: list[BoundingBox] = []
        for obj in root.findall("object"):
            label = obj.findtext("name", defect_type)
            bbox = obj.find("bndbox")
            if bbox is not None:
                boxes.append(BoundingBox(
                    xmin=int(bbox.findtext("xmin", "0")),
                    ymin=int(bbox.findtext("ymin", "0")),
                    xmax=int(bbox.findtext("xmax", "0")),
                    ymax=int(bbox.findtext("ymax", "0")),
                    label=label,
                ))

        return NEUAnnotation(
            filename=filename,
            width=width,
            height=height,
            defect_type=defect_type,
            boxes=boxes,
        )
    except Exception:
        return None


def _defect_type_from_filename(filename: str) -> str:
    """Extract defect category from a NEU-DET filename.

    Examples:
        ``crazing_1.jpg``        → ``crazing``
        ``rolled-in_scale_3.jpg``→ ``rolled-in_scale``
        ``pitted_surface_42.jpg``→ ``pitted_surface``
    """
    stem = os.path.splitext(filename)[0]  # e.g. "crazing_1"
    # Match known categories (longest first to handle multi-word names)
    for cat in sorted(NEU_DEFECT_TYPES, key=len, reverse=True):
        if stem.startswith(cat):
            return cat
    # Fallback: everything before the last underscore+digits
    match = re.match(r"^(.+?)_\d+$", stem)
    return match.group(1) if match else stem


def defect_type_from_path(image_path: str) -> str:
    """Infer NEU-DET defect type from a full image path."""
    return _defect_type_from_filename(os.path.basename(image_path))


class NEUDatasetLoader:
    """Browse and query the NEU-DET dataset."""

    def __init__(
        self,
        images_dir: str = NEU_DET_TRAIN_IMAGES,
        annotations_dir: str = NEU_DET_TRAIN_ANNOTATIONS,
    ) -> None:
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self._index: dict[str, list[str]] = {}  # category → [image_paths]
        self._annotations: dict[str, NEUAnnotation] = {}  # stem → annotation
        self._build_index()

    # ── Indexing ────────────────────────────────────────────────────────

    def _build_index(self) -> None:
        """Walk the images directory and group files by defect category."""
        supported = {".png", ".jpg", ".jpeg", ".bmp"}
        if not os.path.isdir(self.images_dir):
            return
        for root, _, files in os.walk(self.images_dir):
            for f in sorted(files):
                if os.path.splitext(f)[1].lower() not in supported:
                    continue
                path = os.path.join(root, f)
                cat = _defect_type_from_filename(f)
                self._index.setdefault(cat, []).append(path)

        # Load annotations
        if os.path.isdir(self.annotations_dir):
            for f in os.listdir(self.annotations_dir):
                if f.endswith(".xml"):
                    ann = parse_annotation(os.path.join(self.annotations_dir, f))
                    if ann:
                        self._annotations[ann.image_stem] = ann

    # ── Queries ─────────────────────────────────────────────────────────

    @property
    def categories(self) -> list[str]:
        """Return available defect categories (sorted)."""
        return sorted(self._index.keys())

    def images_for_category(self, category: str) -> list[str]:
        """Return all image paths for a given defect category."""
        return self._index.get(category, [])

    def total_images(self) -> int:
        return sum(len(v) for v in self._index.values())

    def category_counts(self) -> dict[str, int]:
        return {cat: len(paths) for cat, paths in sorted(self._index.items())}

    def get_annotation(self, image_path: str) -> Optional[NEUAnnotation]:
        """Retrieve the annotation for a given image path (if available)."""
        stem = os.path.splitext(os.path.basename(image_path))[0]
        return self._annotations.get(stem)

    # ── Visualization helpers ───────────────────────────────────────────

    def draw_annotations(
        self, frame: np.ndarray, annotation: NEUAnnotation
    ) -> np.ndarray:
        """Draw bounding boxes and labels on a frame copy."""
        display = frame.copy()
        color_map = {
            "crazing": (0, 165, 255),        # orange
            "inclusion": (0, 0, 255),         # red
            "patches": (255, 0, 0),           # blue
            "pitted_surface": (0, 255, 255),  # yellow
            "rolled-in_scale": (255, 0, 255), # magenta
            "scratches": (0, 255, 0),         # green
        }
        for box in annotation.boxes:
            color = color_map.get(box.label, (0, 0, 255))
            cv2.rectangle(display, (box.xmin, box.ymin), (box.xmax, box.ymax), color, 2)
            label_text = box.label.replace("_", " ").title()
            # Background for text
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                display,
                (box.xmin, box.ymin - th - 6),
                (box.xmin + tw + 4, box.ymin),
                color,
                -1,
            )
            cv2.putText(
                display,
                label_text,
                (box.xmin + 2, box.ymin - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
        return display

    def get_dataset_stats(self) -> dict:
        """Return a summary dict for display in the UI."""
        counts = self.category_counts()
        return {
            "total_images": self.total_images(),
            "categories": len(counts),
            "category_counts": counts,
            "total_annotations": len(self._annotations),
        }

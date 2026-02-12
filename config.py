"""
Configuration module for Manufacturing Defect Detection Copilot.

Centralizes all environment variables and application constants so that
every other module imports from a single source of truth.
"""

import os
from dotenv import load_dotenv

# Load .env file from project root (one level up from this file's directory
# or the current working directory – whichever contains .env).
load_dotenv()

# ── NVIDIA NIM API ──────────────────────────────────────────────────────────
NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL: str = "https://integrate.api.nvidia.com/v1"
NVIDIA_LLM_MODEL: str = "meta/llama-3.1-70b-instruct"
NVIDIA_EMBED_MODEL: str = "nvidia/nv-embedqa-e5-v5"

# ── Paths ───────────────────────────────────────────────────────────────────
SQLITE_DB_PATH: str = "data/defects.db"
FAISS_INDEX_PATH: str = "data/faiss_index"
DOCS_DIR: str = "docs/"
SENSOR_DATA_PATH: str = "data/sensor_data.csv"
SAMPLE_IMAGES_DIR: str = "data/sample_images/"

# ── NEU-DET dataset paths ────────────────────────────────────────────────
NEU_DET_TRAIN_IMAGES: str = "data/sample_images/NEU-DET/train/images/"
NEU_DET_TRAIN_ANNOTATIONS: str = "data/sample_images/NEU-DET/train/annotations/"
NEU_DET_VAL_IMAGES: str = "data/sample_images/NEU-DET/validation/images/"
NEU_DET_VAL_ANNOTATIONS: str = "data/sample_images/NEU-DET/validation/annotations/"

NEU_DEFECT_TYPES: list[str] = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

# ── Sensor thresholds ──────────────────────────────────────────────────────
# Forming-zone temperature thresholds (°C).
# WARNING  → operators should be alerted.
# CRITICAL → automated slow-down / line stop may be warranted.
TEMP_WARNING: float = 185.0
TEMP_CRITICAL: float = 195.0

# Hydraulic pressure thresholds (bar).
# Below WARNING → investigate; below CRITICAL → stop line.
PRESSURE_WARNING: float = 2.8
PRESSURE_CRITICAL: float = 2.0

# ── Defect taxonomy ────────────────────────────────────────────────────────
DEFECT_TYPES: list[str] = [
    "surface_crack",
    "pitting",
    "edge_deformation",
    "discoloration",
    "dimensional_deviation",
]

# ── Nominal operating parameters ───────────────────────────────────────────
NOMINAL_LINE_SPEED: float = 45.0   # metres per minute
NOMINAL_PRESSURE: float = 3.2      # bar

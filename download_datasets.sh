#!/usr/bin/env bash
# download_datasets.sh — Download external datasets for the Manufacturing Copilot.
#
# Usage:
#   chmod +x download_datasets.sh
#   ./download_datasets.sh
#
# Datasets:
#   1. NEU Surface Defect Database (NEU-DET)  ~30 MB
#   2. Severstal Steel Defect Detection        ~1.6 GB  (requires Kaggle CLI)
#   3. MVTec Anomaly Detection                 ~5.1 GB  (manual download required)

set -euo pipefail

DATA_DIR="data/sample_images"
mkdir -p "$DATA_DIR"

# ── Helper ──────────────────────────────────────────────────────────────────
info()  { echo "ℹ️  $*"; }
ok()    { echo "✅  $*"; }
warn()  { echo "⚠️  $*"; }
error() { echo "❌  $*" >&2; }

# ── 1. NEU-DET Dataset ─────────────────────────────────────────────────────
download_neu_det() {
    local dest="$DATA_DIR/NEU-DET"
    if [ -d "$dest/train/images" ]; then
        ok "NEU-DET already exists at $dest — skipping."
        return
    fi

    info "Downloading NEU Surface Defect Database …"

    # The dataset is hosted on Kaggle. Check if kaggle CLI is available.
    if command -v kaggle &>/dev/null; then
        kaggle datasets download -d kaustubhdikshit/neu-surface-defect-database \
            -p /tmp/neu-det --unzip
        mkdir -p "$dest"
        mv /tmp/neu-det/* "$dest/" 2>/dev/null || true
        rm -rf /tmp/neu-det
        ok "NEU-DET downloaded to $dest"
    else
        warn "Kaggle CLI not found. Install it with:"
        echo "    pip install kaggle"
        echo "    # Then configure ~/.kaggle/kaggle.json with your API token"
        echo ""
        echo "  Alternatively, download manually from:"
        echo "    https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database"
        echo "  Extract to: $dest/"
    fi
}

# ── 2. Severstal Steel Defect Detection ────────────────────────────────────
download_severstal() {
    local dest="$DATA_DIR/severstal-steel-defect-detection"
    if [ -f "$dest/train.csv" ] && [ -d "$dest/train_images" ]; then
        ok "Severstal dataset already exists at $dest — skipping."
        return
    fi

    info "Downloading Severstal Steel Defect Detection dataset …"

    if command -v kaggle &>/dev/null; then
        kaggle competitions download -c severstal-steel-defect-detection \
            -p /tmp/severstal
        mkdir -p "$dest"
        unzip -qo /tmp/severstal/severstal-steel-defect-detection.zip -d "$dest"
        rm -rf /tmp/severstal
        ok "Severstal dataset downloaded to $dest"
    else
        warn "Kaggle CLI not found. Install it with:"
        echo "    pip install kaggle"
        echo ""
        echo "  Alternatively, download manually from:"
        echo "    https://www.kaggle.com/c/severstal-steel-defect-detection/data"
        echo "  Extract to: $dest/"
    fi
}

# ── 3. MVTec Anomaly Detection ─────────────────────────────────────────────
download_mvtec() {
    local dest="$DATA_DIR/mvtec_anomaly_detection"
    if [ -d "$dest" ] && [ "$(ls -A "$dest" 2>/dev/null)" ]; then
        ok "MVTec AD already exists at $dest — skipping."
        return
    fi

    warn "MVTec Anomaly Detection dataset (~5.1 GB) requires manual download."
    echo ""
    echo "  1. Visit: https://www.mvtec.com/company/research/datasets/mvtec-ad"
    echo "  2. Register and download mvtec_anomaly_detection.tar.xz"
    echo "  3. Extract to: $dest/"
    echo ""
    echo "  Example:"
    echo "    mkdir -p $dest"
    echo "    tar -xf mvtec_anomaly_detection.tar.xz -C $dest"
}

# ── Run all downloads ──────────────────────────────────────────────────────
echo "============================================"
echo "  Manufacturing Copilot — Dataset Downloader"
echo "============================================"
echo ""

download_neu_det
echo ""
download_severstal
echo ""
download_mvtec
echo ""

echo "============================================"
echo "  Done. Next steps:"
echo "    1. cp .env.example .env"
echo "    2. Edit .env with your NVIDIA API key"
echo "    3. python setup_rag.py"
echo "    4. streamlit run app.py"
echo "============================================"

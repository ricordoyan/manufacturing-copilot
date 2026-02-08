"""
One-time setup script — ingest documents, build FAISS index, populate DB.

Run:
    python setup_rag.py

Prerequisites:
    1.  Place manufacturing docs (.md / .txt / .pdf) in  docs/
    2.  Run  python generate_sensor_data.py  first to create  data/sensor_data.csv
    3.  Set  NVIDIA_API_KEY  in  .env
"""

import os
import sys

# Ensure project root is on the import path so that `config`, `db`, `rag`
# are importable regardless of the caller's working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DOCS_DIR, SENSOR_DATA_PATH
from db.database import init_db, populate_from_sensor_csv
from rag.ingest import build_faiss_index, chunk_documents, load_documents


def main() -> None:
    print("=" * 60)
    print("  Manufacturing Copilot — Setup")
    print("=" * 60)

    # ── 1. Check prerequisites ──────────────────────────────────────────
    if not os.path.isdir(DOCS_DIR) or not os.listdir(DOCS_DIR):
        print(f"\n⚠️  No files found in '{DOCS_DIR}'.")
        print("   Place .md, .txt, or .pdf manufacturing documents there")
        print("   and re-run this script.  Continuing with empty index.\n")

    if not os.path.exists(SENSOR_DATA_PATH):
        print(f"❌  Sensor data not found at '{SENSOR_DATA_PATH}'.")
        print("   Run `python generate_sensor_data.py` first.")
        sys.exit(1)

    # ── 2. Ingest documents ─────────────────────────────────────────────
    print("\n📄 Loading documents …")
    documents = load_documents(DOCS_DIR)
    print(f"   Loaded {len(documents)} document(s).")

    if documents:
        print("📝 Chunking …")
        chunks = chunk_documents(documents, chunk_size=500, chunk_overlap=50)
        print(f"   Created {len(chunks)} chunks.")

        print("🔢 Embedding & building FAISS index …")
        index = build_faiss_index(chunks)
        print(f"   Index dimension: {index.d}")
    else:
        print("   Skipping FAISS index build (no documents).")

    # ── 3. Database setup ───────────────────────────────────────────────
    print("\n🗄️  Initialising SQLite database …")
    init_db()

    print("📊 Populating defect events from sensor CSV …")
    count = populate_from_sensor_csv()
    print(f"   Inserted {count} defect events.")

    # ── Done ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ✅  Setup complete!  Run:  streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

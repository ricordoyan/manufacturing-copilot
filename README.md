# 🏭 Manufacturing Defect Detection Copilot

An AI-powered manufacturing assistant that ingests simulated production-line video, detects defects, correlates them with sensor signals and historical manufacturing documents, and provides natural-language explanations to operators via a Streamlit web UI — powered by **NVIDIA NIM** API endpoints using RAG (Retrieval-Augmented Generation).

![Architecture](architecture-diagram.png)

<details>
<summary>Mermaid Diagram Source</summary>
```
## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| NVIDIA NIM API key | [Get one here](https://build.nvidia.com/) |

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone <repo-url>
cd manufacturing-copilot
```

### 2. Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and replace `your_nvidia_api_key_here` with your actual NVIDIA NIM API key.

### 5. Add manufacturing documents

Place `.md`, `.txt`, or `.pdf` files describing your manufacturing processes, maintenance logs, and standard operating procedures into the `docs/` folder. These documents will be chunked, embedded, and indexed so the copilot can reference them when answering operator questions.

**Example docs you might add:**

- `line3_maintenance_log.md` — past incidents and fixes
- `sop_forming_zone.md` — standard operating procedures
- `coolant_system_specs.txt` — cooling valve specifications

### 6. Add defect images

For the **Video Feed** and **NEU-DET Dataset** tabs, place sample defect images (`.png`, `.jpg`, `.bmp`) into `data/sample_images/`. The NEU Surface Defect Database is natively supported:

#### NEU-DET Dataset (recommended)

Download the [NEU Surface Defect Database](http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/) and extract it so the folder structure looks like:

```
data/sample_images/NEU-DET/
├── train/
│   ├── images/
│   │   ├── crazing/          # 240 images
│   │   ├── inclusion/         # 240 images
│   │   ├── patches/           # 240 images
│   │   ├── pitted_surface/    # 240 images
│   │   ├── rolled-in_scale/   # 240 images
│   │   └── scratches/         # 240 images
│   └── annotations/           # Pascal-VOC XML files
└── validation/
    ├── images/
    └── annotations/
```

The application automatically:
- Parses XML annotations and draws **bounding boxes** on detected defects
- Extracts the **ground-truth defect type** from filenames (e.g., `crazing_1.jpg` → crazing)
- Provides a dedicated **NEU-DET Dataset Browser** tab with per-category browsing and statistics

#### Other datasets (optional)

You can also use images from:
- [MVTec Anomaly Detection Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- [Severstal Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection)

### 7. Generate synthetic sensor data

```bash
python generate_sensor_data.py
```

This creates `data/sensor_data.csv` with 192 rows of realistic sensor readings spanning a full shift (06:00–22:00) on Production Line 3, including a simulated cooling-valve drift incident.

### 8. Run the setup script

```bash
python setup_rag.py
```

This script:
- Loads and chunks documents from `docs/`
- Embeds chunks via the NVIDIA NIM embedding API
- Builds a FAISS vector index
- Initialises the SQLite database
- Populates defect events from the sensor CSV

### 9. Launch the application

```bash
streamlit run app.py
```

The UI will open at `http://localhost:8501`.

---

## Usage Guide

### Copilot Query Tab

Ask natural-language questions about production behaviour. The copilot retrieves relevant historical documents and correlates them with real-time sensor data.

**Example questions:**

- *"Why did the defect rate increase in the last hour on production line 3?"*
- *"What was the forming zone temperature when surface cracks started appearing?"*
- *"Has coolant valve V-17 caused issues before? What was the fix?"*
- *"Recommend corrective actions for the current defect spike."*
- *"Show me the correlation between coolant flow and defect rate."*

### Dashboard Tab

Real-time sensor charts:
- **Forming Zone Temperature** with warning (185°C) and critical (195°C) threshold lines
- **Defect Rate** over time
- **Coolant Flow %** over time
- Table of recent defect events

### Video Feed Tab

Displays sample images from `data/sample_images/` with anomaly detection overlays. When viewing **NEU-DET images**, the system uses ground-truth labels and draws bounding boxes from the XML annotations. For other images, a heuristic OpenCV-based detector is used. Click **Run Defect Simulation** in the sidebar to start the simulator.

### NEU-DET Dataset Tab

A dedicated browser for the NEU Surface Defect Database:
- **Dataset overview** — total images, category counts, bar chart
- **Category browser** — select a defect type (crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches) and navigate through images
- **Side-by-side view** — original image vs. annotated image with bounding boxes
- **Annotation details** — bounding box coordinates and labels
- **Grid preview** — thumbnail grid of the first 12 images per category

---

## Architecture Overview

```
┌────────────────┐    ┌──────────────────┐    ┌───────────────┐
│  Sensor CSV /  │    │  Manufacturing   │    │  Sample       │
│  Defect Sim    │    │  Documents       │    │  Images       │
└───────┬────────┘    └────────┬─────────┘    └───────┬───────┘
        │                      │                       │
        ▼                      ▼                       ▼
┌───────────────┐    ┌──────────────────┐    ┌───────────────┐
│  SQLite DB    │    │  FAISS Vector    │    │  OpenCV       │
│  (defect      │    │  Index           │    │  Processor    │
│   events)     │    │  (doc chunks)    │    │  (heuristic)  │
└───────┬───────┘    └────────┬─────────┘    └───────────────┘
        │                      │
        ▼                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    RAG Pipeline                             │
│  1. Query defect summary from DB                            │
│  2. Query sensor context from DB                            │
│  3. Retrieve relevant doc chunks from FAISS                 │
│  4. Assemble structured prompt                              │
│  5. Call NVIDIA NIM LLM (Llama 3.1 70B)                     │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Streamlit UI    │
                    │  (4-tab layout)  │
                    └──────────────────┘
```

**Data flow:**
1. `generate_sensor_data.py` creates a synthetic CSV with a cooling-valve drift narrative.
2. `setup_rag.py` ingests documents into a FAISS index and populates the SQLite DB.
3. When an operator asks a question, the RAG pipeline:
   - Pulls defect statistics and sensor readings from SQLite.
   - Embeds the query and retrieves the most relevant document chunks from FAISS.
   - Assembles everything into a structured prompt.
   - Sends it to the NVIDIA NIM LLM endpoint (Llama 3.1 70B Instruct).
4. The Streamlit UI displays the answer with source citations and latency metrics.

---

## Technology Stack

| Component | Technology |
|---|---|
| **UI** | Streamlit |
| **LLM** | NVIDIA NIM — Meta Llama 3.1 70B Instruct |
| **Embeddings** | NVIDIA NIM — nv-embedqa-e5-v5 |
| **Vector Store** | FAISS (CPU) |
| **Database** | SQLite |
| **Vision** | OpenCV (heuristic anomaly detection) |
| **Document Processing** | LangChain RecursiveCharacterTextSplitter |
| **Data** | Pandas, NumPy |
| **API Client** | OpenAI Python SDK (NVIDIA-compatible) |

---

## Latency & Scale Considerations

> *This section is a placeholder for the candidate to fill in with their analysis.*

- **Embedding latency:** Batch size of 10 balances throughput with API rate limits. Larger batch sizes may reduce total wall-clock time but risk 429 errors.
- **FAISS search:** IndexFlatL2 performs exact nearest-neighbour search — O(n) per query. For production scale (>100k chunks), consider IndexIVFFlat or IndexHNSW for sub-linear search.
- **LLM latency:** Dominated by the NVIDIA NIM API round-trip (~1–3s for 1024 tokens). Streaming responses could improve perceived latency.
- **Database:** SQLite is single-writer; for concurrent multi-line monitoring, migrate to PostgreSQL.
- **Real-time video:** The OpenCV heuristic runs locally in <50ms per frame. A production deployment would use NVIDIA NIM vision models with GPU inference for higher accuracy.

---

## Project Structure

```
manufacturing-copilot/
├── README.md
├── requirements.txt
├── .env.example
├── config.py                       # Environment variables & constants
├── app.py                          # Streamlit UI (4 tabs)
├── setup_rag.py                    # One-time setup: ingest docs + build DB
├── generate_sensor_data.py         # Generate synthetic sensor CSV
├── detection/
│   ├── __init__.py
│   ├── video_processor.py          # OpenCV frame processing + NEU-DET integration
│   ├── defect_simulator.py         # Replay sensor CSV as event stream
│   └── neu_det_loader.py           # NEU-DET annotation parser & dataset browser
├── data/
│   ├── sensor_data.csv             # (generated)
│   └── sample_images/              # (user-provided defect images)
│       └── NEU-DET/                # NEU Surface Defect Database (6 categories)
├── docs/                           # (user-provided manufacturing docs)
├── db/
│   ├── __init__.py
│   └── database.py                 # SQLite CRUD for defect events
├── rag/
│   ├── __init__.py
│   ├── ingest.py                   # Document loading + FAISS indexing
│   ├── retriever.py                # Vector similarity search
│   └── generator.py                # Prompt assembly + LLM call
└── utils/
    ├── __init__.py
    └── metrics.py                  # Latency tracking
```

---

## License

MIT



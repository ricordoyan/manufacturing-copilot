"""Generate architecture-diagram.png for the Manufacturing Defect Detection Copilot."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── Colours ──────────────────────────────────────────────────────────────────
BG           = "#0f172a"   # dark navy background
CARD_DATA    = "#1e3a5f"   # data sources — deep blue
CARD_PROCESS = "#1a3c34"   # processing layer — dark teal
CARD_NIM     = "#4a1942"   # NVIDIA NIM — purple
CARD_UI      = "#3b1f0b"   # UI — dark amber
BORDER_DATA  = "#38bdf8"   # sky-400
BORDER_PROC  = "#34d399"   # emerald-400
BORDER_NIM   = "#a78bfa"   # violet-400
BORDER_UI    = "#fb923c"   # orange-400
TEXT_WHITE   = "#f1f5f9"
TEXT_MUTED   = "#94a3b8"
ARROW_CLR    = "#64748b"
ACCENT_GREEN = "#76b900"   # NVIDIA green

fig, ax = plt.subplots(figsize=(18, 13))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 18)
ax.set_ylim(0, 13)
ax.axis("off")

# ── Helpers ──────────────────────────────────────────────────────────────────

def draw_box(x, y, w, h, fill, edge, label, sublabel=None, icon=None, fontsize=11):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.15",
        facecolor=fill, edgecolor=edge, linewidth=2,
    )
    ax.add_patch(box)
    text = f"{icon}  {label}" if icon else label
    ax.text(x + w / 2, y + h / 2 + (0.15 if sublabel else 0),
            text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=TEXT_WHITE)
    if sublabel:
        ax.text(x + w / 2, y + h / 2 - 0.25,
                sublabel, ha="center", va="center",
                fontsize=8.5, color=TEXT_MUTED, style="italic")


def draw_arrow(x1, y1, x2, y2, color=ARROW_CLR, style="-|>", lw=1.5):
    ax.annotate("",
                xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                connectionstyle="arc3,rad=0"))


def draw_section_label(x, y, text, color):
    ax.text(x, y, text, fontsize=9, fontweight="bold", color=color,
            ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color + "22",
                      edgecolor=color, linewidth=1.2))


# ═══════════════════════════════════════════════════════════════════════════
#  TITLE
# ═══════════════════════════════════════════════════════════════════════════
ax.text(9, 12.5, "Manufacturing Defect Detection Copilot — Architecture",
        ha="center", va="center", fontsize=18, fontweight="bold",
        color=ACCENT_GREEN)
ax.text(9, 12.1, "NVIDIA NIM  ·  RAG Pipeline  ·  Streamlit  ·  FAISS  ·  SQLite  ·  OpenCV",
        ha="center", va="center", fontsize=10, color=TEXT_MUTED)

# ═══════════════════════════════════════════════════════════════════════════
#  ROW 1 — DATA SOURCES  (y ≈ 9.5 – 11.2)
# ═══════════════════════════════════════════════════════════════════════════
draw_section_label(0.3, 11.4, "DATA SOURCES", BORDER_DATA)

# sensor CSV
draw_box(0.5, 9.5, 3.5, 1.5, CARD_DATA, BORDER_DATA,
         "Sensor CSV", "sensor_data.csv  |  193 rows")
# Manufacturing docs
draw_box(5.0, 9.5, 3.5, 1.5, CARD_DATA, BORDER_DATA,
         "Mfg. Documents", "8 Markdown docs  |  docs/")
# Sample images
draw_box(9.5, 9.5, 3.5, 1.5, CARD_DATA, BORDER_DATA,
         "Defect Images", "MVTec + Severstal  |  24k imgs")
# Defect simulator
draw_box(14.0, 9.5, 3.5, 1.5, CARD_DATA, BORDER_DATA,
         "Defect Simulator", "defect_simulator.py")

# ═══════════════════════════════════════════════════════════════════════════
#  ROW 2 — PROCESSING / STORAGE  (y ≈ 6.5 – 8.2)
# ═══════════════════════════════════════════════════════════════════════════
draw_section_label(0.3, 8.7, "PROCESSING & STORAGE", BORDER_PROC)

# SQLite
draw_box(0.5, 6.5, 3.5, 1.5, CARD_PROCESS, BORDER_PROC,
         "SQLite DB", "defects.db  |  defect events")
# FAISS
draw_box(5.0, 6.5, 3.5, 1.5, CARD_PROCESS, BORDER_PROC,
         "FAISS Index", "72 vectors  |  dim 1024")
# OpenCV
draw_box(9.5, 6.5, 3.5, 1.5, CARD_PROCESS, BORDER_PROC,
         "OpenCV Processor", "Heuristic anomaly detection")
# LangChain chunker
draw_box(14.0, 6.5, 3.5, 1.5, CARD_PROCESS, BORDER_PROC,
         "Text Splitter", "LangChain  |  800 / 150 chars")

# ── Arrows: data → processing ───────────────────────────────────────────
draw_arrow(2.25, 9.5, 2.25, 8.0, BORDER_DATA)   # CSV → SQLite
draw_arrow(6.75, 9.5, 6.75, 8.0, BORDER_DATA)   # Docs → FAISS
draw_arrow(11.25, 9.5, 11.25, 8.0, BORDER_DATA) # Images → OpenCV
draw_arrow(15.75, 9.5, 15.75, 8.0, BORDER_DATA) # Sim → Splitter
# Docs also go through splitter
draw_arrow(6.75, 9.5, 15.0, 8.0, BORDER_DATA)

# ═══════════════════════════════════════════════════════════════════════════
#  ROW 3 — NVIDIA NIM + RAG PIPELINE  (y ≈ 3.5 – 5.5)
# ═══════════════════════════════════════════════════════════════════════════
draw_section_label(0.3, 5.9, "RAG PIPELINE  ·  NVIDIA NIM", BORDER_NIM)

# Large central RAG box
rag_box = FancyBboxPatch(
    (0.5, 3.5), 17.0, 2.0,
    boxstyle="round,pad=0.2",
    facecolor=CARD_NIM, edgecolor=BORDER_NIM, linewidth=2.5,
)
ax.add_patch(rag_box)

# Steps inside RAG box
steps = [
    ("① Query\nDefect DB", 1.7),
    ("② Sensor\nContext", 4.6),
    ("③ Embed\nQuery", 7.2),
    ("④ Retrieve\nDocs (FAISS)", 9.8),
    ("⑤ Build\nPrompt", 12.4),
    ("⑥ LLM Call\n(Llama 3.1 70B)", 15.3),
]
for label, cx in steps:
    inner = FancyBboxPatch(
        (cx - 1.1, 3.7), 2.2, 1.5,
        boxstyle="round,pad=0.1",
        facecolor="#2a1a42", edgecolor=BORDER_NIM, linewidth=1, alpha=0.7,
    )
    ax.add_patch(inner)
    ax.text(cx, 4.45, label, ha="center", va="center",
            fontsize=9, fontweight="bold", color=TEXT_WHITE, linespacing=1.4)

# Step arrows inside RAG box
for i in range(len(steps) - 1):
    x1 = steps[i][1] + 1.1
    x2 = steps[i + 1][1] - 1.1
    draw_arrow(x1, 4.5, x2, 4.5, ACCENT_GREEN, "-|>", 1.8)

# ── Arrows: processing → RAG ────────────────────────────────────────────
draw_arrow(2.25, 6.5, 1.7, 5.5, BORDER_PROC)   # SQLite → Step 1
draw_arrow(2.25, 6.5, 4.6, 5.5, BORDER_PROC)   # SQLite → Step 2
draw_arrow(6.75, 6.5, 9.8, 5.5, BORDER_PROC)   # FAISS → Step 4

# NVIDIA NIM badge
ax.text(16.8, 5.25, "NVIDIA NIM API",
        fontsize=8, fontweight="bold", color=ACCENT_GREEN, ha="right",
        bbox=dict(boxstyle="round,pad=0.2", facecolor=ACCENT_GREEN + "22",
                  edgecolor=ACCENT_GREEN, linewidth=1))

# ═══════════════════════════════════════════════════════════════════════════
#  ROW 4 — STREAMLIT UI  (y ≈ 0.5 – 2.5)
# ═══════════════════════════════════════════════════════════════════════════
draw_section_label(0.3, 3.0, "STREAMLIT UI", BORDER_UI)

# UI container
ui_box = FancyBboxPatch(
    (0.5, 0.5), 17.0, 2.2,
    boxstyle="round,pad=0.2",
    facecolor=CARD_UI, edgecolor=BORDER_UI, linewidth=2.5,
)
ax.add_patch(ui_box)

# Three tabs
draw_box(1.0, 0.8, 4.5, 1.6, "#4a2a12", BORDER_UI,
         "Copilot Query", "Sample Q's  |  Correlation panel  |  Citations", fontsize=10)
draw_box(6.5, 0.8, 4.5, 1.6, "#4a2a12", BORDER_UI,
         "Dashboard", "Plotly charts  |  Temp / Flow / Defects", fontsize=10)
draw_box(12.0, 0.8, 5.0, 1.6, "#4a2a12", BORDER_UI,
         "Video Feed", "Frame stepper  |  Defect overlay  |  Log", fontsize=10)

# ── Arrows: RAG → UI ────────────────────────────────────────────────────
draw_arrow(3.25, 3.5, 3.25, 2.4, BORDER_NIM)   # RAG → Copilot tab
draw_arrow(9.0, 3.5, 8.75, 2.4, BORDER_NIM)    # RAG → Dashboard
draw_arrow(11.25, 6.5, 14.5, 2.4, BORDER_PROC) # OpenCV → Video tab

# ═══════════════════════════════════════════════════════════════════════════
#  LEGEND
# ═══════════════════════════════════════════════════════════════════════════
legend_items = [
    mpatches.Patch(facecolor=CARD_DATA, edgecolor=BORDER_DATA, linewidth=1.5, label="Data Sources"),
    mpatches.Patch(facecolor=CARD_PROCESS, edgecolor=BORDER_PROC, linewidth=1.5, label="Processing & Storage"),
    mpatches.Patch(facecolor=CARD_NIM, edgecolor=BORDER_NIM, linewidth=1.5, label="NVIDIA NIM / RAG"),
    mpatches.Patch(facecolor=CARD_UI, edgecolor=BORDER_UI, linewidth=1.5, label="Streamlit UI"),
]
leg = ax.legend(handles=legend_items, loc="lower right", frameon=True,
                facecolor=BG, edgecolor=TEXT_MUTED, labelcolor=TEXT_WHITE,
                fontsize=9, ncol=4,
                bbox_to_anchor=(0.98, -0.02))
leg.get_frame().set_linewidth(1)

plt.tight_layout()
plt.savefig("architecture-diagram.png", dpi=180, bbox_inches="tight",
            facecolor=BG, edgecolor="none")
print("✅ Saved architecture-diagram.png")

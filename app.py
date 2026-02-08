"""
Manufacturing Defect Detection Copilot — Streamlit UI.

Run:
    streamlit run app.py
"""

import os
import sys
import threading
import time
from datetime import datetime

import cv2
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    NVIDIA_API_KEY,
    SAMPLE_IMAGES_DIR,
    SENSOR_DATA_PATH,
    TEMP_CRITICAL,
    TEMP_WARNING,
)
from db.database import (
    get_all_sensor_data,
    get_defect_summary,
    get_recent_defects,
    get_sensor_context,
    init_db,
    log_defect_event,
)
from detection.defect_simulator import DefectSimulator
from detection.video_processor import VideoProcessor
from detection.neu_det_loader import NEUDatasetLoader
from rag.generator import query_copilot

# ── Page configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Manufacturing Defect Detection Copilot",
    page_icon="🏭",
    layout="wide",
)

# ── Session state defaults ──────────────────────────────────────────────────
if "simulation_running" not in st.session_state:
    st.session_state.simulation_running = False
if "simulation_done" not in st.session_state:
    st.session_state.simulation_done = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_query_result" not in st.session_state:
    st.session_state.last_query_result = None
if "sim_log" not in st.session_state:
    st.session_state.sim_log = []
if "frame_index" not in st.session_state:
    st.session_state.frame_index = 0
if "detection_log" not in st.session_state:
    st.session_state.detection_log = []

# ── Time-window mapping ────────────────────────────────────────────────────
TIME_WINDOW_MAP = {
    "Last 30 min": 0.5,
    "Last 1 hour": 1.0,
    "Last 2 hours": 2.0,
    "Last shift (8 hours)": 8.0,
}

LINE_OPTIONS = ["LINE-1", "LINE-2", "LINE-3", "LINE-4", "LINE-5"]


# ── Helper: run simulation in background thread ────────────────────────────
def _run_simulation() -> None:
    """Run the defect simulator and log events to the DB.

    Executed in a daemon thread so the Streamlit UI stays responsive.
    """
    try:
        sim = DefectSimulator()
        for event in sim.run_simulation(speed_multiplier=60):
            log_defect_event(event)
            st.session_state.sim_log.append(event)
    except Exception as exc:
        st.session_state.sim_log.append({"error": str(exc)})
    finally:
        st.session_state.simulation_running = False
        st.session_state.simulation_done = True


# ── Helper: check NVIDIA NIM API reachability ──────────────────────────────
def _api_status() -> bool:
    return bool(NVIDIA_API_KEY)


# ── Plotly chart helpers ────────────────────────────────────────────────────

def create_temperature_chart(sensor_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sensor_df["timestamp"],
        y=sensor_df["forming_zone_temp_c"],
        mode="lines",
        name="Forming Zone Temp",
        line=dict(color="#ef4444", width=2),
    ))
    fig.add_hline(y=TEMP_WARNING, line_dash="dash", line_color="orange",
                  annotation_text=f"Warning ({TEMP_WARNING}°C)")
    fig.add_hline(y=TEMP_CRITICAL, line_dash="dash", line_color="red",
                  annotation_text=f"Critical ({TEMP_CRITICAL}°C)")
    fig.update_yaxes(range=[155, 200], title="Temperature (°C)")
    fig.update_xaxes(title="Time")
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def create_defect_rate_chart(sensor_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sensor_df["timestamp"],
        y=sensor_df["defect_rate_pct"],
        fill="tozeroy",
        name="Defect Rate",
        line=dict(color="#3b82f6", width=2),
        fillcolor="rgba(59, 130, 246, 0.3)",
    ))
    fig.add_hline(y=3.0, line_dash="dash", line_color="orange",
                  annotation_text="Action Threshold (3%)")
    fig.update_yaxes(title="Defect Rate (%)")
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
    return fig


def create_coolant_flow_chart(sensor_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sensor_df["timestamp"],
        y=sensor_df["coolant_flow_pct"],
        mode="lines",
        name="Coolant Flow",
        line=dict(color="#10b981", width=2),
    ))
    fig.add_hline(y=80, line_dash="dash", line_color="orange",
                  annotation_text="Low Flow Warning (80%)")
    fig.update_yaxes(range=[60, 105], title="Flow (%)")
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ System Status")

    # API status indicator
    if _api_status():
        st.markdown("🟢 NVIDIA NIM API key configured")
    else:
        st.markdown("🔴 NVIDIA NIM API key **missing** — set `NVIDIA_API_KEY` in `.env`")

    # Simulation indicator
    if st.session_state.simulation_running:
        st.markdown("🟡 Simulation running …")
    elif st.session_state.simulation_done:
        st.markdown("🟢 Simulation complete")
    else:
        st.markdown("⚪ Simulation idle")

    st.divider()

    # Time window selector
    time_label = st.selectbox("⏱️ Time window", list(TIME_WINDOW_MAP.keys()), index=1)
    time_hours = TIME_WINDOW_MAP[time_label]

    # Line selector
    selected_line = st.selectbox("🏭 Production line", LINE_OPTIONS, index=2)

    st.divider()

    # Simulation trigger
    if st.button(
        "▶️ Run Defect Simulation",
        disabled=st.session_state.simulation_running,
    ):
        init_db()
        st.session_state.simulation_running = True
        st.session_state.simulation_done = False
        st.session_state.sim_log = []
        thread = threading.Thread(target=_run_simulation, daemon=True)
        thread.start()
        st.rerun()

    st.divider()

    # Quick stats
    st.subheader("📊 Quick Stats")
    try:
        summary = get_defect_summary(hours=time_hours, line_id=selected_line)
        col1, col2 = st.columns(2)
        col1.metric("Defects", summary["total_defects"])
        col2.metric("Avg rate", f"{summary['defect_rate_avg']}%")
        st.metric("Peak temp (avg during defects)", f"{summary['avg_temp_during_defects']}°C")
    except Exception:
        st.info("No data yet — run the simulation or setup_rag.py first.")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN AREA — TABS
# ═══════════════════════════════════════════════════════════════════════════

tab_copilot, tab_dashboard, tab_video, tab_neudet = st.tabs(
    ["🔍 Copilot Query", "📈 Dashboard", "🎥 Video Feed", "🔬 NEU-DET Dataset"]
)

# ── Tab 1: Copilot Query ────────────────────────────────────────────────────
with tab_copilot:
    st.subheader("🔍 Ask the Manufacturing Copilot")

    # ── Sample questions ────────────────────────────────────────────────
    st.caption("Try these example questions:")

    def _select_sample(question: str) -> None:
        """Callback: write the chosen question into the text-area's state."""
        st.session_state.query_input = question

    sample_cols = st.columns(3)
    sample_questions = [
        "Why did the defect rate increase in the last hour on production line 3?",
        "Has this type of defect happened before on line 3?",
        "What is the maintenance history of valve V-17?",
        "What should I do about the elevated forming zone temperature?",
        "Was there a pressure issue today on line 3?",
        "What are the common causes of surface cracks?",
    ]
    for i, q in enumerate(sample_questions):
        col = sample_cols[i % 3]
        with col:
            st.button(q, key=f"sample_{i}", width="stretch",
                      on_click=_select_sample, args=(q,))

    user_question = st.text_area(
        "Ask about production behaviour…",
        placeholder="Why did the defect rate increase in the last hour on production line 3?",
        height=100,
        key="query_input",
    )

    if st.button("Submit", type="primary"):
        if not user_question.strip():
            st.warning("Please enter a question.")
        elif not _api_status():
            st.error(
                "NVIDIA API key is not configured. "
                "Set `NVIDIA_API_KEY` in your `.env` file."
            )
        else:
            with st.spinner(
                "Analyzing defect data, sensor readings, and historical documents…"
            ):
                try:
                    result = query_copilot(
                        user_question,
                        time_window_hours=time_hours,
                        line_id=selected_line,
                    )
                    st.session_state.last_query_result = result
                    st.session_state.chat_history.append(
                        {"question": user_question, "result": result}
                    )
                except Exception as exc:
                    st.error(f"An error occurred: {exc}")

    # Display latest result
    result = st.session_state.last_query_result
    if result:
        st.markdown("---")
        st.markdown("### 💡 Copilot Response")
        st.markdown(result["answer"])

        # ── Visual correlation panel ────────────────────────────────────
        st.divider()
        st.subheader("📊 Supporting Data")
        m = result.get("metrics", {})

        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        with metric_col1:
            st.metric(
                "Peak Temperature",
                f"{m.get('peak_temp', 'N/A')}°C",
                delta=f"+{m.get('temp_above_threshold', 0):.1f}°C above threshold",
                delta_color="inverse",
            )
        with metric_col2:
            st.metric(
                "Min Coolant Flow",
                f"{m.get('min_flow', 'N/A')}%",
                delta=f"{m.get('flow_below_nominal', 0):.1f}% below nominal",
                delta_color="inverse",
            )
        with metric_col3:
            st.metric(
                "Defect Count",
                m.get("defect_count", "N/A"),
                delta=f"{m.get('rate_vs_baseline', '')} vs baseline",
                delta_color="inverse",
            )
        with metric_col4:
            st.metric(
                "Response Time",
                f"{m.get('total_latency', 0):.2f}s",
            )

        with st.expander("📄 Sources Referenced"):
            if result["sources"]:
                for src in result["sources"]:
                    st.code(src)
            else:
                st.info("No documents were referenced (FAISS index may not be built).")

        with st.expander("⏱️ Performance Metrics"):
            latency = result.get("latency_breakdown", result.get("metrics", {}))
            for step, dur in latency.items():
                if isinstance(dur, (int, float)):
                    st.text(f"{step}: {dur:.4f}s")

    # Conversation history
    if len(st.session_state.chat_history) > 1:
        st.markdown("---")
        with st.expander("📜 Conversation History"):
            for i, entry in enumerate(reversed(st.session_state.chat_history[:-1]), 1):
                st.markdown(f"**Q{i}:** {entry['question']}")
                st.markdown(f"**A{i}:** {entry['result']['answer'][:300]}…")
                st.markdown("---")

# ── Tab 2: Dashboard ────────────────────────────────────────────────────────
with tab_dashboard:
    st.subheader("📈 Sensor & Defect Dashboard")

    sensor_df = get_all_sensor_data(line_id=selected_line)

    if sensor_df.empty:
        st.warning(
            "Sensor data not found. Run `python generate_sensor_data.py` first."
        )
    else:
        # Forming zone temperature chart
        st.markdown("#### 🌡️ Forming Zone Temperature (°C)")
        st.plotly_chart(create_temperature_chart(sensor_df), width="stretch")

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### ⚠️ Defect Rate (%)")
            st.plotly_chart(create_defect_rate_chart(sensor_df), width="stretch")

        with col_right:
            st.markdown("#### 💧 Coolant Flow (%)")
            st.plotly_chart(create_coolant_flow_chart(sensor_df), width="stretch")

        # Recent defect events table
        st.markdown("#### 🔎 Recent Defect Events")
        try:
            recent = get_recent_defects(hours=time_hours, line_id=selected_line)
            if recent:
                events_df = pd.DataFrame(recent)[
                    ["timestamp", "defect_type", "confidence", "forming_zone_temp_c"]
                ].head(20)
                st.dataframe(events_df, width="stretch", hide_index=True)
            else:
                st.info("No defect events in the selected time window.")
        except Exception:
            st.info("No defect data available — run the simulation or setup_rag.py first.")

# ── Tab 3: Video Feed ──────────────────────────────────────────────────────

@st.cache_resource
def _get_video_processor():
    """Cache the VideoProcessor so we don't re-walk 24 k+ images every rerun."""
    return VideoProcessor(SAMPLE_IMAGES_DIR)

with tab_video:
    st.subheader("🎥 Production Line Video Feed")

    if not os.path.isdir(SAMPLE_IMAGES_DIR) or not os.listdir(SAMPLE_IMAGES_DIR):
        st.info(
            "Video feed not active. Place images in `data/sample_images/` "
            "and click **Run Defect Simulation** in the sidebar to start."
        )
    else:
        vp = _get_video_processor()
        images = vp.get_image_paths()

        if images:
            col_img, col_info = st.columns([2, 1])

            with col_img:
                if st.button("⏭️ Next Frame"):
                    st.session_state.frame_index += 1

                frame_index = st.session_state.frame_index
                current_path = images[frame_index % len(images)]
                frame = cv2.imread(current_path)

                if frame is not None:
                    # Run NEU-DET–aware detection
                    detection = vp.detect_with_neu_annotations(frame, current_path)

                    # Use annotated frame (with bounding boxes) if available
                    if detection.get("annotated_frame") is not None:
                        display_frame = detection["annotated_frame"]
                    elif detection["has_defect"]:
                        display_frame = frame.copy()
                        cv2.rectangle(
                            display_frame, (0, 0),
                            (display_frame.shape[1] - 1, display_frame.shape[0] - 1),
                            (0, 0, 255), 5,
                        )
                    else:
                        display_frame = frame

                    rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb)
                    st.image(
                        pil_img,
                        caption=f"Frame {frame_index + 1}/{len(images)} — {os.path.basename(current_path)}",
                        width="stretch",
                    )
                else:
                    detection = {"has_defect": False, "confidence": 0, "anomaly_percentage": 0, "defect_type": None, "ground_truth": False}
                    st.warning(f"Could not read image: {current_path}")

            with col_info:
                st.markdown("**Detection Result:**")
                if detection["has_defect"]:
                    st.error("⚠️ DEFECT DETECTED")
                    st.metric("Defect Type", (detection["defect_type"] or "unknown").replace("_", " ").title())
                    st.metric("Confidence", f"{detection['confidence']:.1%}")
                    st.metric("Anomaly Area", f"{detection['anomaly_percentage']:.1f}%")
                    if detection.get("ground_truth"):
                        st.info("🏷️ Label: NEU-DET ground truth")
                    annotation = detection.get("annotation")
                    if annotation and annotation.boxes:
                        st.caption(f"📦 {len(annotation.boxes)} bounding box(es)")
                else:
                    st.success("✅ No defect")
                    st.metric("Anomaly Area", f"{detection['anomaly_percentage']:.1f}%")

                st.divider()
                st.markdown("**Detection Threshold:**")
                threshold = st.slider(
                    "Anomaly threshold (%)", 1.0, 15.0, 5.0, 0.5,
                    help="Lower = more sensitive. Adjust based on product type.",
                )

            # Log defects
            if detection["has_defect"]:
                st.session_state.detection_log.insert(0, {
                    "frame": frame_index + 1,
                    "file": os.path.basename(current_path),
                    "type": detection["defect_type"],
                    "confidence": detection["confidence"],
                    "anomaly_pct": detection["anomaly_percentage"],
                })
                st.session_state.detection_log = st.session_state.detection_log[:10]
        else:
            st.info("No supported images found in `data/sample_images/`.")

    # Detection log
    st.divider()
    st.subheader("📋 Detection Log (Last 10)")
    if st.session_state.detection_log:
        st.dataframe(
            pd.DataFrame(st.session_state.detection_log),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No defects detected yet. Step through frames to populate the log.")

    # Simulation log
    st.markdown("#### 📋 Simulation Events")
    if st.session_state.sim_log:
        log_items = st.session_state.sim_log[-10:]
        for entry in reversed(log_items):
            if "error" in entry:
                st.error(f"Simulation error: {entry['error']}")
            else:
                st.text(
                    f"[{entry.get('timestamp', '?')}]  "
                    f"{entry.get('defect_type', '?')}  "
                    f"conf={entry.get('confidence', 0):.2f}  "
                    f"temp={entry.get('forming_zone_temp_c', '?')}°C"
                )
    else:
        st.info(
            "No simulation events yet. Click **Run Defect Simulation** in the sidebar."
        )

# ── Tab 4: NEU-DET Dataset Browser ─────────────────────────────────────────
with tab_neudet:
    st.subheader("🔬 NEU Surface Defect Database Browser")
    st.caption(
        "Browse the NEU-DET dataset by defect category. "
        "Images are shown with ground-truth bounding-box annotations."
    )

    from config import NEU_DET_TRAIN_IMAGES, NEU_DET_TRAIN_ANNOTATIONS

    if not os.path.isdir(NEU_DET_TRAIN_IMAGES):
        st.warning(
            "NEU-DET dataset not found. Expected at "
            "`data/sample_images/NEU-DET/train/images/`."
        )
    else:
        @st.cache_resource
        def _get_neu_loader():
            return NEUDatasetLoader(NEU_DET_TRAIN_IMAGES, NEU_DET_TRAIN_ANNOTATIONS)

        neu = _get_neu_loader()
        stats = neu.get_dataset_stats()

        # ── Dataset overview metrics ────────────────────────────────────
        st.markdown("#### 📊 Dataset Overview")
        overview_cols = st.columns(3)
        overview_cols[0].metric("Total Images", stats["total_images"])
        overview_cols[1].metric("Defect Categories", stats["categories"])
        overview_cols[2].metric("Annotations Loaded", stats["total_annotations"])

        # Per-category counts
        cat_counts = stats["category_counts"]
        cat_df = pd.DataFrame(
            [{"Category": k.replace("_", " ").title(), "Count": v}
             for k, v in cat_counts.items()]
        )
        st.bar_chart(cat_df, x="Category", y="Count")

        st.divider()

        # ── Category browser ────────────────────────────────────────────
        st.markdown("#### 🖼️ Browse by Category")
        selected_cat = st.selectbox(
            "Defect category",
            neu.categories,
            format_func=lambda c: f"{c.replace('_', ' ').title()} ({len(neu.images_for_category(c))} images)",
        )

        cat_images = neu.images_for_category(selected_cat)

        if "neu_browse_index" not in st.session_state:
            st.session_state.neu_browse_index = 0

        if cat_images:
            nav_col1, nav_col2, nav_col3 = st.columns([1, 3, 1])
            with nav_col1:
                if st.button("⏮️ Prev", key="neu_prev"):
                    st.session_state.neu_browse_index = max(
                        0, st.session_state.neu_browse_index - 1
                    )
            with nav_col3:
                if st.button("⏭️ Next", key="neu_next"):
                    st.session_state.neu_browse_index = min(
                        len(cat_images) - 1, st.session_state.neu_browse_index + 1
                    )
            with nav_col2:
                idx = st.slider(
                    "Image index",
                    0,
                    len(cat_images) - 1,
                    st.session_state.neu_browse_index,
                    key="neu_slider",
                )
                st.session_state.neu_browse_index = idx

            img_path = cat_images[st.session_state.neu_browse_index % len(cat_images)]
            img_frame = cv2.imread(img_path)

            if img_frame is not None:
                col_original, col_annotated = st.columns(2)
                annotation = neu.get_annotation(img_path)

                with col_original:
                    st.markdown("**Original**")
                    rgb_orig = cv2.cvtColor(img_frame, cv2.COLOR_BGR2RGB)
                    st.image(Image.fromarray(rgb_orig), width="stretch")

                with col_annotated:
                    st.markdown("**With Annotations**")
                    if annotation:
                        annotated = neu.draw_annotations(img_frame, annotation)
                        rgb_ann = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        st.image(Image.fromarray(rgb_ann), width="stretch")

                        # Annotation details
                        st.caption(
                            f"**File:** {annotation.filename}  |  "
                            f"**Size:** {annotation.width}×{annotation.height}  |  "
                            f"**Boxes:** {len(annotation.boxes)}"
                        )
                        for i, box in enumerate(annotation.boxes):
                            st.text(
                                f"  Box {i+1}: [{box.xmin}, {box.ymin}, "
                                f"{box.xmax}, {box.ymax}] — {box.label}"
                            )
                    else:
                        rgb_orig2 = cv2.cvtColor(img_frame, cv2.COLOR_BGR2RGB)
                        st.image(Image.fromarray(rgb_orig2), width="stretch")
                        st.info("No XML annotation found for this image.")
            else:
                st.warning(f"Could not read image: {img_path}")

        # ── Grid preview ────────────────────────────────────────────────
        st.divider()
        st.markdown("#### 🗂️ Category Grid (first 12 images)")
        grid_cols = st.columns(4)
        for i, path in enumerate(cat_images[:12]):
            col = grid_cols[i % 4]
            gframe = cv2.imread(path)
            if gframe is not None:
                rgb_g = cv2.cvtColor(gframe, cv2.COLOR_BGR2RGB)
                col.image(
                    Image.fromarray(rgb_g),
                    caption=os.path.basename(path),
                    width="stretch",
                )

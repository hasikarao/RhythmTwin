import streamlit as st
import numpy as np
import pandas as pd
import wfdb
import plotly.graph_objects as go
import joblib

from pathlib import Path
from datetime import datetime
from scipy.signal import resample

# ---------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="RhythmTwin",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).resolve().parent
MODEL_PATH = DATA_DIR / "rhythmtwin_model.joblib"

AVAILABLE_RECORDS = [
    "100", "101", "102", "103", "104",
    "106", "107", "108", "109", "111",
    "112", "113",
]

CLASS_NAMES = {
    "N": "Normal-like beat",
    "S": "Supraventricular ectopic class",
    "V": "Ventricular ectopic class",
    "F": "Fusion beat class",
    "Q": "Paced/other class",
}

# (card label, accent colour)
CARD_INFO = {
    "N": ("Normal-like", "#2dd4bf"),
    "S": ("Supraventricular ectopic", "#f472b6"),
    "V": ("Ventricular ectopic", "#f87171"),
    "F": ("Fusion", "#fbbf24"),
    "Q": ("Paced / other", "#a78bfa"),
}
CLASS_ORDER = ["N", "S", "V", "F", "Q"]

# Maps MIT-BIH annotation symbols to the five simplified classes.
CLASS_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q",
}

# Exploratory V-class threshold (unchanged from the original app).
V_THRESHOLD = 0.10

PAGES = ["Dashboard", "About", "Methodology", "Results"]
PAGE_ICONS = {"Dashboard": "🏠", "About": "ℹ️", "Methodology": "🧪", "Results": "📊"}


def md(html_str: str) -> None:
    """Render an HTML snippet. Whitespace is collapsed so Markdown never
    mistakes indented HTML for a code block."""
    st.markdown(" ".join(html_str.split()), unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_saved_model(path_str: str):
    return joblib.load(path_str)


# ---------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------
md("""
<style>
.stApp {
    background: radial-gradient(1200px 600px at 80% -10%, #0e2a4a 0%, #07111f 55%) fixed;
    color: #edf5ff;
}
header[data-testid="stHeader"] { background: transparent; }
footer, #MainMenu { visibility: hidden; }
.block-container { max-width: 1600px; padding-top: 1rem; padding-bottom: 2rem; }

h1, h2, h3, h4 { color: #f1f7ff; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c1b2f 0%, #091525 100%);
    border-right: 1px solid #203650;
}
.rt-side-brand { text-align: center; padding: 8px 0 14px 0; }
.rt-side-brand .logo { font-size: 34px; color: #42e0c1; line-height: 1; }
.rt-side-brand .name { font-size: 22px; font-weight: 800; color: #f1f7ff; margin-top: 4px; }
.rt-side-brand .sub { font-size: 12px; color: #8fa8c4; margin-top: 2px; }

[data-testid="stSidebar"] div[role="radiogroup"] { gap: 6px; }
[data-testid="stSidebar"] div[role="radiogroup"] label {
    padding: 10px 14px; border-radius: 10px; width: 100%;
    border: 1px solid transparent; cursor: pointer;
}
[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child { display: none; }
[data-testid="stSidebar"] div[role="radiogroup"] label p { color: #c9d9ec; font-size: 15px; font-weight: 600; }
[data-testid="stSidebar"] div[role="radiogroup"] label:hover { background: #12294421; background: rgba(66,224,193,0.07); }
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(90deg, rgba(13,181,165,0.28), rgba(8,127,140,0.12));
    border-color: #1d6f78;
}
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p { color: #42e0c1; }

.rt-disclaimer {
    margin-top: 18px; padding: 12px 14px; border-radius: 12px;
    background: rgba(248,113,113,0.08); border: 1px solid rgba(248,113,113,0.35);
    color: #d7c3cb; font-size: 12.5px; line-height: 1.5;
}
.rt-disclaimer b { color: #fb8f8f; display: block; margin-bottom: 4px; font-size: 13px; }

/* ---------- Top navbar ---------- */
.rt-topnav {
    display: flex; justify-content: space-between; align-items: center;
    padding: 10px 20px; margin: 0 0 14px 0;
    background: linear-gradient(100deg, #0c2037, #111a3c);
    border: 1px solid #243e5b; border-radius: 14px;
}
.rt-topnav .brand { color: #42e0c1; font-size: 20px; font-weight: 800; }
.rt-topnav .brand small { color: #8fa8c4; font-size: 12px; font-weight: 500; margin-left: 10px; }
.rt-topnav .links a {
    color: #c9d9ec; text-decoration: none; font-size: 14px; font-weight: 600;
    padding: 8px 14px; margin-left: 4px; border-bottom: 2px solid transparent;
}
.rt-topnav .links a:hover { color: #42e0c1; }
.rt-topnav .links a.active { color: #ffffff; border-bottom-color: #42e0c1; }

/* ---------- Hero ---------- */
.rt-hero {
    position: relative; overflow: hidden;
    padding: 22px 28px; margin-bottom: 14px;
    background: linear-gradient(110deg, #0b1d33 0%, #0a1830 60%, #0d2a4a 100%);
    border: 1px solid #1f3a57; border-radius: 18px;
}
.rt-hero .eyebrow { color: #42e0c1; font-size: 12px; font-weight: 700; letter-spacing: 3px; }
.rt-hero h1 { font-size: 40px; line-height: 1.1; margin: 8px 0 8px 0; padding: 0; color: #fff; }
.rt-hero h1 span { color: #2de0c4; }
.rt-hero p { color: #9bb3cc; font-size: 16px; margin: 0; }
.rt-hero svg { position: absolute; right: 24px; top: 24px; width: 42%; height: 110px; opacity: 0.85; }
.rt-hero .heart {
    position: absolute; right: 22%; top: 18px; font-size: 90px; opacity: 0.18; color: #3aa0ff;
}

/* ---------- Cards / headings ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(160deg, rgba(15,32,54,0.92), rgba(9,21,37,0.92));
    border: 1px solid #1f3a57 !important;
    border-radius: 16px;
}
.rt-title { font-size: 17px; font-weight: 700; color: #f1f7ff; margin: 0 0 8px 0; }
.rt-title .ic { color: #42e0c1; margin-right: 8px; }
.rt-sub { color: #8fa8c4; font-size: 12.5px; }
.rt-chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 0 0 8px 0; }
.rt-chip {
    background: rgba(66,224,193,0.08); border: 1px solid #1d5a60; color: #9fe9db;
    border-radius: 999px; padding: 3px 12px; font-size: 12px; font-weight: 600;
}
.rt-profile { color: #c9d9ec; font-size: 14px; line-height: 1.7; margin-top: 6px; }
.rt-profile b { color: #f1f7ff; }

/* ---------- Prediction cards ---------- */
.rt-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.rt-pcard {
    border-radius: 14px; padding: 12px 14px; border: 1px solid; position: relative;
}
.rt-pcard.wide { grid-column: span 2; }
.rt-pcard .code { font-size: 26px; font-weight: 800; line-height: 1; }
.rt-pcard .lbl { font-size: 12px; color: #c9d9ec; margin-top: 2px; }
.rt-pcard .num { font-size: 30px; font-weight: 800; color: #fff; margin-top: 6px; line-height: 1; }
.rt-pcard .pct { font-size: 12px; margin-left: 6px; font-weight: 600; }

/* ---------- Beats table ---------- */
.rt-table-wrap { max-height: 330px; overflow-y: auto; border-radius: 10px; border: 1px solid #1f3a57; }
table.rt-table { width: 100%; border-collapse: collapse; font-size: 13px; color: #dbe8f7; }
table.rt-table th {
    position: sticky; top: 0; background: #0f2338; color: #9bb3cc;
    font-weight: 600; text-align: center; padding: 8px 6px; border-bottom: 1px solid #1f3a57;
}
table.rt-table td { text-align: center; padding: 6px; border-bottom: 1px solid #132840; }
table.rt-table tr:hover td { background: rgba(66,224,193,0.05); }
.rt-pill {
    display: inline-block; min-width: 30px; padding: 2px 10px; border-radius: 8px;
    font-weight: 700; font-size: 12px; border: 1px solid;
}
.rt-empty {
    padding: 28px 12px; text-align: center; color: #8fa8c4; font-size: 14px;
    border: 1px dashed #284866; border-radius: 12px;
}

/* ---------- Widgets (readable on dark even without a dark theme) ---------- */
[data-testid="stWidgetLabel"] p, label p { color: #b7cbe2 !important; font-size: 13px; }
div[data-baseweb="select"] > div { background-color: #0d1b2e; border-color: #254460; }
div[data-baseweb="select"] * { color: #edf5ff; }
textarea, input { background-color: #0d1b2e !important; color: #edf5ff !important; }
[data-testid="stCaptionContainer"] { color: #8fa8c4; }

.stButton > button, button[kind="primary"], button[kind="secondary"] {
    background: linear-gradient(90deg, #0a7fd6, #0db5a5);
    color: #fff; border: none; border-radius: 12px; font-weight: 700; min-height: 50px;
}
.stButton > button:hover { filter: brightness(1.1); color: #fff; }
.stButton > button:disabled { opacity: 0.45; }

[data-testid="stAlert"] { border-radius: 12px; }
hr { border-color: #203650; }
</style>
""")


# ---------------------------------------------------------------------
# Navigation state (sidebar radio <-> ?page= query parameter)
# ---------------------------------------------------------------------
query_page = st.query_params.get("page", "Dashboard")
if query_page not in PAGES:
    query_page = "Dashboard"
if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = query_page


def _sync_page():
    st.query_params["page"] = st.session_state["nav_page"]


with st.sidebar:
    md("""
    <div class="rt-side-brand">
        <div class="logo">♥</div>
        <div class="name">RhythmTwin</div>
        <div class="sub">AI-Powered Heart Rhythm<br>Digital Twin</div>
    </div>
    """)
    st.radio(
        "Navigation",
        PAGES,
        key="nav_page",
        on_change=_sync_page,
        format_func=lambda p: f"{PAGE_ICONS[p]}  {p}",
        label_visibility="collapsed",
    )
    md("""
    <div class="rt-disclaimer">
        <b>⚠ Research Prototype</b>
        Experimental research prototype only. Not a diagnostic tool,
        medical device, or emergency detection system.
    </div>
    """)

page = st.session_state["nav_page"]

# Top navbar (single instance)
nav_links = "".join(
    f'<a href="?page={p}" target="_self" class="{"active" if p == page else ""}">{p}</a>'
    for p in PAGES
)
md(f"""
<div class="rt-topnav">
    <div class="brand">♥ RhythmTwin<small>Research prototype · Not for medical diagnosis</small></div>
    <div class="links">{nav_links}</div>
</div>
""")


# ---------------------------------------------------------------------
# Static information pages
# ---------------------------------------------------------------------
def info_card(title: str, body_html: str) -> None:
    md(f"""
    <div class="rt-hero" style="padding:18px 24px;">
        <div class="rt-title">{title}</div>
        <div style="color:#c9d9ec;font-size:14.5px;line-height:1.7;">{body_html}</div>
    </div>
    """)


if page != "Dashboard":
    if page == "About":
        info_card(
            "About RhythmTwin",
            "RhythmTwin is an experimental research prototype that lets you explore "
            "recorded ECG signals from the MIT-BIH Arrhythmia Database and view "
            "experimental AI heartbeat classifications next to the expert annotations. "
            "<br><br><b>Important:</b> it is not a diagnostic tool, medical device, or "
            "emergency detection system. Patient profiles are synthetic and illustrative. "
            "Predictions must not be used for diagnosis, treatment, or urgent decisions.",
        )
        info_card(
            "The five simplified classes",
            "<b>N</b> – Normal-like beat (annotation symbols N, L, R, e, j)<br>"
            "<b>S</b> – Supraventricular ectopic class (A, a, J, S)<br>"
            "<b>V</b> – Ventricular ectopic class (V, E)<br>"
            "<b>F</b> – Fusion beat class (F)<br>"
            "<b>Q</b> – Paced/other class (/, f, Q)",
        )
    elif page == "Methodology":
        info_card(
            "Methodology (as implemented in this app)",
            "1. The selected MIT-BIH record is read with <code>wfdb</code> "
            "(first ECG channel) together with its expert annotations.<br>"
            "2. For each expert-annotated beat in the chosen window, a segment from "
            "0.25 s before to 0.45 s after the annotation is extracted.<br>"
            "3. Each segment is mean-centred, scaled by its standard deviation, and "
            "resampled to 180 points.<br>"
            "4. The saved Random Forest model "
            "(<code>rhythmtwin_model.joblib</code>) predicts one of N, S, V, F, Q.<br>"
            f"5. An exploratory rule labels a beat as V when the model's V-class "
            f"probability is at least {V_THRESHOLD:.2f}.<br>"
            "6. Expert annotations and AI predictions are always displayed separately.",
        )
    else:  # Results
        info_card(
            "Documented results",
            "An exploratory V-class threshold of 0.10 improved recall to 61.1% with "
            "98.3% precision on the held-out evaluation records. This threshold was "
            "selected using those evaluation results, so independent validation is "
            "still needed. No other performance figures are claimed by this app.",
        )
        saved_results = st.session_state.get("ai_results")
        if saved_results is not None:
            counts_saved = saved_results["AI predicted class"].value_counts()
            summary = ", ".join(
                f"{c}: {int(counts_saved.get(c, 0))}" for c in CLASS_ORDER
            )
            info_card(
                "Latest classification run in this session",
                f"Predicted class counts for the last analysed window — {summary}. "
                "These are model outputs, not confirmed conditions.",
            )
        else:
            info_card(
                "Latest classification run in this session",
                "No classification has been run yet. Open the Dashboard and run the "
                "experimental AI classification.",
            )
    st.stop()


# ---------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------
md("""
<div class="rt-hero">
    <div class="eyebrow">AI-POWERED CARDIAC MONITORING</div>
    <h1>Understand Heart Rhythms.<br><span>Support Better Decisions.</span></h1>
    <p>Explore recorded ECG signals and experimental AI heartbeat classifications.</p>
    <div class="heart">♥</div>
    <svg viewBox="0 0 400 100" preserveAspectRatio="none" fill="none">
        <polyline points="0,55 70,55 85,55 95,15 108,92 120,55 200,55 215,55 225,30 238,72
                          250,55 320,55 335,55 345,10 358,95 370,55 400,55"
                  stroke="#22d3ee" stroke-width="2" stroke-linejoin="round"/>
    </svg>
</div>
""")


# ---------------------------------------------------------------------
# Load the saved model (unchanged logic, cached)
# ---------------------------------------------------------------------
model = None
feature_cols = None
model_error = None

if not MODEL_PATH.exists():
    model_error = "Model file not found: rhythmtwin_model.joblib"
else:
    try:
        saved = load_saved_model(str(MODEL_PATH))
        model = saved["model"]
        feature_cols = saved["feature_cols"]
    except Exception as e:
        model_error = f"Could not load the saved model: {e}"


# ---------------------------------------------------------------------
# Control row: record | window / start | run button | patient profile
# ---------------------------------------------------------------------
c_rec, c_time, c_run, c_pat = st.columns([1.1, 1.7, 1.1, 1.3], gap="medium")

with c_rec:
    with st.container(border=True):
        md('<div class="rt-title"><span class="ic">♥</span>ECG Record</div>')
        record_id = st.selectbox(
            "MIT-BIH ECG record",
            AVAILABLE_RECORDS,
            index=0,
        )

with c_pat:
    with st.container(border=True):
        md('<div class="rt-title"><span class="ic">👤</span>Patient Profile</div>')
        patient_id = st.selectbox(
            "Select illustrative profile",
            ["Patient RT-001", "Patient RT-002"],
        )
        if patient_id == "Patient RT-001":
            age = 58
            history = "Synthetic demo history"
        else:
            age = 42
            history = "Synthetic demo history"
        md(f"""
        <div class="rt-profile"><b>Profile:</b> {patient_id} &nbsp;·&nbsp;
        <b>Age:</b> {age} years<br><b>History:</b> {history}</div>
        """)
        st.caption("Illustrative profile only — not linked to a real patient.")

# --- ECG loading (unchanged logic) -----------------------------------
required_files = [
    f"{record_id}.hea",
    f"{record_id}.dat",
    f"{record_id}.atr",
]
missing = [
    name for name in required_files
    if not (DATA_DIR / name).exists()
]
if missing:
    st.error("Missing ECG files: " + ", ".join(missing))
    st.stop()

try:
    record = wfdb.rdrecord(str(DATA_DIR / record_id))
    annotations = wfdb.rdann(str(DATA_DIR / record_id), "atr")
    ecg = record.p_signal[:, 0]
    fs = record.fs
    duration = len(ecg) / fs
except Exception as e:
    st.error("Could not load the ECG recording.")
    st.code(str(e))
    st.stop()

with c_time:
    with st.container(border=True):
        md('<div class="rt-title"><span class="ic">⏱</span>ECG Window</div>')
        t1, t2 = st.columns(2)
        with t1:
            seconds = st.slider(
                "Window length (seconds)",
                min_value=5,
                max_value=30,
                value=10,
                step=5,
            )
        max_start = max(0, int(duration) - seconds)
        with t2:
            if max_start > 0:
                start_sec = st.slider(
                    "Start time (seconds)",
                    min_value=0,
                    max_value=max_start,
                    value=0,
                )
            else:
                start_sec = 0
                st.caption("Recording is shorter than the window.")

with c_run:
    with st.container(border=True):
        md('<div class="rt-title"><span class="ic">▶</span>AI Classification</div>')
        run_clicked = st.button(
            "▶  Run Experimental AI Classification",
            type="primary",
            use_container_width=True,
            disabled=(model is None),
        )
        if model_error:
            st.caption(model_error)
        else:
            st.caption("Runs on the beats annotated in the selected window.")

# --- Window selection (unchanged logic) ------------------------------
start = int(start_sec * fs)
end = min(len(ecg), int((start_sec + seconds) * fs))
times = np.arange(start, end) / fs

visible_indices = [
    int(i)
    for i, sample in enumerate(annotations.sample)
    if start <= sample < end
]
visible_samples = [int(annotations.sample[i]) for i in visible_indices]

visible_symbols = [
    CLASS_MAP[annotations.symbol[i]]
    for i in visible_indices
    if annotations.symbol[i] in CLASS_MAP
]
# Expert class per visible annotation ("—" if the symbol is not a beat class).
expert_by_sample = {
    int(annotations.sample[i]): CLASS_MAP.get(annotations.symbol[i], "—")
    for i in visible_indices
}

# --- AI classification (prediction logic unchanged) ------------------
progress_slot = st.empty()

if run_clicked and model is not None:
    try:
        predictions = []
        prediction_times = []
        prediction_samples = []
        prediction_conf = []
        skipped = 0

        before = int(0.25 * fs)
        after = int(0.45 * fs)

        progress = progress_slot.progress(0)

        for count, sample in enumerate(visible_samples):
            beat_start = sample - before
            beat_end = sample + after

            if beat_start < 0 or beat_end > len(ecg):
                skipped += 1
                continue

            beat = np.asarray(ecg[beat_start:beat_end], dtype=float)

            if not np.all(np.isfinite(beat)):
                skipped += 1
                continue

            beat = beat - np.mean(beat)
            scale = np.std(beat)

            if scale < 1e-8:
                skipped += 1
                continue

            beat = beat / scale
            beat = resample(beat, 180)

            row = pd.DataFrame(
                [beat],
                columns=[f"x{i}" for i in range(180)]
            )
            row = row[feature_cols]

            probabilities = model.predict_proba(row)[0]
            predicted_class = str(model.predict(row)[0])

            # Experimental V-beat detection threshold
            v_index = list(model.classes_).index("V")
            v_score = float(probabilities[v_index])
            if v_score >= V_THRESHOLD:
                predicted_class = "V"

            # Confidence = model probability of the class finally shown.
            class_index = list(model.classes_).index(predicted_class)
            confidence = float(probabilities[class_index]) * 100

            predictions.append(predicted_class)
            prediction_times.append(sample / fs)
            prediction_samples.append(sample)
            prediction_conf.append(confidence)

            progress.progress(
                min(1.0, (count + 1) / len(visible_samples))
            )

        progress_slot.empty()

        if not predictions:
            st.info(
                "No eligible beats found in this window. "
                "Choose another start time."
            )
        else:
            st.session_state["ai_results"] = pd.DataFrame({
                "Time (seconds)": prediction_times,
                "Sample": prediction_samples,
                "Expert label": [
                    expert_by_sample.get(s, "—") for s in prediction_samples
                ],
                "AI predicted class": predictions,
                "Confidence (%)": prediction_conf,
            })
            st.session_state["ai_window"] = (record_id, start_sec, seconds)
            st.session_state["ai_skipped"] = skipped
    except Exception as e:
        progress_slot.empty()
        st.error("Could not run the saved model.")
        st.code(str(e))

current_window = (record_id, start_sec, seconds)
results = st.session_state.get("ai_results")
if results is not None and st.session_state.get("ai_window") != current_window:
    results = None  # results belong to a different record / window

# ---------------------------------------------------------------------
# Main row: ECG viewer (left) + prediction cards (right)
# ---------------------------------------------------------------------
PLOT_BASE = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(7,17,31,0.55)",
    font={"color": "#c9d9ec"},
)

left, right = st.columns([2.1, 1], gap="medium")

with left:
    with st.container(border=True):
        md('<div class="rt-title"><span class="ic">〰</span>ECG Signal Visualization</div>')
        md(f"""
        <div class="rt-chips">
            <span class="rt-chip">MIT-BIH record {record_id}</span>
            <span class="rt-chip">{fs:g} Hz</span>
            <span class="rt-chip">{duration / 60:.1f} min</span>
            <span class="rt-chip">{len(annotations.sample):,} expert annotations</span>
        </div>
        """)

        y_seg = ecg[start:end]
        y_min, y_max = float(np.min(y_seg)), float(np.max(y_seg))
        span = max(y_max - y_min, 1e-6)
        ai_offset = 0.14 * span  # AI markers sit slightly above the trace

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times,
            y=y_seg,
            mode="lines",
            name="ECG signal",
            line={"width": 1.2, "color": "#e8f1fb"},
        ))

        if visible_samples:
            fig.add_trace(go.Scatter(
                x=[s / fs for s in visible_samples],
                y=[ecg[s] for s in visible_samples],
                mode="markers",
                name="Expert label",
                marker={
                    "symbol": "circle-open", "size": 11,
                    "color": "#4ade80", "line": {"width": 2},
                },
                text=[expert_by_sample.get(s, "—") for s in visible_samples],
                hovertemplate="Expert: %{text}<br>t = %{x:.3f} s<extra></extra>",
            ))

        if results is not None:
            for class_code in CLASS_ORDER:
                subset = results[results["AI predicted class"] == class_code]
                if subset.empty:
                    continue
                fig.add_trace(go.Scatter(
                    x=subset["Time (seconds)"],
                    y=[float(ecg[int(s)]) + ai_offset for s in subset["Sample"]],
                    mode="markers",
                    name=f"AI: {class_code}",
                    marker={
                        "color": CARD_INFO[class_code][1],
                        "size": 9,
                        "symbol": "diamond",
                        "line": {"width": 1, "color": "#07111f"},
                    },
                    text=[f"{c:.1f}%" for c in subset["Confidence (%)"]],
                    hovertemplate=(
                        f"AI: {class_code}<br>t = %{{x:.3f}} s"
                        "<br>class probability %{text}<extra></extra>"
                    ),
                ))

        y_top = y_max + (0.35 if results is not None else 0.15) * span
        fig.update_layout(
            **PLOT_BASE,
            xaxis_title="Time (seconds)",
            yaxis_title="Amplitude (mV)",
            height=430,
            margin={"l": 10, "r": 10, "t": 30, "b": 10},
            legend={"orientation": "h", "y": 1.1, "x": 1, "xanchor": "right"},
            xaxis={"gridcolor": "#15304a"},
            yaxis={"gridcolor": "#15304a", "range": [y_min - 0.1 * span, y_top]},
        )
        st.plotly_chart(fig, use_container_width=True)

        expert_counts = pd.Series(visible_symbols).value_counts().to_dict()
        counts_text = ", ".join(f"{k}: {v}" for k, v in expert_counts.items()) or "none"
        st.caption(
            f"Expert labels in window — {counts_text}. Green rings are expert "
            "annotations, not AI predictions. AI markers (diamonds) are drawn "
            "slightly above the trace so they do not hide the signal."
        )

with right:
    with st.container(border=True):
        md('<div class="rt-title"><span class="ic">🧠</span>AI Prediction Summary</div>')

        if results is not None:
            counts = results["AI predicted class"].value_counts()
            total = len(results)
        else:
            counts = None
            total = 0

        cells = []
        for code in CLASS_ORDER:
            label, color = CARD_INFO[code]
            if counts is None:
                value, pct = "—", ""
            else:
                n = int(counts.get(code, 0))
                value = str(n)
                pct = f"({n / total * 100:.1f}%)" if total else ""
            wide = " wide" if code == "Q" else ""
            cells.append(f"""
            <div class="rt-pcard{wide}" style="background:linear-gradient(145deg,{color}33,{color}0d);
                 border-color:{color}88;">
                <div class="code" style="color:{color};">{code}</div>
                <div class="lbl">{label}</div>
                <div class="num">{value}<span class="pct" style="color:{color};">{pct}</span></div>
            </div>
            """)
        md(f'<div class="rt-grid">{"".join(cells)}</div>')

        if results is None:
            st.caption("Run the experimental classification to fill these cards.")
        else:
            st.caption(
                f"Skipped beats: {st.session_state.get('ai_skipped', 0)}. "
                "Counts describe model outputs, not confirmed conditions."
            )

st.warning(
    "Experimental research output only. An exploratory V-class "
    "threshold of 0.10 improved recall to 61.1% with 98.3% precision "
    "on the held-out evaluation records. This threshold was selected "
    "using those evaluation results, so independent validation is "
    "still needed. Predictions must not be used for diagnosis, "
    "treatment, or urgent decisions."
)

# ---------------------------------------------------------------------
# Lower row: detected beats table + prediction distribution
# ---------------------------------------------------------------------
def pill(code: str) -> str:
    color = CARD_INFO.get(code, ("", "#8fa8c4"))[1]
    return (
        f'<span class="rt-pill" style="background:{color}22;color:{color};'
        f'border-color:{color}66;">{code}</span>'
    )


bottom_l, bottom_r = st.columns([1.6, 1], gap="medium")

with bottom_l:
    with st.container(border=True):
        md('<div class="rt-title"><span class="ic">☰</span>Detected Beats</div>')
        if results is None:
            md('<div class="rt-empty">No AI results for this window yet. '
               'Press <b>Run Experimental AI Classification</b>.</div>')
        else:
            body_rows = []
            for i, r in enumerate(results.itertuples(index=False), start=1):
                body_rows.append(
                    f"<tr><td>{i}</td>"
                    f"<td>{r[0]:.3f}</td>"
                    f"<td>{pill(r[2])}</td>"
                    f"<td>{pill(r[3])}</td>"
                    f"<td>{r[4]:.1f}%</td></tr>"
                )
            md(f"""
            <div class="rt-table-wrap"><table class="rt-table">
            <thead><tr><th>#</th><th>Time (s)</th><th>Expert label</th>
            <th>AI prediction</th><th>Confidence</th></tr></thead>
            <tbody>{"".join(body_rows)}</tbody></table></div>
            """)

            comparable = results[results["Expert label"] != "—"]
            if len(comparable):
                same = int((comparable["Expert label"]
                            == comparable["AI predicted class"]).sum())
                st.caption(
                    f"AI class equals the mapped expert class for {same} of "
                    f"{len(comparable)} beats in this window (descriptive only, "
                    "not a validation metric). Confidence is the model's "
                    "probability for the class shown."
                )

with bottom_r:
    with st.container(border=True):
        md('<div class="rt-title"><span class="ic">◔</span>Prediction Distribution</div>')
        if results is None:
            md('<div class="rt-empty">The distribution appears after a classification run.</div>')
        else:
            present = [c for c in CLASS_ORDER if int(counts.get(c, 0)) > 0]
            donut = go.Figure(go.Pie(
                labels=[f"{c} · {CARD_INFO[c][0]}" for c in present],
                values=[int(counts.get(c, 0)) for c in present],
                hole=0.68,
                marker={"colors": [CARD_INFO[c][1] for c in present],
                        "line": {"color": "#07111f", "width": 2}},
                textinfo="none",
                sort=False,
            ))
            donut.update_layout(
                **PLOT_BASE,
                height=300,
                margin={"l": 10, "r": 10, "t": 10, "b": 10},
                legend={"font": {"size": 12}},
                annotations=[{
                    "text": f"<b>{total}</b><br>Beats",
                    "x": 0.5, "y": 0.5, "showarrow": False,
                    "font": {"size": 22, "color": "#f1f7ff"},
                }],
            )
            st.plotly_chart(donut, use_container_width=True)

st.caption(
    "These simplified categories are research labels. They are not diagnoses, "
    "and this model is not validated for clinical use. The model was trained on "
    "separate MIT-BIH records; displayed results are predictions, not expert labels."
)

# ---------------------------------------------------------------------
# Clinician review (unchanged logic)
# ---------------------------------------------------------------------
with st.container(border=True):
    md('<div class="rt-title"><span class="ic">📝</span>Clinician Review</div>')
    r1, r2 = st.columns([2, 1], gap="medium")
    with r1:
        note = st.text_area("Review note")
    with r2:
        status = st.selectbox(
            "Review status",
            ["Not reviewed", "Reviewed", "Needs follow-up"]
        )
        save_clicked = st.button("Save review", use_container_width=True)

    if save_clicked:
        st.session_state["review"] = {
            "patient": patient_id,
            "note": note,
            "status": status,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        st.success("Review saved for this session.")

    if "review" in st.session_state:
        st.write("**Latest review**")
        st.json(st.session_state["review"])

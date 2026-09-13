"""
streamlit_app.py
------------------
Zynex offline demo UI. Accepts a traffic CSV (synthetic schema, or a real
CIC-IDS-2018/CTU-13 export), runs the full pipeline (feature extraction ->
trained world model -> K-step forward simulation), and displays:
    - the infiltration-probability timeline,
    - flagged flows / windows,
    - MITRE ATT&CK stage annotations,
    - the top contributing features for the current forecast.

Runs fully offline -- no external API calls.

Usage:
    streamlit run app/streamlit_app.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from feature_extraction import build_state_vectors, build_sequences, normalise, FEATURE_NAMES, SEQ_LEN, FORECAST_K, LOADERS
from predict_engine import load_model, rollout

st.set_page_config(page_title="Zynex — Predictive Cyber Defence", layout="wide")

st.title("🛰️ Zynex — AI-based Network Attack Forecasting")
st.caption("SIH 2026 · PS #26153 · World-Model-based predictive cyber defence (fully offline)")

with st.sidebar:
    st.header("Configuration")
    model_path = st.text_input("World model checkpoint", value="models/world_model.pt")
    k_steps = st.slider("Forward simulation horizon (K windows)", 1, 20, value=FORECAST_K)
    dataset_type = st.selectbox(
        "Dataset schema",
        options=["auto-detect", "synthetic", "cic-ids-2018", "cic-ids-2017", "ctu-13"],
        index=0,
        help="Leave on auto-detect unless it guesses wrong. Real CIC-IDS-2018/2017/CTU-13 "
             "exports need their column names remapped before Zynex can read them -- "
             "'auto-detect' does this automatically based on the uploaded file's columns.",
    )
    uploaded = st.file_uploader("Upload traffic CSV (synthetic schema or CIC-IDS-2018 export)", type=["csv"])
    run_btn = st.button("Run Zynex analysis", type="primary")


def _detect_schema(raw_df: pd.DataFrame) -> str:
    """Guesses which loader to use based on the uploaded CSV's column names."""
    cols = set(c.strip() for c in raw_df.columns)
    if {"timestamp", "syn", "ack", "bytes", "packets", "dst_port"}.issubset({c.lower() for c in cols}):
        return "synthetic"
    # any recognisable CICFlowMeter column (long-form or short-form) -> treat as CIC-IDS export
    cic_markers = {"Dst Port", "Destination Port", "Flow Duration", "Tot Fwd Pkts",
                   "Total Fwd Packets", "SYN Flag Cnt", "SYN Flag Count", "Label"}
    if cols & cic_markers:
        return "cic-ids-2018"
    return "synthetic"  # fall back; build_state_vectors will error clearly if columns are still missing


if run_btn:
    if uploaded is None:
        st.warning("Please upload a traffic CSV first.")
        st.stop()
    if not os.path.exists(model_path):
        st.error(f"Model checkpoint not found at `{model_path}`. Train it first with `python src/train.py`.")
        st.stop()

    with st.spinner("Extracting flow + packet features and building state sequences..."):
        raw_df = pd.read_csv(uploaded, low_memory=False)

        schema = dataset_type if dataset_type != "auto-detect" else _detect_schema(raw_df)
        st.caption(f"Detected/selected dataset schema: **{schema}**")

        if schema == "synthetic":
            df = raw_df
            if "stage" not in df.columns:
                df["stage"] = df.get("label", "Benign")
        else:
            # re-save the uploaded file to a temp path so the existing file-path-based
            # loader (LOADERS[...]) can read it with the correct CIC-IDS/CTU-13 adapter
            tmp_path = os.path.join("data", "_uploaded_tmp.csv")
            os.makedirs("data", exist_ok=True)
            raw_df.to_csv(tmp_path, index=False)
            df = LOADERS[schema](tmp_path)

        X, stage_labels = build_state_vectors(df)
        seq_X, next_state_y, infil_y, stage_y = build_sequences(X, stage_labels, seq_len=SEQ_LEN, k=k_steps)

    if len(seq_X) == 0:
        st.error("Not enough time windows in the uploaded file to build a sequence "
                  f"(need at least {SEQ_LEN + 1} one-second windows).")
        st.stop()

    model, mean, std, feature_names = load_model(model_path)
    seq_X_n = (seq_X - mean) / std

    with st.spinner("Running K-step forward simulation..."):
        forecast = rollout(model, seq_X_n[-1], mean, std, feature_names, k=k_steps)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Infiltration probability — forward simulation")
        steps = [r["step"] for r in forecast]
        probs = [r["infiltration_probability"] for r in forecast]
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(steps, probs, marker="o", color="#d62728")
        ax.axhline(0.5, linestyle="--", color="gray", linewidth=1)
        ax.set_xlabel("Windows ahead (t+k)")
        ax.set_ylabel("P(infiltration)")
        ax.set_ylim(0, 1)
        ax.set_title("Forecast infiltration probability timeline")
        st.pyplot(fig)

        st.subheader("Predicted MITRE ATT&CK stage progression")
        stage_df = pd.DataFrame([{"t+k": r["step"], "Stage": r["predicted_stage"],
                                   "Confidence": r["stage_confidence"]} for r in forecast])
        st.dataframe(stage_df, use_container_width=True)

    with col2:
        st.subheader("Current risk")
        latest = forecast[0]
        st.metric("Infiltration probability (t+1)", f"{latest['infiltration_probability']:.1%}")
        st.metric("Predicted stage", latest["predicted_stage"])
        st.caption(latest["rationale"])

        st.subheader("Top contributing features")
        for f in latest["top_contributing_features"]:
            st.write(f"**{f['feature']}** — importance {f['importance']:.3f}")

    st.subheader("Recent flagged windows (raw features)")
    n_recent = min(50, len(X))
    flagged_df = pd.DataFrame(X[-n_recent:], columns=FEATURE_NAMES)
    flagged_df["is_malicious_stage"] = [
        s != 0 for s in stage_labels[-n_recent:]
    ]
    st.dataframe(flagged_df, use_container_width=True)

else:
    st.info("Upload a traffic CSV in the sidebar and click **Run Zynex analysis** to begin. "
            "No dataset handy? Generate one offline with:\n\n"
            "`python src/synthetic_data.py --out data/synthetic_traffic.csv`")
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title="Neural Labs", layout="wide")
st.title("Neural Network Training Labs — Panel de ingeniería")
st.caption("Experimentos, modelos registrados, calidad, deriva y costo de inferencia.")

runs = []
for metrics_path in sorted((ROOT / "runs").glob("*/*/metrics.json"), reverse=True):
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        runs.append({"lab": metrics_path.parents[1].name, "run": metrics_path.parent.name, **metrics})
    except Exception:
        continue

col1, col2, col3 = st.columns(3)
col1.metric("Ejecuciones", len(runs))
registry_path = ROOT / "model-registry.json"
registry = json.loads(registry_path.read_text()) if registry_path.is_file() else {"models": {}}
col2.metric("Modelos registrados", len(registry.get("models", {})))
col3.metric("Laboratorios", len(list((ROOT / "labs").glob("[0-9][0-9]_*"))) + len(list((ROOT / "advanced_labs").glob("[0-9][0-9]_*"))))

st.subheader("Resultados")
if runs:
    frame = pd.json_normalize(runs)
    st.dataframe(frame, use_container_width=True)
    numeric = [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
    if numeric:
        metric = st.selectbox("Métrica", numeric)
        chart = frame[["lab", metric]].dropna().set_index("lab")
        st.bar_chart(chart)
else:
    st.info("Ejecute un laboratorio para poblar el panel.")

st.subheader("Registro de modelos")
st.json(registry)
st.subheader("Operación")
st.code("streamlit run dashboard/app.py\nneural-labs serve\nneural-labs leaderboard")

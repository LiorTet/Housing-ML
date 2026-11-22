"""Minimal UI to test the served model. Uses the public API only."""

import os

import requests
import streamlit as st

from housing.features import CONTRACT

API_URL = os.getenv("API_URL", "http://localhost:8000")
EXAMPLE = {
    "MedInc": 3.87,
    "HouseAge": 29.0,
    "AveRooms": 5.43,
    "AveBedrms": 1.1,
    "Population": 1166.0,
    "AveOccup": 3.07,
    "Latitude": 35.63,
    "Longitude": -119.57,
}

st.title("Housing Value")

try:
    info = requests.get(f"{API_URL}/model", timeout=5).json()
    st.caption(
        f"Model v{info['version']} · {info['tags'].get('family')} · "
        f"gold {info['tags'].get('gold_version')}"
    )
except requests.RequestException:
    st.error(f"API not reachable at {API_URL}")
    st.stop()

with st.form("predict"):
    cols = st.columns(2)
    values = {
        name: cols[i % 2].number_input(name, low, high, EXAMPLE[name], help=desc, format="%.4f")
        for i, (name, (low, high, desc)) in enumerate(CONTRACT.items())
    }
    submitted = st.form_submit_button("Predict")

if submitted:
    resp = requests.post(f"{API_URL}/predict", json={"instances": [values]}, timeout=5)
    if resp.ok:
        body = resp.json()
        st.metric("Median house value", f"${body['predictions'][0] * 100_000:,.0f}")
        st.caption(f"model v{body['model_version']} · request {body['request_id']}")
    else:
        st.error(resp.text)

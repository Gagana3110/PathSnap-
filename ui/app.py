"""PERSON 4 - Evaluator UI (runs on LAPTOP 2).

Start:
    streamlit run ui/app.py
Enter Laptop 1's API address (e.g. http://192.168.1.10:8000), pick a trip, press Run.
The UI only talks to the API - it has no model and no ground truth.
"""
import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(page_title="PathSnap", layout="wide")
st.title("PathSnap - Map-Matching & Route Reconstruction")

api = st.sidebar.text_input("API URL (Laptop 1)", value="http://127.0.0.1:8000").rstrip("/")
use_hmm = st.sidebar.checkbox("Use HMM / Viterbi", value=True)

try:
    trips = requests.get(f"{api}/trips", timeout=10).json()["trips"]
    st.sidebar.success(f"Connected - {len(trips)} evaluation trips")
except Exception as e:
    st.sidebar.error(f"Cannot reach API: {e}")
    st.stop()

trip = st.selectbox("Select evaluation trip", trips)
if st.button("Run prediction", type="primary"):
    with st.spinner("Sending trip to API..."):
        r = requests.post(f"{api}/predict", json={"trip_id": trip, "use_hmm": use_hmm}, timeout=600)
    if r.status_code != 200:
        st.error(r.text)
        st.stop()
    st.session_state["res"] = r.json()

res = st.session_state.get("res")
if res:
    m = res["metrics"]
    main = m.get("hmm" if use_hmm and "hmm" in m else "clf", {})
    c = st.columns(5)
    c[0].metric("Top-1 accuracy", f"{main.get('top1_accuracy', 0):.2%}")
    c[1].metric("Route continuity", f"{main.get('route_continuity', 0):.2%}")
    c[2].metric("Classifier-only Top-1", f"{m.get('clf', {}).get('top1_accuracy', 0):.2%}")
    c[3].metric("GPS points", res["n_points"])
    c[4].metric("API time", f"{res['processing_time_s']:.2f} s")
    st.caption(f"Route precision {m.get('route_precision', 0):.2%} | route recall {m.get('route_recall', 0):.2%} | "
               f"breaks: classifier {m.get('clf', {}).get('route_breaks')} vs HMM {m.get('hmm', {}).get('route_breaks')}")

    gps = res["gps"]
    fm = folium.Map(location=gps[len(gps) // 2], zoom_start=14, tiles="OpenStreetMap")
    g1 = folium.FeatureGroup("Raw GPS")
    for p in gps[:: max(1, len(gps) // 1500)]:
        folium.CircleMarker(p, radius=2, color="#1f77b4", fill=True, weight=0, fill_opacity=.8).add_to(g1)
    g1.add_to(fm)
    g2 = folium.FeatureGroup("Predicted route")
    folium.PolyLine(res["predicted_route"], color="#d62728", weight=4).add_to(g2)
    g2.add_to(fm)
    folium.Marker(gps[0], tooltip="Start", icon=folium.Icon(color="green")).add_to(fm)
    folium.Marker(gps[-1], tooltip="End", icon=folium.Icon(color="red")).add_to(fm)
    lats, lons = [p[0] for p in gps], [p[1] for p in gps]
    fm.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    folium.LayerControl(collapsed=False).add_to(fm)
    st_folium(fm, height=600, use_container_width=True)
    with st.expander("Raw API metrics"):
        st.json(m)

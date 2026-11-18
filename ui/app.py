
"""PathSnap - Map Matching & Route Reconstruction Dashboard."""

import requests
import streamlit as st
import folium
from streamlit_folium import st_folium

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="PathSnap | Route Intelligence",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# CUSTOM LIGHT THEME
# ---------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: #F5F7FB;
        color: #172033;
    }

    [data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid #E7ECF3;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #536179;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    .brand {
        font-size: 14px;
        font-weight: 800;
        letter-spacing: 2px;
        color: #3478F6;
        margin-bottom: 8px;
    }

    .hero-title {
        font-size: 34px;
        font-weight: 800;
        color: #172033;
        letter-spacing: -1.2px;
        margin-bottom: 4px;
    }

    .hero-subtitle {
        color: #758198;
        font-size: 15px;
        margin-bottom: 28px;
    }

    .section-title {
        color: #172033;
        font-size: 19px;
        font-weight: 700;
        margin: 20px 0 12px 0;
    }

    .info-card {
        background: #FFFFFF;
        border: 1px solid #E7ECF3;
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 12px;
    }

    .info-label {
        color: #8490A5;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.7px;
    }

    .info-value {
        color: #172033;
        font-size: 25px;
        font-weight: 800;
        margin-top: 5px;
    }

    .info-caption {
        color: #8490A5;
        font-size: 12px;
        margin-top: 3px;
    }

    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E7ECF3;
        border-radius: 14px;
        padding: 16px 18px;
    }

    div[data-testid="stMetricLabel"] {
        color: #758198;
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        color: #172033;
        font-weight: 800;
    }

    div.stButton > button[kind="primary"] {
        background: #3478F6;
        border: none;
        border-radius: 10px;
        color: white;
        font-weight: 700;
        padding: 0.65rem 1.2rem;
    }

    div.stButton > button[kind="primary"]:hover {
        background: #2563D9;
        border: none;
    }

    div.stButton > button {
        border-radius: 9px;
    }

    [data-testid="stSelectbox"] > div > div {
        border-radius: 9px;
    }

    .status-pill {
        display: inline-block;
        padding: 5px 11px;
        border-radius: 20px;
        background: #E8F7EF;
        color: #218653;
        font-size: 12px;
        font-weight: 700;
    }

    hr {
        border-color: #E7ECF3;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 8px 0 22px 0;">
            <div style="font-size:28px;">📍</div>
            <div style="font-size:22px;font-weight:800;color:#172033;">
                PathSnap
            </div>
            <div style="font-size:12px;color:#8490A5;">
                ROUTE INTELLIGENCE PLATFORM
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("#### CONNECTION")

    api = st.text_input(
        "Backend API URL",
        value="http://127.0.0.1:8000",
        help="Enter the backend laptop's IP address when connecting over Wi-Fi.",
    ).strip().rstrip("/")

    st.markdown("---")
    st.markdown("#### MODEL SETTINGS")

    use_hmm = st.toggle("Use HMM / Viterbi", value=True)

    st.caption(
        "Enable HMM to apply sequence-based route reconstruction "
        "after classifier predictions."
    )

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size:12px;color:#8490A5;line-height:1.8;">
            <b style="color:#536179;">PROJECT</b><br>
            Map Matching & Route Reconstruction<br><br>
            <b style="color:#536179;">INTERFACE</b><br>
            Evaluator Dashboard
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown('<div class="brand">PATHSNAP / DASHBOARD</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-title">Route Intelligence</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-subtitle">'
    'Analyze GPS trajectories and visualize reconstructed routes.'
    '</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# API CONNECTION
# ---------------------------------------------------------

try:
    response = requests.get(f"{api}/trips", timeout=10)
    response.raise_for_status()
    trips = response.json()["trips"]

    st.markdown(
        '<span class="status-pill">● Backend connected</span>',
        unsafe_allow_html=True,
    )
    st.caption(f"{len(trips)} evaluation trips available")

except requests.RequestException as exc:
    st.error("Backend connection unavailable")
    st.info(
        "Check that the FastAPI backend is running and that the API URL "
        "is correct. If using two laptops, both devices must be on the "
        "same Wi-Fi or hotspot."
    )
    st.code(f"Connection details: {exc}")
    st.stop()

except (KeyError, ValueError):
    st.error("The backend response did not contain the expected trip list.")
    st.stop()

if not trips:
    st.warning("No evaluation trips were returned by the backend.")
    st.stop()

# ---------------------------------------------------------
# TRIP SELECTION
# ---------------------------------------------------------

st.markdown('<div class="section-title">Prediction workspace</div>', unsafe_allow_html=True)

selection_col, action_col = st.columns([3, 1])

with selection_col:
    trip = st.selectbox(
        "Select an evaluation trip",
        trips,
        help="Choose a trip to run route reconstruction on.",
    )

with action_col:
    st.write("")
    st.write("")
    run_prediction = st.button(
        "Run prediction  →",
        type="primary",
        use_container_width=True,
    )

if run_prediction:
    try:
        with st.spinner("Analyzing GPS trajectory and reconstructing route..."):
            result_response = requests.post(
                f"{api}/predict",
                json={"trip_id": trip, "use_hmm": use_hmm},
                timeout=600,
            )

        if result_response.status_code != 200:
            st.error(f"Prediction failed: {result_response.text}")
        else:
            st.session_state["res"] = result_response.json()
            st.session_state["result_mode"] = use_hmm
            st.session_state["result_trip"] = trip
            st.success("Prediction completed successfully!")

    except requests.RequestException as exc:
        st.error(f"Could not complete prediction: {exc}")

# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------

res = st.session_state.get("res")

if res:
    metrics = res.get("metrics", {})
    result_mode = st.session_state.get("result_mode", use_hmm)

    main_metrics = metrics.get(
        "hmm" if result_mode and "hmm" in metrics else "clf",
        {},
    )
    classifier_metrics = metrics.get("clf", {})
    hmm_metrics = metrics.get("hmm", {})

    st.markdown("---")
    st.markdown('<div class="section-title">Performance overview</div>', unsafe_allow_html=True)

    metric_cols = st.columns(4)

    metric_cols[0].metric(
        "Top-1 accuracy",
        f"{main_metrics.get('top1_accuracy', 0):.2%}",
    )
    metric_cols[1].metric(
        "Route continuity",
        f"{main_metrics.get('route_continuity', 0):.2%}",
    )
    metric_cols[2].metric(
        "GPS points",
        f"{res.get('n_points', 0):,}",
    )
    metric_cols[3].metric(
        "Processing time",
        f"{res.get('processing_time_s', 0):.2f} s",
    )

    st.markdown('<div class="section-title">Route comparison</div>', unsafe_allow_html=True)

    comparison_cols = st.columns(3)

    comparison_cols[0].metric(
        "Classifier-only accuracy",
        f"{classifier_metrics.get('top1_accuracy', 0):.2%}",
    )
    comparison_cols[1].metric(
        "HMM accuracy",
        f"{hmm_metrics.get('top1_accuracy', 0):.2%}",
    )
    comparison_cols[2].metric(
        "Route precision",
        f"{metrics.get('route_precision', 0):.2%}",
    )

    st.caption(
        f"Route recall: {metrics.get('route_recall', 0):.2%}  |  "
        f"Classifier route breaks: {classifier_metrics.get('route_breaks', 'N/A')}  |  "
        f"HMM route breaks: {hmm_metrics.get('route_breaks', 'N/A')}"
    )

    # -----------------------------------------------------
    # MAP
    # -----------------------------------------------------

    st.markdown('<div class="section-title">Reconstructed route</div>', unsafe_allow_html=True)

    gps = res.get("gps", [])
    predicted_route = res.get("predicted_route", [])

    if gps and predicted_route:
        midpoint = gps[len(gps) // 2]

        route_map = folium.Map(
            location=midpoint,
            zoom_start=14,
            tiles="CartoDB positron",
            control_scale=True,
        )

        raw_layer = folium.FeatureGroup(name="Raw GPS trajectory")

        for point in gps[::max(1, len(gps) // 1500)]:
            folium.CircleMarker(
                location=point,
                radius=2,
                color="#3478F6",
                fill=True,
                fill_color="#3478F6",
                fill_opacity=0.75,
                weight=0,
                tooltip="GPS point",
            ).add_to(raw_layer)

        raw_layer.add_to(route_map)

        predicted_layer = folium.FeatureGroup(name="Predicted route")

        folium.PolyLine(
            predicted_route,
            color="#E45B68",
            weight=5,
            opacity=0.9,
            tooltip="Predicted route",
        ).add_to(predicted_layer)

        predicted_layer.add_to(route_map)

        folium.Marker(
            gps[0],
            tooltip="Start",
            icon=folium.Icon(color="green", icon="play"),
        ).add_to(route_map)

        folium.Marker(
            gps[-1],
            tooltip="End",
            icon=folium.Icon(color="red", icon="stop"),
        ).add_to(route_map)

        latitudes = [point[0] for point in gps]
        longitudes = [point[1] for point in gps]

        route_map.fit_bounds(
            [
                [min(latitudes), min(longitudes)],
                [max(latitudes), max(longitudes)],
            ]
        )

        folium.LayerControl(collapsed=False).add_to(route_map)

        st_folium(
            route_map,
            height=540,
            use_container_width=True,
            returned_objects=[],
        )
    else:
        st.warning("The API returned no GPS points or predicted route to display.")

    # -----------------------------------------------------
    # DETAILS
    # -----------------------------------------------------

    with st.expander("View detailed API metrics"):
        st.json(metrics)

    with st.expander("View prediction response"):
        st.json(res)

else:
    st.markdown("---")
    st.markdown(
        """
        <div style="
            background:#FFFFFF;
            border:1px solid #E7ECF3;
            border-radius:16px;
            padding:38px 24px;
            text-align:center;
        ">
            <div style="font-size:38px;">🗺️</div>
            <div style="font-size:20px;font-weight:800;color:#172033;margin-top:10px;">
                Ready to reconstruct a route?
            </div>
            <div style="font-size:14px;color:#8490A5;margin-top:8px;">
                Select an evaluation trip above and run a prediction
                to see the route visualization and performance metrics.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")
st.markdown(
    """
    <div style="text-align:center;color:#A0AABC;font-size:12px;padding:8px;">
        PathSnap · Map Matching & Route Reconstruction
    </div>
    """,
    unsafe_allow_html=True,
)

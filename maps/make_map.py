"""PERSON 1 - Folium maps.

Run from repo root:
    python -m maps.make_map --split train --trip 00000048
    python -m maps.make_map --split student_test --trip 00000000

Creates maps/html/<split>_<trip>.html with layers:
    road network (grey), raw GPS (blue), reference route (green),
    predicted route (red, optional), start/end markers, legend + layer control.
"""
import argparse
import os

import folium
import numpy as np

from common.io import DATA_DIR, load_trip

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "html")

LEGEND = """
<div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999; background: white;
 padding: 8px 12px; border: 1px solid #999; border-radius: 6px; font-size: 13px; box-shadow: 0 1px 5px rgba(0,0,0,0.2);">
<b>PathSnap (Dataset Visualisation)</b><br>
<span style="color:#777777">&#9644;</span> Road network (<code>.nodes</code>, <code>.arcs</code>)<br>
<span style="color:#1f77b4">&#9679;</span> Raw GPS (<code>.track</code>)<br>
<span style="color:#2ca02c">&#9644;</span> Reference route (<code>.route</code>)<br>
<span style="color:#d62728">&#9644;</span> Predicted route (optional)
</div>"""


def arc_lines(trip, arc_ids):
    nl, na = trip.nodes["lat"].to_numpy(), trip.nodes["lon"].to_numpy()
    u, v = trip.arc_u[arc_ids], trip.arc_v[arc_ids]
    return [[(nl[a], na[a]), (nl[b], na[b])] for a, b in zip(u, v)]


def route_polyline(trip, arc_ids):
    """Continuous polyline through a sequence of arcs (consecutive duplicates removed)."""
    ids = [a for i, a in enumerate(arc_ids) if i == 0 or a != arc_ids[i - 1]]
    nl, na = trip.nodes["lat"].to_numpy(), trip.nodes["lon"].to_numpy()
    pts = []
    for a in ids:
        u, v = trip.arc_u[a], trip.arc_v[a]
        pts += [(nl[u], na[u]), (nl[v], na[v])]
    return pts


# 1. Dataset-only blank tile: 1x1 transparent data-URI (0 external network calls, 100% offline)
BLANK_TILE = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"

# 2. Compatible external basemaps: strictly visual backgrounds, no API key required, no watermark
ESRI_CANVAS_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"
ESRI_CANVAS_ATTR = "Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ"

ESRI_STREET_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}"
ESRI_STREET_ATTR = (
    "Tiles &copy; Esri &mdash; Source: Esri, DeLorme, NAVTEQ, USGS, Intermap, iPC, NRCAN, Esri Japan, METI, TomTom"
)


def make_map(trip, predicted_arcs=None, max_network_arcs=6000, default_basemap="canvas"):
    t = trip.track
    m = folium.Map(location=[t["lat"].mean(), t["lon"].mean()], zoom_start=14, tiles=None)

    # Base layer 1: Dataset Only (No external map tiles needed)
    folium.TileLayer(
        tiles=BLANK_TILE,
        attr="PathSnap Dataset Mode &mdash; Original Project Data Only",
        name="Dataset Only (No Tiles)",
        control=True,
        show=(default_basemap == "dataset_only"),
    ).add_to(m)

    # Base layer 2: Compatible external basemap - Esri Light Gray Canvas (visual background only)
    folium.TileLayer(
        tiles=ESRI_CANVAS_URL,
        attr=ESRI_CANVAS_ATTR,
        name="Esri Light Gray Canvas",
        control=True,
        show=(default_basemap == "canvas"),
        max_zoom=16,
    ).add_to(m)

    # Base layer 3: Compatible external basemap - Esri World Street Map (optional visual reference)
    folium.TileLayer(
        tiles=ESRI_STREET_URL,
        attr=ESRI_STREET_ATTR,
        name="Esri World Street Map",
        control=True,
        show=(default_basemap == "street"),
        max_zoom=18,
    ).add_to(m)

    # road network: from provided .nodes and .arcs (only arcs near the trajectory)
    pad = 0.003
    nl, na = trip.nodes["lat"].to_numpy(), trip.nodes["lon"].to_numpy()
    u, v = trip.arc_u, trip.arc_v
    inside = ((na[u] > t.lon.min() - pad) & (na[u] < t.lon.max() + pad) &
              (nl[u] > t.lat.min() - pad) & (nl[u] < t.lat.max() + pad))
    ids = np.flatnonzero(inside)[:max_network_arcs]
    fg_net = folium.FeatureGroup("Road network (.nodes & .arcs)", show=True)
    folium.PolyLine(arc_lines(trip, ids), color="#777777", weight=1.5, opacity=0.7).add_to(fg_net)
    fg_net.add_to(m)

    # raw GPS observations: from provided .track
    fg_gps = folium.FeatureGroup("Raw GPS (.track)", show=True)
    step = max(1, len(t) // 1500)  # keep html light
    for lat, lon, ts in t[["lat", "lon", "t"]].to_numpy()[::step]:
        folium.CircleMarker((lat, lon), radius=2.5, color="#1f77b4", fill=True, fill_opacity=.85,
                            weight=0, tooltip=f"t={ts:.0f}s").add_to(fg_gps)
    fg_gps.add_to(m)

    # reference route: from provided .route
    if trip.route is not None:
        fg_ref = folium.FeatureGroup("Reference route (.route)", show=True)
        folium.PolyLine(route_polyline(trip, list(trip.route)), color="#2ca02c", weight=6, opacity=.7).add_to(fg_ref)
        fg_ref.add_to(m)

    # predicted route: optional
    if predicted_arcs is not None and len(predicted_arcs):
        fg_pred = folium.FeatureGroup("Predicted route (model)", show=True)
        folium.PolyLine(route_polyline(trip, list(predicted_arcs)), color="#d62728", weight=3.5, opacity=.9).add_to(fg_pred)
        fg_pred.add_to(m)

    folium.Marker((t.lat.iloc[0], t.lon.iloc[0]), tooltip="Start (t=0s)", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker((t.lat.iloc[-1], t.lon.iloc[-1]), tooltip=f"End (t={t.t.iloc[-1]:.0f}s)", icon=folium.Icon(color="red", icon="stop")).add_to(m)
    m.fit_bounds([[t.lat.min(), t.lon.min()], [t.lat.max(), t.lon.max()]])
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(LEGEND))
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--trip", default=None, help="trip id, default = first trip in split")
    ap.add_argument("--basemap", default="canvas", choices=["canvas", "dataset_only", "street"],
                    help="Default basemap layer: canvas, dataset_only, or street")
    a = ap.parse_args()
    base = os.path.join(DATA_DIR, a.split)
    tid = a.trip or sorted(os.listdir(base))[0]
    trip = load_trip(os.path.join(base, tid))
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{a.split}_{tid}.html")
    make_map(trip, default_basemap=a.basemap).save(path)
    print("saved", path)

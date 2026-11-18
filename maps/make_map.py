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
 padding: 8px 12px; border: 1px solid #999; border-radius: 6px; font-size: 13px;">
<b>PathSnap</b><br>
<span style="color:#999">&#9644;</span> Road network<br>
<span style="color:#1f77b4">&#9679;</span> Raw GPS<br>
<span style="color:#2ca02c">&#9644;</span> Reference route<br>
<span style="color:#d62728">&#9644;</span> Predicted route
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


def make_map(trip, predicted_arcs=None, max_network_arcs=6000):
    t = trip.track
    m = folium.Map(location=[t["lat"].mean(), t["lon"].mean()], zoom_start=14, tiles="OpenStreetMap")

    # road network: only arcs near the trajectory (whole network can be 100k+ arcs)
    pad = 0.003
    nl, na = trip.nodes["lat"].to_numpy(), trip.nodes["lon"].to_numpy()
    u, v = trip.arc_u, trip.arc_v
    inside = ((na[u] > t.lon.min() - pad) & (na[u] < t.lon.max() + pad) &
              (nl[u] > t.lat.min() - pad) & (nl[u] < t.lat.max() + pad))
    ids = np.flatnonzero(inside)[:max_network_arcs]
    fg_net = folium.FeatureGroup("Road network", show=True)
    folium.PolyLine(arc_lines(trip, ids), color="#999999", weight=1, opacity=0.6).add_to(fg_net)  # multi-line
    fg_net.add_to(m)

    fg_gps = folium.FeatureGroup("Raw GPS", show=True)
    step = max(1, len(t) // 1500)  # keep html light
    for lat, lon, ts in t[["lat", "lon", "t"]].to_numpy()[::step]:
        folium.CircleMarker((lat, lon), radius=2, color="#1f77b4", fill=True, fill_opacity=.8,
                            weight=0, tooltip=f"t={ts:.0f}s").add_to(fg_gps)
    fg_gps.add_to(m)

    if trip.route is not None:
        fg_ref = folium.FeatureGroup("Reference route", show=True)
        folium.PolyLine(route_polyline(trip, list(trip.route)), color="#2ca02c", weight=6, opacity=.6).add_to(fg_ref)
        fg_ref.add_to(m)

    if predicted_arcs is not None and len(predicted_arcs):
        fg_pred = folium.FeatureGroup("Predicted route", show=True)
        folium.PolyLine(route_polyline(trip, list(predicted_arcs)), color="#d62728", weight=3, opacity=.9).add_to(fg_pred)
        fg_pred.add_to(m)

    folium.Marker((t.lat.iloc[0], t.lon.iloc[0]), tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker((t.lat.iloc[-1], t.lon.iloc[-1]), tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(m)
    m.fit_bounds([[t.lat.min(), t.lon.min()], [t.lat.max(), t.lon.max()]])
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(LEGEND))
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--trip", default=None, help="trip id, default = first trip in split")
    a = ap.parse_args()
    base = os.path.join(DATA_DIR, a.split)
    tid = a.trip or sorted(os.listdir(base))[0]
    trip = load_trip(os.path.join(base, tid))
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{a.split}_{tid}.html")
    make_map(trip).save(path)
    print("saved", path)

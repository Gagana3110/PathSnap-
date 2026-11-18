"""Shared data loading + coordinate projection for PathSnap.

Every other module imports from here, so all parts of the team read the raw
files in exactly the same way.

Raw files per trip (whitespace separated, no header):
  .track  lon lat t        GPS observations (~1 Hz)
  .nodes  lon lat          road nodes      (row index = node_id)
  .arcs   from to          road segments   (row index = arc_id, directed)
  .route  arc_id           reference route, chronological (NOT one per GPS point)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("PATHSNAP_DATA", os.path.join(ROOT, "data"))
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

EARTH_R = 6_371_000.0  # metres


def list_trips(split: str, data_dir: str | None = None) -> list[str]:
    """Return sorted trip folder paths for a split ('train', 'student_test', ...)."""
    base = os.path.join(data_dir or DATA_DIR, split)
    return sorted(
        os.path.join(base, d) for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))
    )


@dataclass
class Trip:
    trip_id: str
    track: pd.DataFrame          # lon, lat, t, x, y   (cleaned, time-sorted)
    nodes: pd.DataFrame          # lon, lat, x, y      (index = node_id)
    arcs: pd.DataFrame           # u, v                (index = arc_id)
    route: np.ndarray | None     # arc ids or None (if no .route file)
    lat0: float = 0.0
    lon0: float = 0.0
    cleaning: dict = field(default_factory=dict)

    # ---------- projection: local equirectangular (metres) ----------
    # Lon/lat are angles, so we cannot use Euclidean distance on degrees.
    # Each trip covers a few km, so a local tangent-plane projection around
    # the trip centre has < 0.1% distance error (checked in EDA vs haversine).
    def to_xy(self, lon, lat):
        x = np.radians(np.asarray(lon) - self.lon0) * EARTH_R * np.cos(np.radians(self.lat0))
        y = np.radians(np.asarray(lat) - self.lat0) * EARTH_R
        return x, y

    def to_lonlat(self, x, y):
        lon = self.lon0 + np.degrees(np.asarray(x) / (EARTH_R * np.cos(np.radians(self.lat0))))
        lat = self.lat0 + np.degrees(np.asarray(y) / EARTH_R)
        return lon, lat

    # convenience arrays
    @property
    def arc_u(self):
        return self.arcs["u"].to_numpy()

    @property
    def arc_v(self):
        return self.arcs["v"].to_numpy()

    def arc_xy(self):
        """(ax, ay, bx, by) arrays for every arc."""
        nx_, ny_ = self.nodes["x"].to_numpy(), self.nodes["y"].to_numpy()
        u, v = self.arc_u, self.arc_v
        return nx_[u], ny_[u], nx_[v], ny_[v]

    def arc_length(self):
        ax, ay, bx, by = self.arc_xy()
        return np.hypot(bx - ax, by - ay)


def _read(path, names, dtype=None):
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=dtype, engine="c")


def load_trip(folder: str, with_route: bool = True) -> Trip:
    tid = os.path.basename(os.path.normpath(folder))
    p = lambda ext: os.path.join(folder, f"{tid}.{ext}")

    track = _read(p("track"), ["lon", "lat", "t"])
    nodes = _read(p("nodes"), ["lon", "lat"])
    arcs = _read(p("arcs"), ["u", "v"], dtype=np.int64)
    route = None
    if with_route and os.path.exists(p("route")):
        route = _read(p("route"), ["arc"], dtype=np.int64)["arc"].to_numpy()

    # ---------------- cleaning (documented in report) ----------------
    n0 = len(track)
    track = track.dropna()
    n_nan = n0 - len(track)
    ok = track["lon"].between(-180, 180) & track["lat"].between(-90, 90)
    n_bad = int((~ok).sum())
    track = track[ok]
    n_unsorted = int((track["t"].diff() < 0).sum())
    track = track.sort_values("t", kind="stable")
    n_dup = int(track["t"].duplicated().sum())
    track = track.drop_duplicates("t", keep="first").reset_index(drop=True)

    lat0, lon0 = float(track["lat"].mean()), float(track["lon"].mean())
    trip = Trip(tid, track, nodes, arcs, route, lat0, lon0,
                dict(raw_rows=n0, nan_rows=n_nan, invalid_coords=n_bad,
                     unsorted_steps=n_unsorted, duplicate_timestamps=n_dup))
    trip.track["x"], trip.track["y"] = trip.to_xy(track["lon"], track["lat"])
    trip.nodes["x"], trip.nodes["y"] = trip.to_xy(nodes["lon"], nodes["lat"])
    return trip


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))

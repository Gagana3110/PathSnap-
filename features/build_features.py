"""PERSON 2 - Candidate generation, label derivation and feature engineering.

Run from repo root:
    python -m features.build_features                 # all train + student_test trips
    python -m features.build_features --radius 30     # try other candidate radius
    python -m features.build_features --radius-study  # candidate-miss-rate table for several radii

Outputs (results/):
    features_train.parquet, features_student_test.parquet
        one row = (GPS observation, candidate arc); label = 1 if candidate is the reference arc
    obs_train.parquet, obs_student_test.parquet
        one row = GPS observation; true_arc, label distance, reliability, candidate-miss flag
    candidate_stats.csv   candidate miss rate etc. per split
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial import cKDTree

from common.geometry import angle_diff, bearing, point_seg
from common.io import RESULTS_DIR, list_trips, load_trip

DEFAULT_CFG = dict(
    radius_m=40.0,        # candidate search radius
    max_candidates=12,    # keep the nearest K arcs inside the radius
    fallback_k=3,         # if nothing inside radius, take K nearest arcs
    densify_step_m=10.0,  # spacing of sample points used for the KD-tree search
    label_window=10,      # route arcs looked at ahead of the pointer when labelling
    label_step_pen_m=2.0, # penalty per route arc skipped (prefers staying on current arc)
    label_jump_m=50.0,    # if best in window is further than this, search whole remaining route
    label_reliable_m=50.0 # labels farther than this from the GPS point are flagged unreliable
)

FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio", "n_cand",
    "proj_frac", "beyond_end_m", "arc_len",
    "heading_diff", "heading_diff_prev", "speed", "low_speed",
    "dist_prev1", "dist_next1", "dist_prev3", "dist_next3", "dist_win_mean",
    "dt_prev", "dt_next", "deg_in_u", "deg_out_v", "has_reverse",
]


# --------------------------------------------------------------------------- candidates
def generate_candidates(trip, cfg=DEFAULT_CFG):
    """Return arrays (obs_idx, arc_id, dist, t_unclipped) of candidate pairs."""
    R, step = cfg["radius_m"], cfg["densify_step_m"]
    px, py = trip.track["x"].to_numpy(), trip.track["y"].to_numpy()
    ax, ay, bx, by = trip.arc_xy()
    L = np.hypot(bx - ax, by - ay)

    # 1) keep only arcs near the trajectory bounding box (networks can have 100k+ arcs)
    pad = R + 100
    near = ((np.minimum(ax, bx) < px.max() + pad) & (np.maximum(ax, bx) > px.min() - pad) &
            (np.minimum(ay, by) < py.max() + pad) & (np.maximum(ay, by) > py.min() - pad))
    ids = np.flatnonzero(near)

    # 2) densify arcs into sample points every `step` metres and build a KD-tree
    n = np.ceil(L[ids] / step).astype(int) + 1
    arc_rep = np.repeat(ids, n)
    k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    f = k / np.repeat(np.maximum(n - 1, 1), n)
    sx = ax[arc_rep] + f * (bx - ax)[arc_rep]
    sy = ay[arc_rep] + f * (by - ay)[arc_rep]
    tree = cKDTree(np.c_[sx, sy])

    # 3) ball query (radius + half spacing guarantees no arc within R is missed)
    hits = tree.query_ball_point(np.c_[px, py], R + step / 2 + 0.5)
    lens = np.fromiter((len(h) for h in hits), int, len(hits))
    pid = np.repeat(np.arange(len(px)), lens)
    arc = arc_rep[np.concatenate([np.asarray(h, int) for h in hits])] if lens.sum() else np.array([], int)

    # 4) fallback for points with no hits: nearest samples
    empty = np.flatnonzero(lens == 0)
    if len(empty):
        _, nn = tree.query(np.c_[px[empty], py[empty]], k=min(60, len(sx)))
        pid = np.concatenate([pid, np.repeat(empty, nn.shape[1])])
        arc = np.concatenate([arc, arc_rep[nn.ravel()]])

    # unique pairs + exact distances
    key = np.unique(pid.astype(np.int64) * len(L) + arc)
    pid, arc = key // len(L), key % len(L)
    d, t = point_seg(px[pid], py[pid], ax[arc], ay[arc], bx[arc], by[arc])

    df = pd.DataFrame(dict(obs_idx=pid, arc_id=arc, dist=d, t=t))
    df["r"] = df.groupby("obs_idx")["dist"].rank(method="first")
    keep = (df["dist"] <= R) & (df["r"] <= cfg["max_candidates"])
    has_any = df[keep].obs_idx.unique()
    fb = (~df.obs_idx.isin(has_any)) & (df["r"] <= cfg["fallback_k"])
    df = df[keep | fb].sort_values(["obs_idx", "dist"]).reset_index(drop=True)
    return df["obs_idx"].to_numpy(), df["arc_id"].to_numpy(), df["dist"].to_numpy(), df["t"].to_numpy()


# --------------------------------------------------------------------------- labels
def derive_labels(trip, cfg=DEFAULT_CFG):
    """Assign one reference arc to every GPS point.

    The .route file is an ordered list of arcs (not one per point). We walk through
    the GPS points in time order with a pointer into the route that only moves
    forward; each point takes the closest route arc within a small window ahead
    (with a small penalty per skipped arc). If nothing in the window is close
    (e.g. after a long GPS gap) we search the rest of the route.
    Returns (true_arc, label_dist).
    """
    r = trip.route
    px, py = trip.track["x"].to_numpy(), trip.track["y"].to_numpy()
    ax, ay, bx, by = (a[r] for a in trip.arc_xy())
    W, pen, jump = cfg["label_window"], cfg["label_step_pen_m"], cfg["label_jump_m"]
    p = 0
    true_arc = np.empty(len(px), dtype=np.int64)
    ldist = np.empty(len(px))
    for i in range(len(px)):
        hi = min(len(r), p + W)
        d, _ = point_seg(px[i], py[i], ax[p:hi], ay[p:hi], bx[p:hi], by[p:hi])
        cost = d + pen * np.arange(hi - p)
        k = int(np.argmin(cost))
        if d[k] > jump and hi < len(r):  # lost: search the remaining route
            d2, _ = point_seg(px[i], py[i], ax[p:], ay[p:], bx[p:], by[p:])
            k2 = int(np.argmin(d2))
            if d2[k2] < d[k]:
                k, d = k2, d2
        p += k
        true_arc[i] = r[p]
        ldist[i] = d[k]
    return true_arc, ldist


# --------------------------------------------------------------------------- features
def build_trip(trip, cfg=DEFAULT_CFG, with_labels=True):
    """Build (pairs_df, obs_df) for one trip. with_labels=False for inference (API)."""
    tr = trip.track
    px, py, ts = tr["x"].to_numpy(), tr["y"].to_numpy(), tr["t"].to_numpy()
    N = len(px)
    pid, arc, dist, tproj = generate_candidates(trip, cfg)
    ax, ay, bx, by = trip.arc_xy()
    L = np.hypot(bx - ax, by - ay)

    df = pd.DataFrame(dict(trip_id=trip.trip_id, obs_idx=pid, arc_id=arc, dist=dist))
    g = df.groupby("obs_idx")["dist"]
    df["dist_rank"] = g.rank(method="min")
    mn = g.transform("min")
    df["dist_minus_min"] = df["dist"] - mn
    df["dist_ratio"] = df["dist"] / (mn + 1.0)
    df["n_cand"] = g.transform("size")
    df["proj_frac"] = np.clip(tproj, 0, 1)
    df["beyond_end_m"] = np.maximum(0, np.maximum(-tproj, tproj - 1)) * L[arc]
    df["arc_len"] = L[arc]

    # vehicle heading & speed from neighbouring points (central difference)
    i0, i1 = np.maximum(np.arange(N) - 1, 0), np.minimum(np.arange(N) + 1, N - 1)
    hx, hy = px[i1] - px[i0], py[i1] - py[i0]
    dtc = np.maximum(ts[i1] - ts[i0], 1e-6)
    speed = np.hypot(hx, hy) / dtc
    head = bearing(hx, hy)
    head_prev = bearing(px - px[i0], py - py[i0])
    arc_b = bearing(bx - ax, by - ay)
    df["heading_diff"] = angle_diff(head[pid], arc_b[arc])
    df["heading_diff_prev"] = angle_diff(head_prev[pid], arc_b[arc])
    df["speed"] = speed[pid]
    df["low_speed"] = (speed[pid] < 1.0).astype(np.int8)  # heading unreliable when nearly stopped

    # distance of neighbouring GPS points to this same arc (temporal context)
    def nb_dist(shift):
        j = np.clip(pid + shift, 0, N - 1)
        return point_seg(px[j], py[j], ax[arc], ay[arc], bx[arc], by[arc])[0]
    df["dist_prev1"], df["dist_next1"] = nb_dist(-1), nb_dist(1)
    df["dist_prev3"], df["dist_next3"] = nb_dist(-3), nb_dist(3)
    df["dist_win_mean"] = np.mean([nb_dist(s) for s in (-2, -1, 0, 1, 2)], axis=0)

    dtp = np.r_[0.0, np.diff(ts)]
    df["dt_prev"] = dtp[pid]
    df["dt_next"] = np.r_[np.diff(ts), 0.0][pid]

    u, v = trip.arc_u, trip.arc_v
    nn = len(trip.nodes)
    deg_in, deg_out = np.bincount(v, minlength=nn), np.bincount(u, minlength=nn)
    df["deg_in_u"] = deg_in[u[arc]]
    df["deg_out_v"] = deg_out[v[arc]]
    rev = pd.MultiIndex.from_arrays([v, u]).isin(pd.MultiIndex.from_arrays([u, v]))
    df["has_reverse"] = rev[arc].astype(np.int8)

    obs = pd.DataFrame(dict(trip_id=trip.trip_id, obs_idx=np.arange(N), t=ts,
                            lon=tr["lon"].to_numpy(), lat=tr["lat"].to_numpy()))
    obs["n_cand"] = np.bincount(pid, minlength=N)

    if with_labels and trip.route is not None:
        true_arc, ldist = derive_labels(trip, cfg)
        df["label"] = (df["arc_id"].to_numpy() == true_arc[pid]).astype(np.int8)
        obs["true_arc"] = true_arc
        obs["label_dist"] = ldist
        obs["reliable"] = (ldist <= cfg["label_reliable_m"]).astype(np.int8)
        obs["in_cand"] = df.groupby("obs_idx")["label"].max().reindex(obs.obs_idx, fill_value=0).to_numpy()
        df = df.merge(obs[["obs_idx", "reliable"]], on="obs_idx", how="left")

    float_cols = [c for c in FEATURES if df[c].dtype == np.float64]
    df[float_cols] = df[float_cols].astype(np.float32)
    return df, obs


def _one(folder, split, cfg):
    trip = load_trip(folder)
    df, obs = build_trip(trip, cfg, with_labels=True)
    df.insert(1, "split", split)
    obs.insert(1, "split", split)
    return df, obs


def build_split(split, cfg, n_jobs=-1):
    res = Parallel(n_jobs=n_jobs, verbose=5)(delayed(_one)(f, split, cfg) for f in list_trips(split))
    return pd.concat([r[0] for r in res], ignore_index=True), pd.concat([r[1] for r in res], ignore_index=True)


def candidate_stats(obs, df, split, cfg):
    rel = obs[obs.reliable == 1]
    return dict(split=split, radius_m=cfg["radius_m"], n_obs=len(obs), n_reliable_obs=len(rel),
                unreliable_frac=1 - len(rel) / len(obs),
                candidate_miss_rate=float(1 - rel.in_cand.mean()),
                mean_candidates=float(obs.n_cand.mean()), max_candidates=int(obs.n_cand.max()),
                n_pairs=len(df), positive_rate=float(df.label.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=DEFAULT_CFG["radius_m"])
    ap.add_argument("--radius-study", action="store_true")
    ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args()

    if a.radius_study:
        rows = []
        for R in (15, 25, 40, 60):
            cfg = {**DEFAULT_CFG, "radius_m": R}
            for split in ("train", "student_test"):
                df, obs = build_split(split, cfg, a.jobs)
                rows.append(candidate_stats(obs, df, split, cfg))
                print(rows[-1])
        pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "candidate_radius_study.csv"), index=False)
        return

    cfg = {**DEFAULT_CFG, "radius_m": a.radius}
    stats = []
    for split in ("train", "student_test"):
        df, obs = build_split(split, cfg, a.jobs)
        df.to_parquet(os.path.join(RESULTS_DIR, f"features_{split}.parquet"), index=False)
        obs.to_parquet(os.path.join(RESULTS_DIR, f"obs_{split}.parquet"), index=False)
        stats.append(candidate_stats(obs, df, split, cfg))
        print(stats[-1])
    pd.DataFrame(stats).to_csv(os.path.join(RESULTS_DIR, "candidate_stats.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "feature_config.json"), "w") as fh:
        json.dump(dict(cfg=cfg, features=FEATURES), fh, indent=2)


if __name__ == "__main__":
    main()

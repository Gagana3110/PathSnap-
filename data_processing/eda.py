"""PERSON 1 - Exploratory data analysis.

Run from repo root:
    python -m data_processing.eda

Outputs (results/eda/):
    inventory.csv          one row per trip with all statistics
    summary_by_split.csv   aggregated statistics per split
    data_quality_notes.txt issues found + decisions
    *.png                  plots for the report
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from common.io import RESULTS_DIR, haversine, list_trips, load_trip
from common.geometry import point_seg

OUT = os.path.join(RESULTS_DIR, "eda")
os.makedirs(OUT, exist_ok=True)


def trip_stats(folder, split):
    tr = load_trip(folder)
    tid = tr.trip_id
    t = tr.track
    dt = t["t"].diff().dropna().to_numpy()
    step = np.hypot(np.diff(t["x"]), np.diff(t["y"]))
    sizes = {ext: os.path.getsize(os.path.join(folder, f"{tid}.{ext}"))
             for ext in ("track", "nodes", "arcs", "route") if os.path.exists(os.path.join(folder, f"{tid}.{ext}"))}

    # projection sanity: local projection vs haversine on consecutive points
    hv = haversine(t["lon"].to_numpy()[:-1], t["lat"].to_numpy()[:-1], t["lon"].to_numpy()[1:], t["lat"].to_numpy()[1:])
    proj_err = float(np.abs(step - hv).max()) if len(hv) else 0.0

    # road network
    G = nx.DiGraph()
    G.add_nodes_from(range(len(tr.nodes)))
    G.add_edges_from(zip(tr.arc_u, tr.arc_v))
    deg = np.array([d for _, d in G.degree()])
    arcset = set(zip(tr.arc_u.tolist(), tr.arc_v.tolist()))
    rev_frac = np.mean([(v, u) in arcset for u, v in arcset])
    alen = tr.arc_length()

    row = dict(split=split, trip_id=tid, n_points=len(t), **{f"raw_{k}": v for k, v in tr.cleaning.items()},
               duration_s=float(t["t"].iloc[-1] - t["t"].iloc[0]),
               dt_frac_1s=float(np.mean(dt == 1)), dt_max=float(dt.max()), n_gaps_gt5s=int((dt > 5).sum()),
               n_gaps_gt30s=int((dt > 30).sum()),
               speed_med_mps=float(np.median(step / np.maximum(dt, 1e-9))),
               speed_p99_mps=float(np.percentile(step / np.maximum(dt, 1e-9), 99)),
               trip_length_km=float(step.sum() / 1000),
               lon_min=t["lon"].min(), lon_max=t["lon"].max(), lat_min=t["lat"].min(), lat_max=t["lat"].max(),
               proj_max_err_m=proj_err,
               n_nodes=len(tr.nodes), n_arcs=len(tr.arcs), arc_len_med_m=float(np.median(alen)),
               zero_len_arcs=int((alen < 1e-6).sum()),
               reverse_arc_frac=float(rev_frac),
               weak_components=nx.number_weakly_connected_components(G),
               largest_wcc_frac=len(max(nx.weakly_connected_components(G), key=len)) / len(tr.nodes),
               isolated_nodes=int((deg == 0).sum()),
               **{f"bytes_{k}": v for k, v in sizes.items()})

    if tr.route is not None:
        r = tr.route
        row["route_len"] = len(r)
        row["route_unique_arcs"] = len(set(r.tolist()))
        row["route_breaks"] = int((tr.arc_v[r[:-1]] != tr.arc_u[r[1:]]).sum())
        row["route_invalid_ids"] = int(((r < 0) | (r >= len(tr.arcs))).sum())
        # GPS noise: distance of each GPS point to the closest route arc
        ax, ay, bx, by = (a[r] for a in tr.arc_xy())
        d, _ = point_seg(t["x"].to_numpy()[:, None], t["y"].to_numpy()[:, None], ax[None], ay[None], bx[None], by[None])
        dmin = d.min(1)
        row.update(gps_to_route_med_m=float(np.median(dmin)), gps_to_route_p95_m=float(np.percentile(dmin, 95)),
                   gps_to_route_max_m=float(dmin.max()))
        return row, dt, dmin
    return row, dt, np.array([])


def main():
    rows, all_dt, all_d = [], {}, {}
    for split in ("train", "student_test"):
        all_dt[split], all_d[split] = [], []
        for f in list_trips(split):
            r, dt, d = trip_stats(f, split)
            rows.append(r)
            all_dt[split].append(dt)
            all_d[split].append(d)
            print(f"{split:12s} {r['trip_id']}  pts={r['n_points']:6d} arcs={r['n_arcs']:7d} "
                  f"gps->route med={r.get('gps_to_route_med_m', float('nan')):.1f}m")
    inv = pd.DataFrame(rows)
    inv.to_csv(os.path.join(OUT, "inventory.csv"), index=False)

    num = inv.select_dtypes("number").columns
    summ = inv.groupby("split")[num].agg(["count", "mean", "median", "min", "max"]).T
    summ.to_csv(os.path.join(OUT, "summary_by_split.csv"))
    print("\nTrips per split:\n", inv["split"].value_counts())

    # -------- plots (train split unless stated) --------
    def save(name):
        plt.tight_layout(); plt.savefig(os.path.join(OUT, name), dpi=130); plt.close()

    plt.figure(figsize=(6, 4))
    for s in ("train", "student_test"):
        plt.hist(inv.loc[inv.split == s, "n_points"], bins=20, alpha=.6, label=s)
    plt.xlabel("GPS points per trip"); plt.ylabel("trips"); plt.legend(); plt.title("Trajectory length")
    save("points_per_trip.png")

    dt = np.concatenate(all_dt["train"])
    plt.figure(figsize=(6, 4))
    vals, cnts = np.unique(np.clip(dt, 0, 10), return_counts=True)
    plt.bar(vals, cnts); plt.yscale("log"); plt.xlabel("time gap between points (s, clipped at 10)")
    plt.ylabel("count (log)"); plt.title("Sampling interval (train)")
    save("sampling_interval.png")

    d = np.concatenate(all_d["train"])
    plt.figure(figsize=(6, 4))
    plt.hist(np.clip(d, 0, 60), bins=60); plt.xlabel("distance GPS point -> nearest route arc (m, clipped 60)")
    plt.ylabel("points"); plt.title("GPS noise (train)")
    save("gps_noise.png")

    plt.figure(figsize=(6, 4))
    plt.scatter(inv["n_arcs"], inv["n_nodes"], c=(inv.split == "train"), cmap="coolwarm", s=15)
    plt.xscale("log"); plt.yscale("log"); plt.xlabel("arcs"); plt.ylabel("nodes"); plt.title("Road network size per trip")
    save("network_size.png")

    plt.figure(figsize=(7, 7))
    for f in list_trips("train")[:12]:
        t = load_trip(f).track
        plt.plot(t["lon"], t["lat"], lw=.8)
    plt.xlabel("lon"); plt.ylabel("lat"); plt.title("12 training trajectories")
    save("trajectories.png")

    missing = inv[[c for c in inv.columns if c.startswith("raw_")]].sum()
    missing.to_csv(os.path.join(OUT, "missing_values.csv"))

    with open(os.path.join(OUT, "data_quality_notes.txt"), "w") as fh:
        fh.write("DATA QUALITY SUMMARY (auto-generated from inventory.csv)\n\n")
        fh.write(f"Trips: {inv.split.value_counts().to_dict()}\n")
        fh.write(f"Raw cleaning counts (all splits):\n{missing.to_string()}\n\n")
        fh.write(f"Fraction of 1-second steps (train): {np.mean(dt == 1):.4f}\n")
        fh.write(f"Trips with gaps > 30 s: {(inv.n_gaps_gt30s > 0).sum()}\n")
        fh.write(f"Max gap: {inv.dt_max.max():.0f} s\n")
        fh.write(f"Median GPS->route distance (train): {np.median(d):.2f} m, 95th pct: {np.percentile(d, 95):.2f} m\n")
        fh.write(f"Trips whose route has breaks (consecutive arcs not connected): {(inv.route_breaks > 0).sum()}\n")
        fh.write(f"Median fraction of arcs that also exist reversed: {inv.reverse_arc_frac.median():.3f}\n")
        fh.write(f"Max local-projection error vs haversine: {inv.proj_max_err_m.max():.4f} m\n")
        fh.write("\nDECISIONS:\n- drop NaN/invalid rows, sort by time, drop duplicate timestamps\n"
                 "- use local metric projection (error shown above) for all distances\n"
                 "- arcs are directed and most roads exist in both directions -> heading feature needed\n"
                 "- route is a list of arcs, so per-point labels are derived by monotonic matching (features/)\n")
    print("\nSaved to", OUT)


if __name__ == "__main__":
    main()

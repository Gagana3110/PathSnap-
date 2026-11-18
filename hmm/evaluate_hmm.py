"""PERSON 4 - Tune HMM parameters and compare classifier-only vs classifier + HMM.

Run from repo root (after tuning.optuna_tune, or after models.train_models):
    python -m hmm.evaluate_hmm

1. Grid search beta / w_emission on 15 training trips (decoder params use train data only).
2. Evaluate classifier-only vs HMM on the 10 student_test trips.
Outputs: results/hmm_grid.csv, results/hmm_comparison.csv, results/plots/hmm_vs_clf.png,
         maps/html/hmm_compare_<trip>.html, updates results/final_config.json["hmm"]
"""
import itertools
import json
import os

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.io import RESULTS_DIR, list_trips, load_trip
from common.pipeline import load_artifacts, predict_trip, score_prediction
from hmm.viterbi import DEFAULT_HMM
from maps.make_map import OUT as MAP_DIR, make_map, route_polyline
from models.train_models import PLOT_DIR


def run(trips, art):
    rows = []
    for t in trips:
        p = predict_trip(t, art)
        s = score_prediction(t, p, art["feat_cfg"])
        rows.append(dict(trip_id=t.trip_id, n_obs=s["n_obs"],
                         clf_top1=s["clf"]["top1_accuracy"], hmm_top1=s["hmm"]["top1_accuracy"],
                         clf_continuity=s["clf"]["route_continuity"], hmm_continuity=s["hmm"]["route_continuity"],
                         clf_breaks=s["clf"]["route_breaks"], hmm_breaks=s["hmm"]["route_breaks"],
                         route_precision=s["route_precision"], route_recall=s["route_recall"], time_s=p["time_s"]))
    return pd.DataFrame(rows)


def main():
    art = load_artifacts()
    print("model:", art["model_path"])

    # ---- 1. grid search on training trips ----
    tune_trips = [load_trip(f) for f in list_trips("train")[::5][:15]]
    grid = []
    for beta, we in itertools.product([2, 5, 10, 25], [0.5, 1.0, 2.0]):
        art["hmm"] = {**DEFAULT_HMM, "beta": beta, "w_emission": we}
        r = run(tune_trips, art)
        w = r.n_obs / r.n_obs.sum()
        grid.append(dict(beta=beta, w_emission=we, hmm_top1=(r.hmm_top1 * w).sum(),
                         hmm_continuity=(r.hmm_continuity * w).sum()))
        print(grid[-1])
    grid = pd.DataFrame(grid)
    grid.to_csv(os.path.join(RESULTS_DIR, "hmm_grid.csv"), index=False)
    best = grid.sort_values("hmm_top1", ascending=False).iloc[0]
    art["hmm"] = {**DEFAULT_HMM, "beta": float(best.beta), "w_emission": float(best.w_emission)}
    print("best HMM params:", art["hmm"])

    cfg_path = os.path.join(RESULTS_DIR, "final_config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    cfg["hmm"] = art["hmm"]
    json.dump(cfg, open(cfg_path, "w"), indent=2)

    # ---- 2. student_test comparison ----
    test_trips = [load_trip(f) for f in list_trips("student_test")]
    res = run(test_trips, art)
    w = res.n_obs / res.n_obs.sum()
    overall = {c: (res[c] * w).sum() for c in ("clf_top1", "hmm_top1", "clf_continuity", "hmm_continuity",
                                               "route_precision", "route_recall")}
    overall.update(trip_id="ALL (obs-weighted)", n_obs=res.n_obs.sum(), clf_breaks=res.clf_breaks.sum(),
                   hmm_breaks=res.hmm_breaks.sum(), time_s=res.time_s.sum())
    res = pd.concat([res, pd.DataFrame([overall])], ignore_index=True)
    res.to_csv(os.path.join(RESULTS_DIR, "hmm_comparison.csv"), index=False)
    print(res.round(4).to_string(index=False))

    per = res.iloc[:-1]
    x = np.arange(len(per))
    plt.figure(figsize=(9, 4))
    plt.bar(x - .2, per.clf_top1, .4, label="classifier only")
    plt.bar(x + .2, per.hmm_top1, .4, label="classifier + HMM")
    plt.xticks(x, per.trip_id, rotation=45); plt.ylabel("Top-1 accuracy"); plt.ylim(0.5, 1)
    plt.title("Student-test: classifier-only vs HMM/Viterbi"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "hmm_vs_clf.png"), dpi=130); plt.close()

    # ---- 3. comparison map for the trip where HMM helped most ----
    i = int((per.hmm_top1 - per.clf_top1).values.argmax())
    t = test_trips[i]
    p = predict_trip(t, art)
    m = make_map(t, p["route"])
    fg = folium.FeatureGroup("Classifier-only (argmax)", show=False)
    folium.PolyLine(route_polyline(t, list(p["clf_arcs"])), color="orange", weight=2).add_to(fg)
    fg.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    os.makedirs(MAP_DIR, exist_ok=True)
    m.save(os.path.join(MAP_DIR, f"hmm_compare_{t.trip_id}.html"))
    print("map saved for trip", t.trip_id)


if __name__ == "__main__":
    main()

"""End-to-end inference pipeline used by hmm/evaluate_hmm.py and the API.

trip folder -> candidates + features -> classifier probabilities -> Viterbi -> route
"""
import json
import os
import time

import joblib
import numpy as np

from common.io import RESULTS_DIR
from common.metrics import route_breaks, route_continuity
from features.build_features import DEFAULT_CFG, FEATURES, build_trip, derive_labels
from hmm.viterbi import DEFAULT_HMM, RoadGraph, continuous_route, viterbi


def load_artifacts(model_path=None):
    model_path = model_path or next(p for p in (os.path.join(RESULTS_DIR, "final_model.joblib"),
                                                os.path.join(RESULTS_DIR, "models", "lgbm.joblib"))
                                    if os.path.exists(p))
    cfg_path = os.path.join(RESULTS_DIR, "final_config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    return dict(model=joblib.load(model_path), model_path=model_path,
                feat_cfg={**DEFAULT_CFG, **cfg.get("feature_cfg", {})},
                hmm=cfg.get("hmm", DEFAULT_HMM))


def predict_trip(trip, art, use_hmm=True):
    """Prediction only - never looks at trip.route."""
    t0 = time.time()
    df, obs = build_trip(trip, art["feat_cfg"], with_labels=False)
    df["p"] = art["model"].predict_proba(df[FEATURES])[:, 1]
    clf = df.sort_values(["obs_idx", "p"], ascending=[True, False]).drop_duplicates("obs_idx")
    out = dict(obs_idx=clf["obs_idx"].to_numpy(), clf_arcs=clf["arc_id"].to_numpy())
    g = RoadGraph(trip)
    if use_hmm:
        _, out["hmm_arcs"] = viterbi(trip, df[["obs_idx", "arc_id", "p", "proj_frac"]], art["hmm"], g)
    final = out.get("hmm_arcs", out["clf_arcs"])
    out["route"] = continuous_route(g, list(final))
    out["time_s"] = time.time() - t0
    out["graph"] = g
    return out


def score_prediction(trip, pred, feat_cfg=DEFAULT_CFG):
    """Server-side scoring against the reference route (labels never go to the predictor)."""
    true_arc, ldist = derive_labels(trip, feat_cfg)
    rel = ldist <= feat_cfg["label_reliable_m"]
    res = {}
    for k in ("clf_arcs", "hmm_arcs"):
        if k in pred:
            a = pred[k]
            res[k.replace("_arcs", "")] = dict(
                top1_accuracy=float(np.mean(a[rel] == true_arc[pred["obs_idx"]][rel])),
                route_continuity=route_continuity(a, trip.arc_u, trip.arc_v),
                route_breaks=route_breaks(a, trip.arc_u, trip.arc_v))
    ref = set(trip.route.tolist())
    pr = set(pred["route"])
    res["route_precision"] = len(pr & ref) / max(len(pr), 1)
    res["route_recall"] = len(pr & ref) / max(len(ref), 1)
    res["n_obs"] = int(len(pred["obs_idx"]))
    res["n_reliable_obs"] = int(rel.sum())
    return res

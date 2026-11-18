"""Metrics shared by models/, hmm/ and api/."""
import numpy as np
import pandas as pd


def top1_predictions(df, prob_col="p"):
    """Highest-probability candidate per observation -> DataFrame(trip_id, obs_idx, pred_arc, p)."""
    best = (df.sort_values(["trip_id", "obs_idx", prob_col], ascending=[True, True, False])
              .drop_duplicates(["trip_id", "obs_idx"]))
    return best[["trip_id", "obs_idx", "arc_id", prob_col]].rename(columns={"arc_id": "pred_arc"})


def top1_accuracy(pred, obs, only_reliable=True):
    """Observation-level Top-1 accuracy. Observations without any candidate count as wrong."""
    o = obs[obs.reliable == 1] if only_reliable else obs
    m = o.merge(pred, on=["trip_id", "obs_idx"], how="left")
    return float((m["pred_arc"] == m["true_arc"]).mean())


def route_continuity(pred_arcs, arc_u, arc_v):
    """Fraction of consecutive predictions that are the same arc or directly connected (a.v == b.u)."""
    a = np.asarray(pred_arcs)
    if len(a) < 2:
        return 1.0
    p, q = a[:-1], a[1:]
    ok = (p == q) | (arc_v[p] == arc_u[q])
    return float(ok.mean())


def route_breaks(pred_arcs, arc_u, arc_v):
    a = np.asarray(pred_arcs)
    if len(a) < 2:
        return 0
    p, q = a[:-1], a[1:]
    return int((~((p == q) | (arc_v[p] == arc_u[q]))).sum())

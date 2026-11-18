"""PERSON 3 - Train and compare >= 5 models (candidate classifier).

Run from repo root (after features.build_features):
    python -m models.train_models
    python -m models.train_models --models lgbm xgb       # only some models

Task: binary classification of (GPS observation, candidate arc) pairs:
      label 1 = candidate is the reference arc.
CV:   GroupKFold(5) on the 80 training trips, grouped by trip_id (no trip in 2 folds).
Test: the 10 student_test trips.

Outputs:
    results/model_comparison.csv   one row per model (CV + student-test metrics, timings)
    results/cv_folds.csv           per-fold CV scores
    results/models/<name>.joblib   fitted model (on all training trips)
    results/plots/*.png            comparison plots, confusion matrices, MLP loss curve
"""
import argparse
import json
import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score)
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.io import DATA_DIR, RESULTS_DIR, load_trip
from common.metrics import route_continuity, top1_accuracy, top1_predictions
from features.build_features import FEATURES

SEED = 42
MODEL_DIR = os.path.join(RESULTS_DIR, "models")
PLOT_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


def get_models(pos_weight):
    """name -> (family, estimator, max training rows (None = all), config string)."""
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    return {
        "logreg": ("Linear", make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1000)), None, "C=1, balanced, scaled"),
        "gnb": ("Probabilistic", make_pipeline(StandardScaler(), GaussianNB()), None, "default, scaled"),
        "knn": ("Instance-based", make_pipeline(StandardScaler(), KNeighborsClassifier(
            n_neighbors=25, weights="distance", n_jobs=-1)), 60_000, "k=25, distance-weighted, 60k sample"),
        "rf": ("Ensemble (bagging)", RandomForestClassifier(
            n_estimators=150, min_samples_leaf=5, max_features="sqrt", class_weight="balanced",
            n_jobs=-1, random_state=SEED), 300_000, "150 trees, leaf=5, balanced, 300k sample"),
        "lgbm": ("Boosting", LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=63, scale_pos_weight=1.0,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1),
            None, "400 trees, lr=0.05, leaves=63"),
        "xgb": ("Boosting", XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=7, subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", eval_metric="logloss", random_state=SEED, n_jobs=-1),
            None, "400 trees, lr=0.05, depth=7"),
        "mlp": ("Neural network", make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=512, learning_rate_init=1e-3,
            max_iter=40, early_stopping=True, validation_fraction=0.1, n_iter_no_change=5,
            random_state=SEED)), 250_000, "64-32 ReLU, adam, early stop, 250k sample"),
    }


def load_split(split):
    df = pd.read_parquet(os.path.join(RESULTS_DIR, f"features_{split}.parquet"))
    obs = pd.read_parquet(os.path.join(RESULTS_DIR, f"obs_{split}.parquet"))
    return df, obs


def sample_rows(df, n, seed=SEED):
    if n is None or len(df) <= n:
        return df
    # sample whole observations so candidate groups stay intact
    keys = df[["trip_id", "obs_idx"]].drop_duplicates()
    frac = n / len(df)
    keep = keys.sample(frac=frac, random_state=seed)
    return df.merge(keep, on=["trip_id", "obs_idx"])


def clf_scores(y, p):
    yhat = (p >= 0.5).astype(int)
    return dict(accuracy=accuracy_score(y, yhat), precision=precision_score(y, yhat, average="macro", zero_division=0),
                recall=recall_score(y, yhat, average="macro"), macro_f1=f1_score(y, yhat, average="macro"),
                log_loss=log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]))


def continuity_on(df_pred, trips):
    vals = []
    for tid, g in df_pred.groupby("trip_id"):
        u, v = trips[tid]
        vals.append(route_continuity(g.sort_values("obs_idx")["pred_arc"].to_numpy(), u, v))
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()

    tr, obs_tr = load_split("train")
    te, obs_te = load_split("student_test")
    tr = tr[tr.reliable == 1].reset_index(drop=True)  # train only on reliable labels
    pos_w = (1 - tr.label.mean()) / tr.label.mean()
    models = get_models(pos_w)
    names = a.models or list(models)

    trips_te = {}
    for tid in te.trip_id.unique():
        t = load_trip(os.path.join(DATA_DIR, "student_test", tid))
        trips_te[tid] = (t.arc_u, t.arc_v)

    gkf = GroupKFold(n_splits=a.folds)
    rows, fold_rows = [], []
    for name in names:
        fam, est, cap, conf = models[name]
        print(f"\n===== {name} ({fam}) =====")
        f1s, lls, t1s = [], [], []
        for k, (i_tr, i_va) in enumerate(gkf.split(tr, tr.label, tr.trip_id)):
            dtr = sample_rows(tr.iloc[i_tr], cap)
            dva = tr.iloc[i_va].copy()
            m = joblib.load(joblib.dump(est, "/tmp/_m.joblib")[0])  # fresh copy
            m.fit(dtr[FEATURES], dtr.label)
            dva["p"] = m.predict_proba(dva[FEATURES])[:, 1]
            s = clf_scores(dva.label, dva.p)
            s["top1"] = top1_accuracy(top1_predictions(dva), obs_tr[obs_tr.trip_id.isin(dva.trip_id.unique())])
            fold_rows.append(dict(model=name, fold=k, **s))
            f1s.append(s["macro_f1"]); lls.append(s["log_loss"]); t1s.append(s["top1"])
            print(f"  fold {k}: macroF1={s['macro_f1']:.4f} logloss={s['log_loss']:.4f} top1={s['top1']:.4f}")
            # training-set score of last fold (overfitting check)
        train_part = sample_rows(tr, cap)
        t0 = time.time()
        est.fit(train_part[FEATURES], train_part.label)
        fit_time = time.time() - t0
        train_f1 = clf_scores(train_part.label, est.predict_proba(train_part[FEATURES])[:, 1])["macro_f1"]

        dte = te.copy()
        t0 = time.time()
        dte["p"] = est.predict_proba(dte[FEATURES])[:, 1]
        inf_ms = (time.time() - t0) / len(dte) * 1000 * 1000  # ms per 1000 rows
        s = clf_scores(dte.label, dte.p)
        pred = top1_predictions(dte)
        row = dict(model=name, family=fam, config=conf,
                   cv_macro_f1=np.mean(f1s), cv_macro_f1_std=np.std(f1s), cv_log_loss=np.mean(lls),
                   cv_top1=np.mean(t1s), train_macro_f1=train_f1,
                   test_top1=top1_accuracy(pred, obs_te), test_macro_f1=s["macro_f1"],
                   test_precision=s["precision"], test_recall=s["recall"], test_accuracy=s["accuracy"],
                   test_log_loss=s["log_loss"], test_route_continuity=continuity_on(pred, trips_te),
                   fit_time_s=fit_time, infer_ms_per_1k_rows=inf_ms)
        rows.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
        joblib.dump(est, os.path.join(MODEL_DIR, f"{name}.joblib"))

        cm = confusion_matrix(dte.label, (dte.p >= .5).astype(int))
        ConfusionMatrixDisplay(cm, display_labels=["not ref", "ref arc"]).plot(cmap="Blues")
        plt.title(f"{name} - student_test"); plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"cm_{name}.png"), dpi=120); plt.close()

        if name == "mlp":  # only model with epoch-wise history
            mlp = est[-1]
            plt.figure(figsize=(6, 4))
            plt.plot(mlp.loss_curve_, label="training loss")
            if mlp.validation_scores_ is not None:
                ax2 = plt.gca().twinx()
                ax2.plot(mlp.validation_scores_, color="orange", label="validation accuracy")
                ax2.set_ylabel("validation accuracy")
            plt.xlabel("epoch"); plt.title("MLP training history"); plt.tight_layout()
            plt.savefig(os.path.join(PLOT_DIR, "mlp_loss_curve.png"), dpi=120); plt.close()

    res = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, "model_comparison.csv")
    if a.models and os.path.exists(out):  # merge with earlier runs
        old = pd.read_csv(out)
        res = pd.concat([old[~old.model.isin(res.model)], res], ignore_index=True)
    res.to_csv(out, index=False)
    fr = pd.DataFrame(fold_rows)
    fout = os.path.join(RESULTS_DIR, "cv_folds.csv")
    if a.models and os.path.exists(fout):
        old = pd.read_csv(fout)
        fr = pd.concat([old[~old.model.isin(fr.model)], fr], ignore_index=True)
    fr.to_csv(fout, index=False)
    make_plots(res)
    print("\n", res[["model", "cv_macro_f1", "cv_top1", "test_top1", "test_macro_f1", "test_log_loss",
                     "test_route_continuity", "infer_ms_per_1k_rows"]].round(4).to_string(index=False))


def make_plots(res):
    res = res.sort_values("test_top1")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(res.model, res.cv_macro_f1, yerr=res.cv_macro_f1_std, capsize=4, color="#4c72b0")
    ax.set_ylabel("CV macro F1 (GroupKFold)"); ax.set_ylim(max(0, res.cv_macro_f1.min() - .1), 1)
    ax.set_title("Cross-validation macro F1"); plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "cv_macro_f1.png"), dpi=130); plt.close()

    x = np.arange(len(res))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - .2, res.test_top1, .4, label="Top-1 accuracy")
    ax.bar(x + .2, res.test_macro_f1, .4, label="Macro F1")
    ax.set_xticks(x, res.model); ax.set_ylim(max(0, min(res.test_top1.min(), res.test_macro_f1.min()) - .1), 1)
    ax.legend(); ax.set_title("Student-test performance"); plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "test_top1_f1.png"), dpi=130); plt.close()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(res.model, res.test_log_loss, color="#dd8452")
    ax.set_ylabel("log loss (lower is better)"); ax.set_title("Student-test log loss"); plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "test_log_loss.png"), dpi=130); plt.close()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(res.model, res.infer_ms_per_1k_rows, color="#55a868")
    ax.set_yscale("log"); ax.set_ylabel("ms per 1000 candidate rows"); ax.set_title("Inference time")
    plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, "inference_time.png"), dpi=130); plt.close()


if __name__ == "__main__":
    main()

"""PERSON 3 - Bayesian hyperparameter optimisation (Optuna TPE) of the selected model (LightGBM).

Run from repo root (after models.train_models):
    python -m tuning.optuna_tune --trials 30
    (search uses 3-fold GroupKFold on a 30% observation sample for speed: --folds / --sample to change;
     the final tuned-vs-untuned comparison uses all training rows and 5 folds)

Objective : maximise mean GroupKFold(5) CV observation-level Top-1 accuracy on the 80 training trips
            (macro F1 and log loss of every trial are also logged).
Outputs   : results/optuna_trials.csv, results/plots/optuna_history.png,
            results/final_model.joblib, results/final_config.json, results/tuned_vs_untuned.csv
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
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

from common.io import RESULTS_DIR
from common.metrics import top1_accuracy, top1_predictions
from features.build_features import DEFAULT_CFG, FEATURES
from hmm.viterbi import DEFAULT_HMM
from models.train_models import PLOT_DIR, SEED, clf_scores, load_split

SEARCH_SPACE = {
    "n_estimators": "int 100..800",
    "learning_rate": "log-uniform 0.01..0.3",
    "num_leaves": "int 15..255",
    "min_child_samples": "int 5..200",
    "subsample": "uniform 0.5..1.0",
    "colsample_bytree": "uniform 0.4..1.0",
    "reg_lambda": "log-uniform 1e-3..10",
    "reg_alpha": "log-uniform 1e-3..10",
}


def suggest(trial):
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 800),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 255),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 200),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
    )


def make(params):
    return LGBMClassifier(**params, subsample_freq=1, random_state=SEED, verbose=-1, n_jobs=-1)


def cv_eval(params, tr, obs_tr, folds):
    f1, ll, t1 = [], [], []
    for i_tr, i_va in GroupKFold(folds).split(tr, tr.label, tr.trip_id):
        m = make(params).fit(tr.iloc[i_tr][FEATURES], tr.iloc[i_tr].label)
        va = tr.iloc[i_va].copy()
        va["p"] = m.predict_proba(va[FEATURES])[:, 1]
        s = clf_scores(va.label, va.p)
        f1.append(s["macro_f1"]); ll.append(s["log_loss"])
        o = obs_tr.merge(va[["trip_id", "obs_idx"]].drop_duplicates(), on=["trip_id", "obs_idx"])
        t1.append(top1_accuracy(top1_predictions(va), o))
    return np.mean(f1), np.mean(ll), np.mean(t1), np.std(t1)


def test_eval(model, te, obs_te):
    te = te.copy()
    t0 = time.time()
    te["p"] = model.predict_proba(te[FEATURES])[:, 1]
    inf = (time.time() - t0) / len(te) * 1e6
    s = clf_scores(te.label, te.p)
    return dict(test_top1=top1_accuracy(top1_predictions(te), obs_te), test_macro_f1=s["macro_f1"],
                test_log_loss=s["log_loss"], infer_ms_per_1k_rows=inf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--sample", type=float, default=0.3)
    a = ap.parse_args()

    tr, obs_tr = load_split("train")
    te, obs_te = load_split("student_test")
    tr = tr[tr.reliable == 1].reset_index(drop=True)

    keys = tr[["trip_id", "obs_idx"]].drop_duplicates().sample(frac=a.sample, random_state=SEED)
    tr_s = tr.merge(keys, on=["trip_id", "obs_idx"]).reset_index(drop=True)
    print(f"search data: {len(tr_s)} rows ({a.sample:.0%} of observations), {a.folds} folds")

    def objective(trial):
        f1, ll, t1, sd = cv_eval(suggest(trial), tr_s, obs_tr, a.folds)
        trial.set_user_attr("cv_macro_f1", f1)
        trial.set_user_attr("cv_log_loss", ll)
        trial.set_user_attr("cv_top1_std", sd)
        return t1

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
                                study_name="pathsnap_lgbm")
    t0 = time.time()
    study.optimize(objective, n_trials=a.trials, show_progress_bar=False)
    search_time = time.time() - t0
    trials = study.trials_dataframe()
    trials.to_csv(os.path.join(RESULTS_DIR, "optuna_trials.csv"), index=False)
    print("best CV top1:", study.best_value, "\nbest params:", study.best_params)

    plt.figure(figsize=(7, 4))
    plt.plot(trials.number, trials.value, "o", alpha=.6, label="trial")
    plt.plot(trials.number, trials.value.cummax(), "r-", label="best so far")
    plt.xlabel("trial"); plt.ylabel("CV Top-1 accuracy"); plt.title("Optuna (TPE) optimisation history")
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, "optuna_history.png"), dpi=130); plt.close()

    # ---- tuned vs untuned on identical data / metrics ----
    untuned_params = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8, colsample_bytree=0.8)
    rows = []
    for name, params in (("untuned", untuned_params), ("tuned", study.best_params)):
        f1, ll, t1, sd = cv_eval(params, tr, obs_tr, 5)
        t0 = time.time()
        m = make(params).fit(tr[FEATURES], tr.label)
        fit = time.time() - t0
        rows.append(dict(model=f"lgbm_{name}", cv_macro_f1=f1, cv_log_loss=ll, cv_top1=t1,
                         fit_time_s=fit, **test_eval(m, te, obs_te)))
        if name == "tuned":
            joblib.dump(m, os.path.join(RESULTS_DIR, "final_model.joblib"))
    res = pd.DataFrame(rows)
    res["search_time_s"] = [np.nan, search_time]
    res.to_csv(os.path.join(RESULTS_DIR, "tuned_vs_untuned.csv"), index=False)
    print(res.round(4).to_string(index=False))

    cfg_path = os.path.join(RESULTS_DIR, "final_config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    cfg.update(model="lightgbm", best_params=study.best_params, search_space=SEARCH_SPACE,
               n_trials=a.trials, objective="mean GroupKFold CV observation-level Top-1 accuracy",
               features=FEATURES, feature_cfg=DEFAULT_CFG)
    cfg.setdefault("hmm", DEFAULT_HMM)
    json.dump(cfg, open(cfg_path, "w"), indent=2)


if __name__ == "__main__":
    main()

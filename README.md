# PathSnap – Distributed Map-Matching & Route Reconstruction

PathSnap matches noisy GPS observations to a road network and reconstructs the route the vehicle actually drove.
Pipeline: **candidate generation → feature engineering → candidate classifier (LightGBM) → HMM/Viterbi decoding → FastAPI backend → evaluator UI**.

## Results summary (student-test partition, 10 trips, 19,160 GPS observations)
| Stage | Top-1 accuracy | Macro F1 | Log loss | Route continuity |
|---|---|---|---|---|
| Best untuned model (XGBoost / LightGBM, tied) | 86.2% / 86.1% | 0.915 | 0.104 | 93.8% |
| LightGBM after Bayesian optimisation (Optuna TPE, 10 trials) | 86.3% | 0.917 | 0.103 | – |
| **LightGBM + HMM/Viterbi (final system)** | **89.5%** | – | – | **98.1%** |

- 7 models compared (Logistic Regression, Gaussian NB, KNN, Random Forest, LightGBM, XGBoost, MLP) with GroupKFold(5) cross-validation by trip – see `results/model_comparison.csv`.
- HMM reduced route breaks from 1,154 to 361 (−69%) and improved Top-1 on 8 of 10 student-test trips – see `results/hmm_comparison.csv`.
- Candidate radius 40 m chosen from a radius study (miss rate 0.63% train / 0.26% test) – see `results/candidate_radius_study.csv`.
- All numbers in this README and in the report are produced by the scripts in this repository.

## Repository structure
| Folder | Contents |
|---|---|
| `common/` | Shared code: data loading + local metric projection (`io.py`), geometry, metrics, end-to-end inference pipeline (`pipeline.py`) |
| `data_processing/` | Raw-data inspection, cleaning and EDA (`eda.py`) |
| `features/` | Candidate generation, label derivation, feature engineering (`build_features.py`) |
| `models/` | Training and comparison of all models, metrics, plots (`train_models.py`) |
| `tuning/` | Bayesian hyperparameter optimisation with Optuna (`optuna_tune.py`) |
| `hmm/` | HMM + Viterbi route reconstruction (`viterbi.py`) and evaluation (`evaluate_hmm.py`) |
| `api/` | FastAPI backend – runs on Laptop 1 (`main.py`) |
| `ui/` | Streamlit evaluator dashboard – runs on Laptop 2 (`app.py`) |
| `maps/` | Folium map generation (`make_map.py`) and generated HTML maps (`html/`) |
| `results/` | EDA outputs, experiment CSVs, Optuna trials, plots, saved models, final config |
| `report/` | Technical report (PDF) |

## Setup
Python 3.10+.
```bash
pip install -r requirements.txt
```
Place the dataset (not committed) as:
```
data/train/<trip_id>/<trip_id>.track | .nodes | .arcs | .route        (80 trips)
data/student_test/<trip_id>/...                                       (10 trips)
data/eval/<trip_id>/...                                               (evaluator trips – API laptop only)
```

## Reproducing the experiments (run from the repository root)
| Step | Command | Main outputs |
|---|---|---|
| 1. EDA | `python -m data_processing.eda` | `results/eda/` |
| 2. Folium map | `python -m maps.make_map --split train --trip 00000048` | `maps/html/` |
| 3. Candidates + features | `python -m features.build_features` (add `--radius-study` for the radius table) | `results/features_*.parquet`, `results/candidate_stats.csv` |
| 4. Train & compare models | `python -m models.train_models` | `results/model_comparison.csv`, `results/cv_folds.csv`, `results/plots/`, `results/models/` |
| 5. Bayesian optimisation | `python -m tuning.optuna_tune --trials 10` | `results/final_model.joblib`, `results/optuna_trials.csv`, `results/tuned_vs_untuned.csv` |
| 6. HMM tuning + comparison | `python -m hmm.evaluate_hmm` | `results/hmm_comparison.csv`, `results/hmm_grid.csv`, `maps/html/hmm_compare_*.html` |

Random seed 42 is used throughout. `results/features_*.parquet` and `results/models/rf.joblib` (150 MB) are not committed because of size; regenerate them with steps 3 and 4.

## Two-laptop deployment
**Laptop 1 – API**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
- Loads `results/final_model.joblib` and the HMM parameters from `results/final_config.json`.
- Serves trips from `data/eval/` (falls back to `data/student_test/` for testing). Override with the `PATHSNAP_EVAL_DIR` environment variable.
- Endpoints: `GET /health`, `GET /trips`, `POST /predict` with body `{"trip_id": "...", "use_hmm": true}`.
- Every request is logged in the terminal (trip, Top-1, continuity, processing time).
- Reference routes stay on the server: trips are loaded **without** the `.route` file for prediction; the route is read separately only to score the result.
- Windows firewall (run as Administrator once):
  `New-NetFirewallRule -DisplayName "PathSnap API" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow`

**Laptop 2 – UI**
```bash
streamlit run ui/app.py
```
- Set the Backend API URL in the sidebar to `http://<laptop-1-IPv4>:8000` (find it with `ipconfig`), or set the `PATHSNAP_API` environment variable before starting.
- Both laptops must be on the same Wi-Fi network or phone hotspot. Check with `curl http://<laptop-1-IPv4>:8000/trips` from Laptop 2.
- The UI only talks to the API: it holds no model and no ground truth.

## Method notes
- **Coordinates:** trips come from different regions, so distances use a local equirectangular projection centred on each trip (max error < 1 m vs haversine, checked in EDA).
- **Labels:** `.route` lists arcs, not one per GPS point, so per-point labels are derived by walking forward along the route; points more than 50 m from the route are flagged unreliable and excluded from training and scoring.
- **Classifier:** each (GPS point, candidate arc) pair is a row with 22 features; label 1 = reference arc.
- **HMM:** hidden states = candidate arcs; emission = 2.0 × log(classifier probability); transition = −|network distance − GPS distance| / β with β = 10 (Dijkstra on the road graph); unreachable transitions get a fixed penalty; transitions reset across GPS gaps > 10 s or > 500 m.
- **Leakage:** GroupKFold by trip; decoder parameters tuned on training trips only; evaluator data never used for training or selection.

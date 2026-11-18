# PathSnap – Distributed Map-Matching & Route Reconstruction

## Repository structure
| Folder | Contents |
|---|---|
| `data_processing/` | Reading, inspecting, cleaning raw trips; EDA |
| `features/` | Candidate generation, labelling, feature engineering |
| `models/` | Training scripts for all models, comparison, metrics, plots |
| `tuning/` | Bayesian hyperparameter optimisation (Optuna) and trial logs |
| `hmm/` | HMM + Viterbi route reconstruction |
| `api/` | FastAPI backend (Laptop 1) |
| `ui/` | Evaluator UI (Laptop 2) |
| `maps/` | Folium map-generation code and HTML maps |
| `results/` | CSVs, experiment logs, saved model, configs |
| `report/` | Technical report (PDF) |

## Setup
```bash
pip install -r requirements.txt
```
Place the dataset (not committed) as:
```
data/train/<trip_id>/<trip_id>.track|.nodes|.arcs|.route      (80 trips)
data/student_test/<trip_id>/...                                 (10 trips)
data/eval/<trip_id>/...                                         (evaluator trips, API laptop only)
```

## Running (always from the repo root)
| Step | Command | Output |
|---|---|---|
| 1. EDA | `python -m data_processing.eda` | `results/eda/` (CSV tables, data quality summary, 5 plots) |
| 2. Folium map | `python -m maps.make_map --split train --trip 00000048` | `maps/html/train_00000048.html` |
| 2a. Folium map (custom trip) | `python -m maps.make_map --split <split> --trip <trip_id>` | `maps/html/<split>_<trip_id>.html` |
| 3. Candidates + features | `python -m features.build_features` (`--radius-study` for miss-rate table) | `results/features_*.parquet` |
| 4. Train/compare models | `python -m models.train_models` | `results/model_comparison.csv`, `results/plots/` |
| 5. Bayesian optimisation | `python -m tuning.optuna_tune --trials 30` | `results/final_model.joblib`, `results/optuna_trials.csv` |
| 6. HMM tuning + comparison | `python -m hmm.evaluate_hmm` | `results/hmm_comparison.csv` |
| 7. API (Laptop 1) | `uvicorn api.main:app --host 0.0.0.0 --port 8000` | logs in terminal |
| 8. UI (Laptop 2) | `streamlit run ui/app.py` | browser opens |

### Person 1 – EDA and Map Generation Commands
- **Run Complete Dataset EDA:**
  ```bash
  python -m data_processing.eda
  ```
  Generates `results/eda/inventory.csv`, `results/eda/summary_by_split.csv`, `results/eda/missing_values.csv`, `results/eda/data_quality_notes.txt`, and 5 diagnostic plots (`points_per_trip.png`, `sampling_interval.png`, `gps_noise.png`, `network_size.png`, `trajectories.png`).

- **Generate Interactive Folium Maps:**
  ```bash
  # Baseline training trip (default: Esri Light Gray Canvas basemap)
  python -m maps.make_map --split train --trip 00000048

  # Pure Dataset-Only mode (100% offline, zero external map tiles)
  python -m maps.make_map --split train --trip 00000048 --basemap dataset_only

  # Representative regional training trip
  python -m maps.make_map --split train --trip 00000001

  # Representative trip with route break
  python -m maps.make_map --split train --trip 00000002

  # Student test trip
  python -m maps.make_map --split student_test --trip 00000000
  ```
  Generated maps are saved in `maps/html/` and include road network from `.nodes` & `.arcs` (grey), raw GPS from `.track` (blue), reference route from `.route` (green), predicted route (red, when available), start/end markers, interactive layer controls (toggle between Dataset Only, Esri Canvas, and Esri Street), and a floating legend. Detailed analysis is documented in [`report/person1_eda_and_maps.md`](report/person1_eda_and_maps.md).

Shared code lives in `common/` (`io.py` loading + projection, `geometry.py`, `metrics.py`, `pipeline.py`).

### Two-laptop deployment
1. Both laptops on the same Wi-Fi / phone hotspot.
2. Laptop 1: `ipconfig` -> note IPv4 address; allow port 8000 in the firewall; start the API.
3. Laptop 2: open `http://<laptop1-ip>:8000/docs` in a browser to check, then `streamlit run ui/app.py`
   and enter `http://<laptop1-ip>:8000` in the sidebar.
4. The API terminal on Laptop 1 prints a log line for every request (trip, accuracy, time).

## AI-use disclosure
_TODO_

"""PERSON 4 - FastAPI backend (runs on LAPTOP 1).

Start from repo root:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Evaluation trips folder (server-side only; contains .track/.nodes/.arcs and the .route used for scoring):
    default: data/eval        override: set PATHSNAP_EVAL_DIR=path\\to\\eval_trips
    (if data/eval does not exist, data/student_test is used so you can test)

Endpoints:
    GET  /health            -> status + model in use
    GET  /trips             -> list of trip ids available for evaluation
    POST /predict           -> body {"trip_id": "..."}; runs the full pipeline and returns route + metrics

The predictor never sees the .route file: the trip is loaded WITHOUT the route for prediction,
and the route is loaded separately afterwards only to compute accuracy on the server.
"""
import logging
import os
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from common.io import DATA_DIR, load_trip
from common.pipeline import load_artifacts, predict_trip, score_prediction

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap-api")

EVAL_DIR = os.environ.get("PATHSNAP_EVAL_DIR") or (
    os.path.join(DATA_DIR, "eval") if os.path.isdir(os.path.join(DATA_DIR, "eval"))
    else os.path.join(DATA_DIR, "student_test"))

app = FastAPI(title="PathSnap API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ART = load_artifacts()
log.info("Loaded model %s | HMM params %s", ART["model_path"], ART["hmm"])
log.info("Evaluation trips folder: %s", EVAL_DIR)


class PredictRequest(BaseModel):
    trip_id: str
    use_hmm: bool = True


def trip_ids():
    return sorted(d for d in os.listdir(EVAL_DIR) if os.path.isdir(os.path.join(EVAL_DIR, d)))


@app.get("/health")
def health():
    return {"status": "ok", "model": os.path.basename(ART["model_path"]), "eval_dir": EVAL_DIR}


@app.get("/trips")
def trips():
    ids = trip_ids()
    log.info("GET /trips -> %d trips", len(ids))
    return {"trips": ids}


@app.post("/predict")
def predict(req: PredictRequest):
    if req.trip_id not in trip_ids():
        raise HTTPException(404, f"unknown trip {req.trip_id}")
    log.info("POST /predict trip=%s use_hmm=%s", req.trip_id, req.use_hmm)
    t0 = time.time()
    folder = os.path.join(EVAL_DIR, req.trip_id)

    trip = load_trip(folder, with_route=False)          # prediction: no labels
    pred = predict_trip(trip, ART, use_hmm=req.use_hmm)

    ref = load_trip(folder, with_route=True)            # scoring: server-side only
    metrics = score_prediction(ref, pred, ART["feat_cfg"]) if ref.route is not None else {}

    nl, na = trip.nodes["lat"].to_numpy(), trip.nodes["lon"].to_numpy()
    route_pts = []
    for a in pred["route"]:
        u, v = int(trip.arc_u[a]), int(trip.arc_v[a])
        route_pts += [[float(nl[u]), float(na[u])], [float(nl[v]), float(na[v])]]
    final_arcs = pred.get("hmm_arcs", pred["clf_arcs"])
    elapsed = time.time() - t0

    main = metrics.get("hmm" if req.use_hmm else "clf", {})
    log.info("trip=%s points=%d candidates->route arcs=%d | top1=%.4f continuity=%.4f | clf_top1=%.4f | %.2fs",
             req.trip_id, len(trip.track), len(pred["route"]), main.get("top1_accuracy", float("nan")),
             main.get("route_continuity", float("nan")),
             metrics.get("clf", {}).get("top1_accuracy", float("nan")), elapsed)

    return {
        "trip_id": req.trip_id,
        "n_points": len(trip.track),
        "gps": trip.track[["lat", "lon"]].round(6).values.tolist(),
        "predicted_route": route_pts,
        "predicted_arcs_per_point": [int(a) for a in final_arcs],
        "metrics": metrics,
        "processing_time_s": round(elapsed, 3),
    }

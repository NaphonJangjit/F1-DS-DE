"""HTTP API for the F1 web page: serves the ONLINE model lineage (SGD / ARF / kNN) of the drift lab.

Every sector of 2024 onward was predicted by the online model that was serving at that race, BEFORE the
model learned that sector (predict 10 laps, then learn them). The served model per race and the
champion/challenger decisions come from _drift_lab.py, so the web's AI Predict panel and its drift-lab
section always show the same models and the same numbers.

Run (from this folder):
    py -3.14 _web_api.py                  # http://127.0.0.1:8000
Needs model/prod/ from:  py -3.14 _drift_lab.py   (which needs the online stream of _online.py)

Endpoints
    GET  /health    lineage + coverage
    GET  /sessions  races with predictions: [{session_key, year, country, circuit, date, model, version}]
    POST /predict   {"driver_number": 81, "lap": 42, "session_key": 9998}
"""

import argparse
import json
import os
import threading
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROD = ROOT / "model" / "prod"
SECTORS = (1, 2, 3)
MODELS = ("SGD", "ARF", "kNN")
CORS_ORIGINS = [o.strip() for o in os.environ.get("F1_CORS_ORIGINS", "*").split(",") if o.strip()]


def _log(msg):
    print(f"[web-api] {msg}", flush=True)


class Store:
    def __init__(self):
        self._lock = threading.Lock()
        self.rows = self.sessions = self.meta = None
        self._mtime = None

    def get(self):
        with self._lock:
            p = PROD / "web_predictions.parquet"
            if not p.exists() or not (PROD / "meta.json").exists():
                raise FileNotFoundError(f"no predictions in {PROD}: run py -3.14 _drift_lab.py")
            if self._mtime != p.stat().st_mtime:             # reload after a new _drift_lab.py run
                self.rows = pd.read_parquet(p)
                self.meta = json.loads((PROD / "meta.json").read_text(encoding="utf-8"))
                self.sessions = pd.DataFrame(self.meta["sessions"])
                self._mtime = p.stat().st_mtime
                _log(f"loaded {len(self.rows):,} streamed sectors, {len(self.sessions)} races")
        return self


STORE = Store()


def _num(v, nd=3):
    return None if v is None or not np.isfinite(v) else round(float(v), nd)


def resolve_session(st, session_key=None, country=None, year=None):
    s = st.sessions
    span = f"{st.meta['stream']} (2023 is the warm-up the models learn from)"
    if session_key is not None:
        hit = s[s["session_key"] == int(session_key)]
        if hit.empty:
            raise LookupError(f"No AI prediction for this race. The online models predict the races of {span}.")
        return hit.iloc[0], []
    if country is None or year is None:
        raise ValueError("send session_key, or country + year")
    hit = s[(s["year"] == int(year)) & (s["country"].str.lower() == str(country).strip().lower())]
    if hit.empty:
        raise LookupError(f"No AI prediction for {year} {country}. The online models predict the races of {span}.")
    warn = [f"{country} hosted {len(hit)} races in {year}; used the first. Send session_key to choose."] if len(hit) > 1 else []
    return hit.sort_values("date").iloc[0], warn


def predict_lap(driver_number, lap, session_key=None, country=None, year=None, compare_actual=True):
    st = STORE.get()
    sess, warnings = resolve_session(st, session_key, country, year)
    sk, driver_number, lap = int(sess["session_key"]), int(driver_number), int(lap)
    rows = st.rows[(st.rows["Session_Key"] == sk) & (st.rows["Driver"] == driver_number)]
    if rows.empty:
        raise LookupError(f"Driver #{driver_number} has no predicted laps in this race.")
    r = rows[rows["Lap"] == lap].set_index("Sector")
    if r.empty:
        raise LookupError(f"No prediction for lap {lap} of driver #{driver_number}: this lap is not in the data "
                          f"(lap 1 has no previous lap to build the inputs from; the AI predicts every other lap, "
                          f"this driver has laps {int(rows['Lap'].min())}-{int(rows['Lap'].max())}).")
    preds = []
    for s in SECTORS:
        if s not in r.index:
            preds.append({"sector": s, "predicted_duration": None, "actual_duration": None, "error": None})
            continue
        x = r.loc[s]
        p, a, anchor = float(x["Pred_Duration_s"]), float(x["actual"]), float(x["anchor"])
        act = a if compare_actual and np.isfinite(a) else None
        preds.append({
            "sector": s, "predicted_duration": _num(p), "actual_duration": _num(act),
            "error": _num(p - act) if act is not None else None,
            "last_clean_lap": _num(anchor),
            "change_pct": _num((p / anchor - 1) * 100, 2) if anchor > 0 else None,
            "track_status": str(x["Track_Status"]),
            "models": {m: _num(float(x[f"pred_{m}"])) for m in MODELS},
        })
    missing = [p["sector"] for p in preds if p["predicted_duration"] is None]
    if missing:
        warnings.append(f"No prediction for sector {', '.join(map(str, missing))} (not in the data); "
                        "the lap total is left empty.")
    sc = [p["sector"] for p in preds if p.get("track_status") in ("SC", "VSC")]
    if sc:
        warnings.append(f"Safety Car / VSC at the start of sector {', '.join(map(str, sc))}.")
    model, version = sess["model"], sess["version"]
    all_pred = not missing
    all_act = all(p["actual_duration"] is not None for p in preds)
    return {
        "request": {"driver_number": driver_number, "lap": lap, "session_key": sk, "country": sess["country"],
                    "year": int(sess["year"]), "session_name": "Race"},
        "session_key": sk,
        "lap_start": str(r["Time_Start"].iloc[0]),
        "predictions": preds,
        "predicted_lap_time": _num(sum(p["predicted_duration"] for p in preds)) if all_pred else None,
        "actual_lap_time": _num(sum(p["actual_duration"] for p in preds)) if all_act else None,
        "warnings": warnings,
        "model": f"{version} · {model} online (predicted live, then learned every 10 laps)",
        "model_version": version, "model_family": model,
        "model_test_mae": st.meta.get("live_mae_all"),
        "in_training": False,
    }


def create_app():
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field

    class PredictRequest(BaseModel):
        driver_number: int
        lap: int = Field(..., ge=1)
        session_key: int | None = None
        country: str | None = None
        year: int | None = None
        session_name: str = "Race"
        compare_actual: bool = True

    api = FastAPI(title="F1 Sector AI (online lineage)")
    api.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])

    @api.middleware("http")
    async def private_network(request: Request, call_next):
        resp = await call_next(request)
        if request.headers.get("access-control-request-private-network") == "true":
            resp.headers["Access-Control-Allow-Private-Network"] = "true"
        return resp

    @api.get("/health")
    def health():
        try:
            st = STORE.get()
        except FileNotFoundError as exc:
            return {"status": "no-model", "detail": str(exc)}
        m = st.meta
        return {"status": "ok", "family": m["family"], "stream": m["stream"], "champion": m["champion"],
                "champion_model": m["champion_model"], "races": int(len(st.sessions)),
                "served_mae": m["live_mae_all"], "lineage": m["lineage"]}

    @api.get("/sessions")
    def sessions():
        return STORE.get().sessions.sort_values("date").to_dict("records")

    @api.get("/race/{session_key}")
    def race(session_key: int):
        """every streamed sector of one race, for the replay: {driver: [[lap, sector, pred, actual], ...]}"""
        st = STORE.get()
        hit = st.sessions[st.sessions["session_key"] == session_key]
        if hit.empty:
            raise HTTPException(status_code=404, detail="No AI predictions for this race "
                                                        f"(the online models cover {st.meta['stream']}).")
        rows = st.rows[st.rows["Session_Key"] == session_key].sort_values(["Driver", "Lap", "Sector"])
        out = {}
        for drv, g in rows.groupby("Driver"):
            out[str(int(drv))] = [[int(l), int(s), round(float(p), 3), round(float(a), 3)]
                                  for l, s, p, a in zip(g["Lap"], g["Sector"], g["Pred_Duration_s"], g["actual"])]
        s = hit.iloc[0]
        return {"session_key": session_key, "model": f"{s['version']} {s['model']} online",
                "version": s["version"], "family": s["model"], "sectors": out}

    @api.post("/predict")
    def predict_endpoint(req: PredictRequest):
        if req.session_name.lower() != "race":
            raise HTTPException(status_code=422, detail="The models predict race laps only (session: Race).")
        try:
            return predict_lap(req.driver_number, req.lap, req.session_key, req.country, req.year, req.compare_actual)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc))

    return api


try:
    app = create_app()
except ImportError:
    app = None


def main():
    ap = argparse.ArgumentParser(description="Serve the online model lineage to the F1 web page")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    if app is None:
        raise SystemExit("install first: pip install fastapi uvicorn")
    STORE.get()
    import uvicorn
    _log(f"ready on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

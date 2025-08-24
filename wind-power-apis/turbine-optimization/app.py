# turbine-optimization/app.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import os, joblib, numpy as np, pandas as pd

API_KEY = os.getenv("API_KEY")

@app.middleware("http")
async def require_api_key(request: Request, call_next):
    path = request.url.path
    # allow unauthenticated access to health/docs/schema if you want
    open_paths = ("/", "/healthz", "/docs", "/openapi.json", "/schema", "/info")
    if any(path.startswith(p) for p in open_paths):
        return await call_next(request)
    if request.headers.get("x-api-key") != API_KEY:
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    return await call_next(request)

MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.joblib")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.joblib")
FEATURES_PATH = os.getenv("FEATURES_PATH", "models/features.joblib")

app = FastAPI(title="Turbine Control Optimization API", version="1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# Load artifacts saved by your training pipeline (model, scaler, features) :contentReference[oaicite:5]{index=5}
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURES_PATH)
except Exception as e:
    raise RuntimeError(f"Failed loading artifacts: {e}")

class PredictRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(..., description="List of feature dicts")

@app.get("/healthz")
def health():
    return {"status": "ok", "model": type(model).__name__, "features": len(features)}

@app.get("/schema")
def schema():
    return {"required_features": list(features), "count": len(features)}

@app.post("/predict")
def predict(req: PredictRequest):
    df = pd.DataFrame(req.records)

    # Ensure all expected features exist
    for col in features:
        if col not in df.columns:
            df[col] = np.nan
    df = df[list(features)]
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.fillna(df.median(numeric_only=True))

    try:
        X = scaler.transform(df)
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[:, 1]
        y = model.predict(X)

        results = []
        for i, pred in enumerate(y):
            conf = float(proba[i]) if proba is not None else None
            results.append({
                "operate_turbine": bool(pred),
                "confidence": conf,
                "recommendation": "OPERATE" if pred == 1 else "STANDBY"
            })
        return {"results": results}
    except Exception as e:
        raise HTTPException(400, f"Inference failed: {e}")

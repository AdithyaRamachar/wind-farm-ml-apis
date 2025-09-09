# turbine-optimization/app.py (updated)
"""
Turbine Control Optimization FastAPI with S3-backed artifact downloads at startup.
- Downloads model, scaler, and features files from S3 into local paths when env vars are provided.
- Falls back to local artifacts if present.
- Preserves API key middleware and endpoints.

Environment variables used:
- MODEL_S3_BUCKET         : S3 bucket name
- MODEL_S3_KEY           : S3 key for the model (e.g. turbine/best_model.joblib)
- SCALER_S3_KEY          : (optional) S3 key for the scaler (e.g. turbine/scaler.joblib)
- FEATURES_S3_KEY        : (optional) S3 key for features list (e.g. turbine/features.joblib)
- MODEL_PATH             : local path to model (default: models/best_model.joblib)
- SCALER_PATH            : local path to scaler (default: models/scaler.joblib)
- FEATURES_PATH          : local path to features list (default: models/features.joblib)
- AWS_REGION             : optional region for boto3 client
- API_KEY                : optional x-api-key expected in requests
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os
import joblib
import numpy as np
import pandas as pd
import logging
from contextlib import asynccontextmanager

# S3
import boto3
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("turbine-api")

# Config from env
API_KEY = os.getenv("API_KEY")
MODEL_S3_BUCKET = os.getenv("MODEL_S3_BUCKET")
MODEL_S3_KEY = os.getenv("MODEL_S3_KEY")
SCALER_S3_KEY = os.getenv("SCALER_S3_KEY")
FEATURES_S3_KEY = os.getenv("FEATURES_S3_KEY")

MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.joblib")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.joblib")
FEATURES_PATH = os.getenv("FEATURES_PATH", "models/features.joblib")
AWS_REGION = os.getenv("AWS_REGION")


def download_from_s3(bucket: str, key: str, dest_path: str, region: Optional[str] = None) -> bool:
    logger.info(f"Attempting to download s3://{bucket}/{key} -> {dest_path}")
    s3 = boto3.client("s3", region_name=region) if region else boto3.client("s3")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    try:
        s3.download_file(bucket, key, dest_path)
        logger.info(f"Downloaded {key} to {dest_path}")
        return True
    except ClientError as e:
        logger.error(f"S3 download failed for {key}: {e}")
        return False
    except Exception as e:
        logger.exception(f"Unexpected error downloading from S3: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup handler: download artifacts if requested and load them."""
    global model, scaler, features

    # Download model artifact if requested
    if MODEL_S3_BUCKET and MODEL_S3_KEY and not os.path.exists(MODEL_PATH):
        ok = download_from_s3(MODEL_S3_BUCKET, MODEL_S3_KEY, MODEL_PATH, AWS_REGION)
        if not ok:
            logger.error("Could not download model from S3; startup will continue to try local files and may fail.")

    # Download scaler/features if S3 keys provided
    if MODEL_S3_BUCKET and SCALER_S3_KEY and not os.path.exists(SCALER_PATH):
        download_from_s3(MODEL_S3_BUCKET, SCALER_S3_KEY, SCALER_PATH, AWS_REGION)
    if MODEL_S3_BUCKET and FEATURES_S3_KEY and not os.path.exists(FEATURES_PATH):
        download_from_s3(MODEL_S3_BUCKET, FEATURES_S3_KEY, FEATURES_PATH, AWS_REGION)

    # Load artifacts (fail fast if model missing)
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        logger.info(f"Loaded model from {MODEL_PATH}")
    except Exception as e:
        logger.exception(f"Failed loading model: {e}")
        raise RuntimeError(f"Failed loading model: {e}")

    # Load scaler (optional)
    try:
        scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
        if scaler is not None:
            logger.info(f"Loaded scaler from {SCALER_PATH}")
        else:
            logger.warning("Scaler not found; proceeding without scaler")
    except Exception as e:
        logger.exception(f"Failed loading scaler: {e}")
        raise RuntimeError(f"Failed loading scaler: {e}")

    # Load features (optional but recommended)
    try:
        if os.path.exists(FEATURES_PATH):
            features = joblib.load(FEATURES_PATH)
            logger.info(f"Loaded features list ({len(features)}) from {FEATURES_PATH}")
        else:
            features = []
            logger.warning("Features list not found; endpoints that rely on features may fail")
    except Exception as e:
        logger.exception(f"Failed loading features: {e}")
        raise RuntimeError(f"Failed loading features: {e}")

    yield

    logger.info("Shutting down turbine-optimization service")


# Attach lifespan
app = FastAPI(title="Turbine Control Optimization API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.router.lifespan_context = lifespan

# API key middleware
@app.middleware("http")
async def require_api_key(request: Request, call_next):
    path = request.url.path
    open_paths = ("/", "/healthz", "/docs", "/openapi.json", "/schema", "/info")
    if any(path.startswith(p) for p in open_paths):
        return await call_next(request)
    if API_KEY and request.headers.get("x-api-key") != API_KEY:
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    return await call_next(request)


class PredictRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(..., description="List of feature dicts")


@app.get("/healthz")
def health():
    return {"status": "ok", "model": type(model).__name__ if 'model' in globals() else None, "features": len(features) if 'features' in globals() else 0}


@app.get("/schema")
def schema():
    return {"required_features": list(features) if 'features' in globals() else [], "count": len(features) if 'features' in globals() else 0}


@app.post("/predict")
def predict(req: PredictRequest):
    if 'features' not in globals() or not features:
        raise HTTPException(status_code=500, detail="Features schema not loaded")

    df = pd.DataFrame(req.records)

    # Ensure all expected features exist
    for col in features:
        if col not in df.columns:
            df[col] = np.nan
    df = df[list(features)]
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.fillna(df.median(numeric_only=True))

    try:
        X = scaler.transform(df) if scaler is not None else df.values
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
        logger.exception(f"Inference failed: {e}")
        raise HTTPException(400, f"Inference failed: {e}")

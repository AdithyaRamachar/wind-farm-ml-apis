# alert-classification/app.py (updated)
"""
FastAPI app with S3-backed model download at startup.
- Downloads model bundle from S3 into local path at service startup when env vars are provided.
- Falls back to local bundle if present.
- Keeps your existing API key middleware and endpoints.

Environment variables expected (set these in App Runner or locally for testing):
- BUNDLE_S3_BUCKET (optional) : S3 bucket name containing the bundle
- BUNDLE_S3_KEY    (optional) : S3 key/path to the bundle (e.g. alert/model_bundle.pkl)
- BUNDLE_PATH      (optional) : local path to write/read bundle (default: models/model_bundle.pkl)
- AWS_REGION       (optional) : region for boto3 client
- API_KEY          (optional) : expected x-api-key for protected endpoints

Requirements: boto3, botocore
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import os
import pickle
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
logger = logging.getLogger("alert-api")

# Config from env
API_KEY = os.getenv("API_KEY")
BUNDLE_S3_BUCKET = os.getenv("BUNDLE_S3_BUCKET")
BUNDLE_S3_KEY = os.getenv("BUNDLE_S3_KEY")
BUNDLE_PATH = os.getenv("BUNDLE_PATH", "models/model_bundle.pkl")
AWS_REGION = os.getenv("AWS_REGION")

# Helper: download file from S3 to dest path
def download_from_s3(bucket: str, key: str, dest_path: str, region: str | None = None) -> bool:
    logger.info(f"Attempting to download s3://{bucket}/{key} -> {dest_path}")
    s3 = boto3.client("s3", region_name=region) if region else boto3.client("s3")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    try:
        s3.download_file(bucket, key, dest_path)
        logger.info("S3 download succeeded")
        return True
    except ClientError as e:
        logger.error(f"S3 download failed: {e}")
        return False
    except Exception as e:
        logger.exception(f"Unexpected error downloading from S3: {e}")
        return False


app = FastAPI(title="Wind Farm Alert Classification API", version="1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# API key middleware (optional)
@app.middleware("http")
async def require_api_key(request: Request, call_next):
    path = request.url.path
    # allow unauthenticated access to health/docs/schema/info
    open_paths = ("/", "/healthz", "/docs", "/openapi.json", "/schema", "/info")
    if any(path.startswith(p) for p in open_paths):
        return await call_next(request)
    if API_KEY and request.headers.get("x-api-key") != API_KEY:
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup handler: ensure bundle is present (download from S3 if requested) and load artifacts."""
    global model, scaler, target_encoder, feature_columns, target_classes, model_name

    # Attempt S3 download if bucket/key provided AND local file missing
    if BUNDLE_S3_BUCKET and BUNDLE_S3_KEY and not os.path.exists(BUNDLE_PATH):
        ok = download_from_s3(BUNDLE_S3_BUCKET, BUNDLE_S3_KEY, BUNDLE_PATH, AWS_REGION)
        if not ok:
            logger.error("Could not download model bundle from S3. If you expect a local bundle, ensure BUNDLE_PATH exists.")
    else:
        logger.info("No S3 download requested or bundle already present locally")

    # Load the bundle
    try:
        if not os.path.exists(BUNDLE_PATH):
            raise FileNotFoundError(f"Bundle not found at {BUNDLE_PATH}")

        logger.info(f"Loading model bundle from {BUNDLE_PATH}")
        with open(BUNDLE_PATH, "rb") as f:
            artifacts = pickle.load(f)

        model = artifacts["model"]
        scaler = artifacts.get("scaler")
        target_encoder = artifacts.get("target_encoder")
        feature_columns = artifacts.get("feature_columns", [])
        target_classes = list(artifacts.get("target_classes", []))
        model_name = artifacts.get("model_name", "unknown")

        logger.info(f"Loaded model bundle: {model_name}, features={len(feature_columns)}")

    except Exception as e:
        logger.exception(f"Failed to load bundle from {BUNDLE_PATH}: {e}")
        # Raise so App Runner / container startup fails visibly if model is required
        raise RuntimeError(f"Failed to load bundle: {e}")

    yield

    # Shutdown cleanup if needed
    logger.info("Shutting down alert-classification service")


# Attach lifespan to app
app.router.lifespan_context = lifespan

# Pydantic request model
class PredictRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(..., description="List of feature dicts")


# Basic endpoints (health/schema/predict)
@app.get("/healthz")
def health():
    return {"status": "ok", "model": model_name, "n_features": len(feature_columns)}


@app.get("/schema")
def schema():
    return {"required_features": list(feature_columns), "count": len(feature_columns)}


@app.post("/predict")
def predict(req: PredictRequest):
    df = pd.DataFrame(req.records)

    # Align to training features
    for col in feature_columns:
        if col not in df.columns:
            df[col] = np.nan
    df = df[feature_columns]
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.fillna(df.median(numeric_only=True))

    try:
        if scaler is not None:
            X = scaler.transform(df)
        else:
            # if scaler wasn't saved (unlikely), use raw numeric
            X = df.values

        y_encoded = model.predict(X)
        y = target_encoder.inverse_transform(y_encoded) if target_encoder is not None else y_encoded
        probs = model.predict_proba(X) if hasattr(model, "predict_proba") else None

        results = []
        for i, label in enumerate(y):
            item = {"predicted_alert": label}
            if probs is not None:
                p = probs[i]
                top_idx = np.argsort(p)[::-1][:3]
                item["top_classes"] = [
                    {"label": target_encoder.inverse_transform([j])[0], "prob": float(p[j])}
                    for j in top_idx
                ]
            results.append(item)
        return {"results": results}
    except Exception as e:
        logger.exception(f"Inference failed: {e}")
        raise HTTPException(400, f"Inference failed: {e}")


# If running locally for testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8003)), reload=True)

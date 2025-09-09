# power-prediction/app.py (updated)
"""
FastAPI application for wind power generation prediction with S3 model download at startup.
- Downloads model package from S3 into local path at service startup when env vars are provided.
- Falls back to local MODEL_PATH if present.
- Keeps API key middleware and existing endpoints.

Environment variables expected:
- MODEL_S3_BUCKET (optional) : S3 bucket name containing the model package
- MODEL_S3_KEY    (optional) : S3 key/path to the model package (e.g. power/power_model.joblib)
- MODEL_PATH      (optional) : local path to write/read model (default: models/power_model.joblib)
- AWS_REGION      (optional) : region for boto3 client
- API_KEY         (optional) : expected x-api-key for protected endpoints
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
logger = logging.getLogger("power-api")

# Config from env
API_KEY = os.getenv("API_KEY")
MODEL_S3_BUCKET = os.getenv("MODEL_S3_BUCKET")
MODEL_S3_KEY = os.getenv("MODEL_S3_KEY")
MODEL_PATH = os.getenv("MODEL_PATH", "models/power_model.joblib")
AWS_REGION = os.getenv("AWS_REGION")

# Helper: download file from S3 to dest path
def download_from_s3(bucket: str, key: str, dest_path: str, region: Optional[str] = None) -> bool:
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler to ensure model is present and loaded."""
    global model, feature_names, model_name, model_version, model_info

    # Attempt S3 download if bucket/key provided AND local file missing
    if MODEL_S3_BUCKET and MODEL_S3_KEY and not os.path.exists(MODEL_PATH):
        ok = download_from_s3(MODEL_S3_BUCKET, MODEL_S3_KEY, MODEL_PATH, AWS_REGION)
        if not ok:
            logger.error("Could not download model from S3. If you expect a local model, ensure MODEL_PATH exists.")
    else:
        logger.info("No S3 download requested or model already present locally")

    # Load model package
    try:
        logger.info(f"Loading model from {MODEL_PATH}")
        pkg = joblib.load(MODEL_PATH)

        model = pkg["model"]
        feature_names = pkg.get("feature_names") or pkg.get("features") or []
        model_name = pkg.get("model_name", "unknown")
        model_version = pkg.get("model_version", "v1.0")

        model_info = {
            "model_name": model_name,
            "model_version": model_version,
            "feature_count": len(feature_names),
            "target_variable": pkg.get("target_variable", "actual_power_kw"),
            "model_type": pkg.get("model_type", "regression"),
            "training_timestamp": pkg.get("training_timestamp", "unknown"),
            "test_performance": pkg.get("test_performance", {})
        }

        logger.info(f"Model loaded successfully: {model_name} v{model_version}")
        logger.info(f"Features required: {len(feature_names)}")

    except FileNotFoundError:
        error_msg = f"Model file not found at {MODEL_PATH}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    except Exception as e:
        error_msg = f"Failed to load model from {MODEL_PATH}: {str(e)}"
        logger.exception(error_msg)
        raise RuntimeError(error_msg)

    yield

    # Shutdown (cleanup if needed)
    logger.info("Shutting down power-api...")


# Initialize FastAPI app
app = FastAPI(
    title="Wind Power Generation Prediction API",
    version="1.0.0",
    description="API for predicting wind turbine power generation based on weather and operational data",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model components
model = None
feature_names: List[str] = []
model_name = "unknown"
model_version = "v1.0"
model_info = {}

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


# Pydantic models
class PredictRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(
        ...,
        description="List of feature dictionaries for prediction",
    )


class PredictionResponse(BaseModel):
    predictions_kw: List[float]
    model_used: str
    records_processed: int


class HealthResponse(BaseModel):
    status: str
    model: str
    version: str
    features_loaded: int


class SchemaResponse(BaseModel):
    required_features: List[str]
    feature_count: int
    sample_record: Dict[str, Any]


# API Endpoints
@app.get("/", summary="Root endpoint")
async def root():
    return {
        "message": "Wind Power Generation Prediction API",
        "version": model_version,
        "model": model_name,
        "endpoints": {"health": "/healthz", "schema": "/schema", "predict": "/predict", "info": "/info"},
    }


@app.get("/healthz", response_model=HealthResponse, summary="Health check")
async def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(status="healthy", model=model_name, version=model_version, features_loaded=len(feature_names))


@app.get("/info", summary="Model information")
async def model_information():
    if not model_info:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model_info


@app.get("/schema", response_model=SchemaResponse, summary="Get feature schema")
async def get_schema():
    if not feature_names:
        raise HTTPException(status_code=503, detail="Model schema not available")
    sample_record = {}
    for feature in feature_names:
        if "temperature" in feature.lower():
            sample_record[feature] = 15.5
        elif "pressure" in feature.lower():
            sample_record[feature] = 1013.2
        elif "wind_speed" in feature.lower():
            sample_record[feature] = 8.5
        elif "direction" in feature.lower():
            sample_record[feature] = 180.0
        elif "density" in feature.lower():
            sample_record[feature] = 1.225
        elif "capacity" in feature.lower():
            sample_record[feature] = 2000.0
        elif "height" in feature.lower():
            sample_record[feature] = 80.0
        elif "power" in feature.lower():
            sample_record[feature] = 1500.0
        else:
            sample_record[feature] = 0.0
    return SchemaResponse(required_features=feature_names, feature_count=len(feature_names), sample_record=sample_record)


@app.post("/predict", response_model=PredictionResponse, summary="Predict power generation")
async def predict_power(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not feature_names:
        raise HTTPException(status_code=500, detail="Model feature list missing")
    if not request.records:
        raise HTTPException(status_code=400, detail="No records provided")
    try:
        df = pd.DataFrame(request.records)
        logger.info(f"Processing {len(df)} records for prediction")
        for col in feature_names:
            if col not in df.columns:
                df[col] = np.nan
        df = df[feature_names]
        df = df.apply(pd.to_numeric, errors="coerce")
        df_filled = df.fillna(df.median(numeric_only=True))
        df_filled = df_filled.fillna(0)
        predictions = model.predict(df_filled)
        predictions = np.maximum(predictions, 0)
        logger.info(f"Successfully generated {len(predictions)} predictions")
        return PredictionResponse(predictions_kw=[float(pred) for pred in predictions], model_used=model_name, records_processed=len(predictions))
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(status_code=404, content={"error": "Endpoint not found", "available_endpoints": ["/", "/healthz", "/info", "/schema", "/predict"]})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True, log_level="info")

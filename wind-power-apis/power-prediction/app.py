# power-prediction/app.py
"""
FastAPI application for wind power generation prediction
Serves the trained model via REST API endpoints
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os
import joblib
import numpy as np

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

import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "models/power_model.joblib")

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    global model, feature_names, model_name, model_version, model_info
    
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
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    yield
    
    # Shutdown (cleanup if needed)
    logger.info("Shutting down...")

# Initialize FastAPI app
app = FastAPI(
    title="Wind Power Generation Prediction API", 
    version="1.0.0",
    description="API for predicting wind turbine power generation based on weather and operational data",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

# Global variables for model components
model = None
feature_names = []
model_name = "unknown"
model_version = "v1.0"
model_info = {}

# Pydantic models
class PredictRequest(BaseModel):
    """Request model for power prediction"""
    records: List[Dict[str, Any]] = Field(
        ..., 
        description="List of feature dictionaries for prediction",
        example=[{
            "temperature_2m": 15.5,
            "pressure_msl": 1013.2,
            "wind_speed_100m": 8.5,
            "wind_direction_100m": 180.0,
            "air_density_kg_m3": 1.225,
            "capacity_kw": 2000.0,
            "hub_height_m": 80.0
        }]
    )

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    model_config = {"protected_namespaces": ()}
    
    predictions_kw: List[float] = Field(..., description="Predicted power output in kW")
    model_used: str = Field(..., description="Name of the model used for prediction")
    records_processed: int = Field(..., description="Number of records processed")

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    model: str
    version: str
    features_loaded: int

class SchemaResponse(BaseModel):
    """Schema response model"""
    required_features: List[str]
    feature_count: int
    sample_record: Dict[str, Any]

# API Endpoints
@app.get("/", summary="Root endpoint")
async def root():
    """Root endpoint with basic API information"""
    return {
        "message": "Wind Power Generation Prediction API",
        "version": model_version,
        "model": model_name,
        "endpoints": {
            "health": "/healthz",
            "schema": "/schema", 
            "predict": "/predict",
            "info": "/info"
        }
    }

@app.get("/healthz", response_model=HealthResponse, summary="Health check")
async def health():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return HealthResponse(
        status="healthy",
        model=model_name,
        version=model_version,
        features_loaded=len(feature_names)
    )

@app.get("/info", summary="Model information")
async def model_information():
    """Get detailed model information"""
    if not model_info:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return model_info

@app.get("/schema", response_model=SchemaResponse, summary="Get feature schema")
async def get_schema():
    """Get required features schema"""
    if not feature_names:
        raise HTTPException(status_code=503, detail="Model schema not available")
    
    # Create a sample record with example values
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
    
    return SchemaResponse(
        required_features=feature_names,
        feature_count=len(feature_names),
        sample_record=sample_record
    )

@app.post("/predict", response_model=PredictionResponse, summary="Predict power generation")
async def predict_power(request: PredictRequest):
    """
    Predict wind turbine power generation
    
    Takes weather and operational data and returns predicted power output in kW
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not feature_names:
        raise HTTPException(status_code=500, detail="Model feature list missing")
    
    if not request.records:
        raise HTTPException(status_code=400, detail="No records provided")
    
    try:
        # Convert records to DataFrame
        df = pd.DataFrame(request.records)
        logger.info(f"Processing {len(df)} records for prediction")
        
        # Align columns to training features
        for col in feature_names:
            if col not in df.columns:
                df[col] = np.nan
        
        # Select only the features used in training
        df = df[feature_names]
        
        # Convert to numeric and handle missing values
        df = df.apply(pd.to_numeric, errors="coerce")
        
        # Fill missing values with median (same strategy as training)
        df_filled = df.fillna(df.median(numeric_only=True))
        
        # If still NaN values (all NaN column), fill with 0
        df_filled = df_filled.fillna(0)
        
        # Make predictions
        predictions = model.predict(df_filled)
        
        # Ensure non-negative predictions (power can't be negative)
        predictions = np.maximum(predictions, 0)
        
        logger.info(f"Successfully generated {len(predictions)} predictions")
        
        return PredictionResponse(
            predictions_kw=[float(pred) for pred in predictions],
            model_used=model_name,
            records_processed=len(predictions)
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "available_endpoints": ["/", "/healthz", "/info", "/schema", "/predict"]}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=8001, 
        reload=True,
        log_level="info"
    )
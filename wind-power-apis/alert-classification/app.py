# alert-classification/app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import os, pickle, joblib, numpy as np, pandas as pd

BUNDLE_PATH = os.getenv("BUNDLE_PATH", "models/model_bundle.pkl")

app = FastAPI(title="Wind Farm Alert Classification API", version="1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# Load the single bundle that contains model, scaler, target_encoder, feature_columns :contentReference[oaicite:6]{index=6}
try:
    with open(BUNDLE_PATH, "rb") as f:
        artifacts = pickle.load(f)
    model = artifacts["model"]
    scaler = artifacts["scaler"]    # also saved separately in your script
    target_encoder = artifacts["target_encoder"]
    feature_columns = artifacts["feature_columns"]
    target_classes = list(artifacts.get("target_classes", []))
    model_name = artifacts.get("model_name", "unknown")
except Exception as e:
    raise RuntimeError(f"Failed to load bundle: {e}")

class PredictRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(..., description="List of feature dicts")

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
        X = scaler.transform(df)
        y_encoded = model.predict(X)
        y = target_encoder.inverse_transform(y_encoded)
        probs = model.predict_proba(X) if hasattr(model, "predict_proba") else None

        results = []
        for i, label in enumerate(y):
            item = {"predicted_alert": label}
            if probs is not None:
                # Return top-3 classes by probability
                p = probs[i]
                top_idx = np.argsort(p)[::-1][:3]
                item["top_classes"] = [
                    {"label": target_encoder.inverse_transform([j])[0], "prob": float(p[j])}
                    for j in top_idx
                ]
            results.append(item)
        return {"results": results}
    except Exception as e:
        raise HTTPException(400, f"Inference failed: {e}")

# train.py
"""
Wind Farm Alert Classification Training Pipeline
- Preprocess dataset
- Encode categorical + target
- Train candidate models
- Save best model + scaler + encoders in a bundle
"""

import os, joblib, pickle, warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, classification_report

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
DATA_PATH = os.getenv("DATA_PATH", "unified_windfarm_dataset.csv")
MODEL_DIR = os.getenv("MODEL_DIR", "models")
BUNDLE_PATH = os.path.join(MODEL_DIR, "model_bundle.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")

np.random.seed(42)


# -------------------------------------------------------------------
# PIPELINE
# -------------------------------------------------------------------
def preprocess(df):
    # Handle missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ["float64", "int64"]:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

    # Drop ID/time cols
    drop_cols = ["turbine_id", "farm_name", "time", "commissioned_date"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Feature engineering
    if {"actual_power_kw","expected_power_kw"}.issubset(df.columns):
        df["power_ratio"] = df["actual_power_kw"] / (df["expected_power_kw"] + 1e-6)
    if {"nacelle_temp_c","vibration_mm_s"}.issubset(df.columns):
        df["temp_vibration_interaction"] = df["nacelle_temp_c"] * df["vibration_mm_s"]

    return df


def train():
    print("🌪 Training Wind Farm Alert Classification Model")
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df = preprocess(df)

    # Split features / target
    y = df["alert"]
    X = df.drop(columns=["alert"])

    # Encode target
    target_encoder = LabelEncoder()
    y_enc = target_encoder.fit_transform(y)

    # Encode categorical features
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
    )

    # Scale numeric features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Candidate models
    models = {
        "rf": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
        "xgb": xgb.XGBClassifier(random_state=42, eval_metric="mlogloss"),
        "gb": GradientBoostingClassifier(random_state=42),
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    }

    best_model, best_name, best_score = None, None, -1
    results = {}
    for name, mdl in models.items():
        cv = cross_val_score(
            mdl, X_train_scaled, y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring="f1_weighted"
        )
        mdl.fit(X_train_scaled, y_train)
        preds = mdl.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        results[name] = {"cv": cv.mean(), "acc": acc, "f1": f1}
        print(f"Model {name}: CV={cv.mean():.3f} Acc={acc:.3f} F1={f1:.3f}")
        if f1 > best_score:
            best_model, best_name, best_score = mdl, name, f1

    print(f"🏆 Best model: {best_name} (F1={best_score:.3f})")

    # Save artifacts bundle
    bundle = {
        "model": best_model,
        "scaler": scaler,
        "target_encoder": target_encoder,
        "feature_columns": list(X.columns),
        "target_classes": list(target_encoder.classes_),
        "model_name": best_name,
        "performance": results[best_name],
        "training_date": datetime.now().isoformat(),
    }
    with open(BUNDLE_PATH, "wb") as f:
        pickle.dump(bundle, f)

    joblib.dump(scaler, SCALER_PATH)

    print(f"✅ Saved bundle: {BUNDLE_PATH}")
    print(f"✅ Saved scaler: {SCALER_PATH}")


if __name__ == "__main__":
    train()

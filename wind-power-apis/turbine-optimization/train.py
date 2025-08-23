# train.py
"""
Turbine Control Optimization Training Pipeline
- Preprocess data
- Engineer features
- Train multiple classifiers
- Save best model + scaler + feature list
"""

import os, joblib, warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import classification_report, roc_auc_score

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
DATA_PATH = os.getenv("DATA_PATH", "unified_windfarm_dataset.csv")
MODEL_DIR = os.getenv("MODEL_DIR", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
FEATURES_PATH = os.path.join(MODEL_DIR, "features.joblib")

np.random.seed(42)


# -------------------------------------------------------------------
# PIPELINE COMPONENTS
# -------------------------------------------------------------------
def preprocess_data(df):
    df["time"] = pd.to_datetime(df["time"])
    if "commissioned_date" in df.columns:
        df["commissioned_date"] = pd.to_datetime(df["commissioned_date"])

    # fill NaNs numeric
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())

    # create target
    eff_threshold = 0.85
    df["turbine_optimal_advanced"] = (
        (df["efficiency"] >= eff_threshold)
        & (df.get("alert_Normal", pd.Series(True, index=df.index)) == True)
        & (df["actual_power_kw"] >= df["expected_power_kw"] * 0.8)
    ).astype(int)
    return df


def engineer_features(df):
    df["wind_power_density"] = 0.5 * df["air_density_kg_m3"] * (df["wind_speed_100m"] ** 3)
    df["temp_wind_interaction"] = df["temperature_2m"] * df["wind_speed_100m"]
    df["pressure_altitude_factor"] = df["pressure_msl"] / (df["hub_height_m"] + 100)
    df["power_efficiency_ratio"] = df["actual_power_kw"] / df["expected_power_kw"]
    df["capacity_efficiency"] = df["actual_power_kw"] / df["capacity_kw"]
    df["wind_volatility"] = df["wind_speed_100m_24h_std"] / (df["wind_speed_100m_24h_avg"] + 1)
    df["temp_volatility"] = df["temperature_2m_24h_std"] / (abs(df["temperature_2m_24h_avg"]) + 1)
    df["wind_direction_sin"] = np.sin(np.radians(df["wind_direction_100m"]))
    df["wind_direction_cos"] = np.cos(np.radians(df["wind_direction_100m"]))
    df["maintenance_risk"] = (
        (df["vibration_mm_s"] > df["vibration_mm_s"].quantile(0.75))
        | (df["nacelle_temp_c"] > df["nacelle_temp_c"].quantile(0.75))
    ).astype(int)
    df["optimal_weather_window"] = (
        (df["wind_speed_100m"].between(3.5, 15))
        & (df["temperature_2m"].between(-10, 40))
    ).astype(int)
    return df


def select_features(df):
    # keep only numeric columns for safety
    return [col for col in df.columns if df[col].dtype in [np.int64, np.float64]]


def train_models(X_train, X_test, y_train, y_test):
    cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_ratio = cw[0] / cw[1]

    models = {
        "rf": RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, class_weight="balanced", n_jobs=-1),
        "gb": GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=8, random_state=42),
        "xgb": xgb.XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=8, scale_pos_weight=class_ratio, random_state=42, eval_metric="logloss"),
        "lgb": lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, max_depth=8, class_weight="balanced", random_state=42),
        "logreg": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
    }

    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    best_model, best_name, best_score = None, None, -1
    for name, mdl in models.items():
        mdl.fit(X_train_bal, y_train_bal)
        proba = mdl.predict_proba(X_test)[:, 1]
        score = roc_auc_score(y_test, proba)
        print(f"Model {name}: ROC-AUC={score:.4f}")
        print(classification_report(y_test, mdl.predict(X_test)))
        if score > best_score:
            best_model, best_name, best_score = mdl, name, score

    print(f"🏆 Best model: {best_name} (ROC-AUC={best_score:.4f})")
    return best_model, best_name


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    df = preprocess_data(df)
    df = engineer_features(df)

    features = select_features(df)
    X, y = df[features], df["turbine_optimal_advanced"]

    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    best_model, best_name = train_models(X_train_scaled, X_test_scaled, y_train, y_test)

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(features, FEATURES_PATH)

    print(f"✅ Saved model to {MODEL_PATH}")
    print(f"✅ Saved scaler to {SCALER_PATH}")
    print(f"✅ Saved features to {FEATURES_PATH}")


if __name__ == "__main__":
    main()

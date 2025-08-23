# -*- coding: utf-8 -*-
"""
Power Generation Prediction Model Training Script
- Trains regressors
- Evaluates on test split
- Saves final model package for deployment
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings, os, joblib
warnings.filterwarnings('ignore')

# ML
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
DATA_PATH = os.getenv("DATA_PATH", "unified_windfarm_dataset.csv")
MODEL_DIR = os.getenv("MODEL_DIR", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "power_model.joblib")
VERSION = "v1.0"

np.random.seed(42)


# -------------------------------------------------------------------
# LOAD + FEATURE ENGINEERING
# -------------------------------------------------------------------
def load_and_prepare_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df['time'] = pd.to_datetime(df['time'])
    target = 'actual_power_kw'
    df_clean = df.dropna(subset=[target]).copy()

    features = [
        'temperature_2m', 'pressure_msl', 'wind_speed_100m', 'wind_direction_100m',
        'air_density_kg_m3', 'capacity_kw', 'hub_height_m',
        'actual_power_kw_24h_avg', 'actual_power_kw_24h_std',
        'wind_speed_100m_24h_avg', 'wind_speed_100m_24h_std',
        'temperature_2m_24h_avg', 'temperature_2m_24h_std',
        'actual_power_kw_lag1h', 'wind_speed_100m_lag1h',
        'nacelle_temp_c', 'vibration_mm_s', 'expected_power_kw'
    ]
    features = [f for f in features if f in df_clean.columns]

    X = df_clean[features].copy()
    y = df_clean[target].copy()

    # Fill missing
    X = X.fillna(X.median(numeric_only=True))
    return X, y, features, df_clean


# -------------------------------------------------------------------
# SPLITTING
# -------------------------------------------------------------------
def create_time_splits(X, y, df, test_size=0.2, val_size=0.15):
    df_sorted = pd.concat([df[['time']], X, y], axis=1).sort_values('time')
    times = df_sorted['time'].unique()
    n = len(times)

    n_test, n_val = int(n * test_size), int(n * val_size)
    n_train = n - n_test - n_val

    train_end = times[n_train - 1]
    val_end = times[n_train + n_val - 1]

    train = df_sorted[df_sorted['time'] <= train_end]
    val = df_sorted[(df_sorted['time'] > train_end) & (df_sorted['time'] <= val_end)]
    test = df_sorted[df_sorted['time'] > val_end]

    return (train[X.columns], train[y.name],
            val[X.columns], val[y.name],
            test[X.columns], test[y.name])


# -------------------------------------------------------------------
# TRAIN & EVAL
# -------------------------------------------------------------------
def evaluate(model, X, y):
    pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, pred))
    mae = mean_absolute_error(y, pred)
    r2 = r2_score(y, pred)
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def train_models(X_train, y_train, X_val, y_val):
    candidates = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1),
        "XGBoost": xgb.XGBRegressor(n_estimators=100, max_depth=8, learning_rate=0.1, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, max_depth=8, learning_rate=0.1, random_state=42)
    }

    results, models = {}, {}
    for name, mdl in candidates.items():
        mdl.fit(X_train, y_train)
        results[name] = evaluate(mdl, X_val, y_val)
        models[name] = mdl

    best = min(results.items(), key=lambda kv: kv[1]['RMSE'])
    best_name, _ = best
    return models[best_name], best_name, results


# -------------------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------------------
def main():
    print("🌪 Training Power Generation Prediction Model")
    os.makedirs(MODEL_DIR, exist_ok=True)

    X, y, features, df = load_and_prepare_data()
    X_train, y_train, X_val, y_val, X_test, y_test = create_time_splits(X, y, df)

    model, model_name, results = train_models(X_train, y_train, X_val, y_val)
    test_results = evaluate(model, X_test, y_test)

    package = {
        "model": model,
        "model_name": model_name,
        "feature_names": list(X.columns),
        "feature_info": features,
        "test_performance": test_results,
        "training_timestamp": datetime.now().isoformat(),
        "model_version": VERSION,
        "target_variable": "actual_power_kw",
        "model_type": "regression"
    }
    joblib.dump(package, MODEL_PATH)
    print(f"✅ Model saved at {MODEL_PATH}")
    print(f"🏆 Best model: {model_name}")
    print(f"📈 Test RMSE: {test_results['RMSE']:.2f}")


if __name__ == "__main__":
    main()

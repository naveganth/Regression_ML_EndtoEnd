"""
Train XGBoost model for Fish Weight prediction with MLflow tracking.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
from joblib import dump
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

DEFAULT_TRAIN = Path("data/processed/feature_engineered_train.csv")
DEFAULT_EVAL = Path("data/processed/feature_engineered_eval.csv")
DEFAULT_OUT = Path("models/xgb_model.pkl")


def _maybe_sample(
    df: pd.DataFrame, sample_frac: Optional[float], random_state: int
) -> pd.DataFrame:
    if sample_frac is None:
        return df
    return df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)


def train_model(
    train_path: Path | str = DEFAULT_TRAIN,
    eval_path: Path | str = DEFAULT_EVAL,
    model_output: Path | str = DEFAULT_OUT,
    model_params: Optional[Dict] = None,
    sample_frac: Optional[float] = None,
    random_state: int = 42,
):
    # Configuração do MLflow
    mlflow.set_experiment("fish_weight_prediction")

    train_df = pd.read_csv(train_path)
    eval_df = pd.read_csv(eval_path)

    train_df = _maybe_sample(train_df, sample_frac, random_state)
    eval_df = _maybe_sample(eval_df, sample_frac, random_state)

    # --- Target: Weight ---
    target = "Weight"
    X_train, y_train = train_df.drop(columns=[target]), train_df[target]
    X_eval, y_eval = eval_df.drop(columns=[target]), eval_df[target]

    params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": random_state,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    if model_params:
        params.update(model_params)

    # Iniciar Run do MLflow
    with mlflow.start_run(run_name="baseline_xgb"):
        mlflow.log_params(params)

        model = XGBRegressor(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_eval)
        mae = float(mean_absolute_error(y_eval, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_eval, y_pred)))
        r2 = float(r2_score(y_eval, y_pred))

        metrics = {"mae": mae, "rmse": rmse, "r2": r2}

        # Logar métricas
        mlflow.log_metrics(metrics)

        # Logar o modelo no MLflow (com assinatura de input opcional, mas recomendado)
        mlflow.xgboost.log_model(model, "model")

        # Salvar localmente (para a API/Docker usar)
        out = Path(model_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        dump(model, out)

        print(f"✅ Model trained & logged to MLflow.")
        print(f"   MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")

    return model, metrics


if __name__ == "__main__":
    train_model()

"""
Hyperparameter tuning with Optuna + MLflow for Fish Data.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from joblib import dump
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import mlflow
import mlflow.xgboost

DEFAULT_TRAIN = Path("data/processed/feature_engineered_train.csv")
DEFAULT_EVAL = Path("data/processed/feature_engineered_eval.csv")
DEFAULT_OUT = Path("models/xgb_best_model.pkl")


def _load_data(train_path, eval_path):
    train_df = pd.read_csv(train_path)
    eval_df = pd.read_csv(eval_path)

    target = "Weight"  # Ajustado para Peixes

    X_train, y_train = train_df.drop(columns=[target]), train_df[target]
    X_eval, y_eval = eval_df.drop(columns=[target]), eval_df[target]
    return X_train, y_train, X_eval, y_eval


def tune_model(
    train_path: Path | str = DEFAULT_TRAIN,
    eval_path: Path | str = DEFAULT_EVAL,
    model_output: Path | str = DEFAULT_OUT,
    n_trials: int = 10,  # Reduzi para ser mais rápido no teste
    experiment_name: str = "fish_weight_tuning",
):
    mlflow.set_experiment(experiment_name)
    X_train, y_train, X_eval, y_eval = _load_data(train_path, eval_path)

    def objective(trial: optuna.Trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "n_jobs": -1,
            "tree_method": "hist",
        }

        with mlflow.start_run(nested=True):
            model = XGBRegressor(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_eval)
            rmse = float(np.sqrt(mean_squared_error(y_eval, y_pred)))

            mlflow.log_params(params)
            mlflow.log_metric("rmse", rmse)

        return rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    print("✅ Best params:", study.best_trial.params)

    # Retreinar melhor modelo e salvar
    best_params = study.best_trial.params
    best_model = XGBRegressor(**best_params)
    best_model.fit(X_train, y_train)

    # Salvar localmente
    dump(best_model, model_output)
    print(f"✅ Best model saved to {model_output}")


if __name__ == "__main__":
    tune_model()

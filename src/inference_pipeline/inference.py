"""
Inference pipeline for Fish Market Weight Prediction.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from joblib import load

# Import preprocessing helper (agora simplificado)
from src.feature_pipeline.preprocess import preprocess_split
# Nota: Podemos usar preprocess_split logic ou reimplementar simples aqui para um único dict.
# Abaixo reimplemento a lógica simples de limpeza in-memory.

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "xgb_model.pkl"  # ou xgb_best_model
DEFAULT_TARGET_ENCODER = PROJECT_ROOT / "models" / "target_encoder.pkl"
TRAIN_FE_PATH = PROJECT_ROOT / "data" / "processed" / "feature_engineered_train.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "predictions.csv"

# Load feature columns
if TRAIN_FE_PATH.exists():
    _train_cols = pd.read_csv(TRAIN_FE_PATH, nrows=1)
    TRAIN_FEATURE_COLUMNS = [c for c in _train_cols.columns if c != "Weight"]
else:
    TRAIN_FEATURE_COLUMNS = None


def predict(
    input_df: pd.DataFrame,
    model_path: Path | str = DEFAULT_MODEL,
    target_encoder_path: Path | str = DEFAULT_TARGET_ENCODER,
) -> pd.DataFrame:
    df = input_df.copy()

    # 1. Preprocessamento Básico
    if "Species" in df.columns:
        df["Species"] = df["Species"].astype(str).str.strip()

    # 2. Feature Engineering (Target Encoding)
    if Path(target_encoder_path).exists() and "Species" in df.columns:
        target_encoder = load(target_encoder_path)
        df["Species_encoded"] = target_encoder.transform(df["Species"])
        df = df.drop(columns=["Species"], errors="ignore")

    # 3. Separate actuals if present
    y_true = None
    if "Weight" in df.columns:
        y_true = df["Weight"].tolist()
        df = df.drop(columns=["Weight"])

    # 4. Align columns
    if TRAIN_FEATURE_COLUMNS is not None:
        df = df.reindex(columns=TRAIN_FEATURE_COLUMNS, fill_value=0)

    # 5. Predict
    model = load(model_path)
    preds = model.predict(df)

    # 6. Build output
    out = input_df.copy()  # Return original data + prediction
    out["predicted_weight"] = preds

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL))

    args = parser.parse_args()

    raw_df = pd.read_csv(args.input)
    preds_df = predict(raw_df, model_path=args.model)
    preds_df.to_csv(args.output, index=False)
    print(f"✅ Predictions saved to {args.output}")

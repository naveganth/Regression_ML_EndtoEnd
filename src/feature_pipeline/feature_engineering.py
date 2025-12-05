"""
Feature engineering: Target Encoding for Species.
"""

from pathlib import Path
import pandas as pd
from category_encoders import TargetEncoder
from joblib import dump

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def target_encode(train: pd.DataFrame, eval: pd.DataFrame, col: str, target: str):
    """
    Use TargetEncoder on `col`.
    Fit on TRAIN, transform on EVAL.
    """
    te = TargetEncoder(cols=[col])
    encoded_col = f"{col}_encoded"

    # Fit transform no treino usando o alvo (Weight)
    train[encoded_col] = te.fit_transform(train[col], train[target])

    # Transform no eval (sem olhar o alvo, para não vazar dados)
    eval[encoded_col] = te.transform(eval[col])

    return train, eval, te


def drop_unused_columns(df: pd.DataFrame):
    # Remove a coluna original de texto 'Species', pois já temos a codificada
    return df.drop(columns=["Species"], errors="ignore")


def run_feature_engineering(
    in_train_path: Path | str | None = None,
    in_eval_path: Path | str | None = None,
    in_holdout_path: Path | str | None = None,
    output_dir: Path | str = PROCESSED_DIR,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if in_train_path is None:
        in_train_path = PROCESSED_DIR / "cleaning_train.csv"
    if in_eval_path is None:
        in_eval_path = PROCESSED_DIR / "cleaning_eval.csv"
    if in_holdout_path is None:
        in_holdout_path = PROCESSED_DIR / "cleaning_holdout.csv"

    train_df = pd.read_csv(in_train_path)
    eval_df = pd.read_csv(in_eval_path)
    holdout_df = pd.read_csv(in_holdout_path)

    # --- Target Encoding para Species ---
    target_encoder = None
    if "Species" in train_df.columns:
        # Importante: o target é 'Weight'
        train_df, eval_df, target_encoder = target_encode(
            train_df, eval_df, "Species", "Weight"
        )

        # Aplicar no holdout
        holdout_df["Species_encoded"] = target_encoder.transform(holdout_df["Species"])

        # Salvar encoder para usar na inferência
        dump(target_encoder, MODELS_DIR / "target_encoder.pkl")
        print("✅ Target Encoder (Species) saved.")

    # Drop da coluna categórica original
    train_df = drop_unused_columns(train_df)
    eval_df = drop_unused_columns(eval_df)
    holdout_df = drop_unused_columns(holdout_df)

    # Save
    train_df.to_csv(output_dir / "feature_engineered_train.csv", index=False)
    eval_df.to_csv(output_dir / "feature_engineered_eval.csv", index=False)
    holdout_df.to_csv(output_dir / "feature_engineered_holdout.csv", index=False)

    print("✅ Feature engineering complete.")
    print("   Train shape:", train_df.shape)
    print("   Eval  shape:", eval_df.shape)

    return train_df, eval_df, holdout_df, target_encoder


if __name__ == "__main__":
    run_feature_engineering()

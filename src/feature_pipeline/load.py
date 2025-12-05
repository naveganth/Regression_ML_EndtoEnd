"""
Load & split the Fish Market dataset.
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data/raw")


def load_and_split_data(
    raw_path: str = "data/raw/Fish.csv",
    output_dir: Path | str = DATA_DIR,
):
    """Load raw dataset, split into train/eval/holdout, and save to output_dir."""
    df = pd.read_csv(raw_path)

    # Filtrar dados inválidos (ex: peso zero ou negativo)
    df = df[df["Weight"] > 0]

    # Split: 60% Train, 20% Eval, 20% Holdout
    # Estratificar por Species garante que todas as espécies apareçam em todos os sets
    train_df, temp_df = train_test_split(
        df, test_size=0.4, random_state=42, stratify=df["Species"]
    )
    eval_df, holdout_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df["Species"]
    )

    # Save
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(outdir / "train.csv", index=False)
    eval_df.to_csv(outdir / "eval.csv", index=False)
    holdout_df.to_csv(outdir / "holdout.csv", index=False)

    print(f"✅ Data split completed (saved to {outdir}).")
    print(
        f"   Train: {train_df.shape}, Eval: {eval_df.shape}, Holdout: {holdout_df.shape}"
    )

    return train_df, eval_df, holdout_df


if __name__ == "__main__":
    load_and_split_data()

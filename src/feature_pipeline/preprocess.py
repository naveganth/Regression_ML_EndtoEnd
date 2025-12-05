"""
Preprocessing Script for Fish Market Regression.
- Reads train/eval/holdout CSVs from data/raw/.
- Basic cleaning and type enforcement.
- Saves cleaned splits to data/processed/.
"""

from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def preprocess_split(
    split: str,
    raw_dir: Path | str = RAW_DIR,
    processed_dir: Path | str = PROCESSED_DIR,
) -> pd.DataFrame:
    """Run preprocessing for a split and save to processed_dir."""
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    path = raw_dir / f"{split}.csv"
    df = pd.read_csv(path)

    # Garante que Species é string (para o encoder funcionar bem depois)
    if "Species" in df.columns:
        df["Species"] = df["Species"].astype(str)
        # Remove espaços extras se houver
        df["Species"] = df["Species"].str.strip()

    # Se houver duplicatas exatas, remove
    df = df.drop_duplicates()

    out_path = processed_dir / f"cleaning_{split}.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Preprocessed {split} saved to {out_path} ({df.shape})")
    return df


def run_preprocess(
    splits: tuple[str, ...] = ("train", "eval", "holdout"),
    raw_dir: Path | str = RAW_DIR,
    processed_dir: Path | str = PROCESSED_DIR,
):
    for s in splits:
        preprocess_split(s, raw_dir=raw_dir, processed_dir=processed_dir)


if __name__ == "__main__":
    run_preprocess()

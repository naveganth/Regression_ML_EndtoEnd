import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.feature_pipeline.feature_engineering import run_feature_engineering
from src.feature_pipeline.preprocess import preprocess_split


# Dados fake para teste rápido (sem depender do CSV real)
@pytest.fixture
def sample_raw_data(tmp_path):
    df = pd.DataFrame(
        {
            "Species": ["Bream", "Roach", "Bream", "Pike"],
            "Weight": [242.0, 160.0, 340.0, 200.0],
            "Length1": [23.2, 21.1, 26.8, 30.0],
            "Length2": [25.4, 22.5, 29.7, 32.5],
            "Length3": [30.0, 25.0, 34.5, 36.0],
            "Height": [11.52, 6.4, 12.4, 5.6],
            "Width": [4.02, 3.3, 4.7, 3.5],
        }
    )

    # Salvar como se fossem os arquivos brutos
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    df.to_csv(raw_dir / "train.csv", index=False)
    df.to_csv(raw_dir / "eval.csv", index=False)
    df.to_csv(raw_dir / "holdout.csv", index=False)

    return raw_dir


def test_feature_engineering_pipeline(sample_raw_data, tmp_path):
    """Teste de integração: Preprocess -> Feature Engineering"""
    processed_dir = tmp_path / "processed"

    # 1. Rodar Preprocessamento
    preprocess_split("train", raw_dir=sample_raw_data, processed_dir=processed_dir)
    preprocess_split("eval", raw_dir=sample_raw_data, processed_dir=processed_dir)
    preprocess_split("holdout", raw_dir=sample_raw_data, processed_dir=processed_dir)

    assert (processed_dir / "cleaning_train.csv").exists()

    # 2. Rodar Engenharia de Features
    train_df, eval_df, _, encoder = run_feature_engineering(
        in_train_path=processed_dir / "cleaning_train.csv",
        in_eval_path=processed_dir / "cleaning_eval.csv",
        in_holdout_path=processed_dir / "cleaning_holdout.csv",
        output_dir=processed_dir,
    )

    # Asserts Específicos para Peixes
    assert "Species_encoded" in train_df.columns, "A coluna Species não foi codificada!"
    assert "Species" not in train_df.columns, (
        "A coluna original Species devia ter sido removida."
    )
    assert "Weight" in train_df.columns, "O target Weight desapareceu."

    # Verificar se não há nulos gerados pelo encoder
    assert not train_df["Species_encoded"].isnull().any()

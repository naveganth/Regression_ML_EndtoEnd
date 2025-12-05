from fastapi import FastAPI, HTTPException
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import traceback

# Import inference pipeline
from src.inference_pipeline.inference import predict

app = FastAPI(title="Fish Weight Prediction API")

# Caminhos (ajustados para o padrão do container)
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "xgb_model.pkl"  # O nome gerado pelo train.py
ENCODER_PATH = BASE_DIR / "models" / "target_encoder.pkl"


@app.get("/")
def root():
    return {"message": "Fish Weight Prediction API is running 🐟",
     "local":"Verifique o link abaixo para a documentação do FastAPI",
     "url": "http://localhost:8000/docs"}


@app.get("/health")
def health():
    # Verifica se os artefatos existem
    model_exists = MODEL_PATH.exists()
    encoder_exists = ENCODER_PATH.exists()

    status = "healthy" if model_exists and encoder_exists else "unhealthy"

    return {
        "status": status,
        "model_path": str(MODEL_PATH),
        "model_exists": model_exists,
        "encoder_path": str(ENCODER_PATH),
        "encoder_exists": encoder_exists,
    }


@app.post("/predict")
def predict_batch(data: List[dict]):
    """
    Recebe uma lista de peixes e retorna o peso previsto.
    Exemplo:
    [
      {
        "Species": "Perch",
        "Length1": 20.0,
        "Length2": 22.0,
        "Length3": 23.5,
        "Height": 5.5,
        "Width": 3.3
      }
    ]
    """
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Model not found at {MODEL_PATH}. Did you run training?",
        )

    try:
        # Converter input para DataFrame
        df = pd.DataFrame(data)
        if df.empty:
            raise HTTPException(status_code=400, detail="No data provided")

        # Rodar inferência (passando caminhos explícitos)
        preds_df = predict(df, model_path=MODEL_PATH, target_encoder_path=ENCODER_PATH)

        return {"predictions": preds_df["predicted_weight"].astype(float).tolist()}

    except Exception as e:
        # Imprime o erro no log do Docker e retorna 500 detalhado
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

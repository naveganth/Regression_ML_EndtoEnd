# Fish Weight Prediction MLOps 🐟

Este projeto implementa um pipeline End-to-End de Machine Learning para prever o peso de peixes com base em suas medidas físicas. O projeto segue boas práticas de MLOps, incluindo versionamento de código, containerização (Docker) e testes.

## 📌 O Problema

O objetivo é utilizar regressão polinomial/linear (via XGBoost) para estimar a variável `Weight` (peso) com base em features como `Species`, `Length`, `Height` e `Width`.

## 🏗 Arquitetura

A solução está modularizada em:

1.  **Feature Pipeline:** Ingestão (`load.py`), Limpeza (`preprocess.py`) e Engenharia de Features (`feature_engineering.py`).
2.  **Training Pipeline:** Treinamento do modelo XGBoost (`train.py`) e log de métricas.
3.  **Inference Pipeline:** API REST (`FastAPI`) servindo o modelo.
4.  **DevOps:** Dockerfile para containerização e Makefile para automação.

## 🚀 Como Executar

### Pré-requisitos

- Python 3.11+
- Docker (opcional)

### 1. Instalação Local

```bash
pip install uv
uv sync
# Ou use pip install -r requirements.txt se preferir
```

2. Treinamento do Modelo

Execute o pipeline completo para gerar o modelo (models/xgb_model.pkl):

```bash
make train
# Ou manualmente: python src/training_pipeline/train.py (após rodar os anteriores)
```

3. Executando a API

```bash
make run-api
```

Acesse a documentação em: http://localhost:8000/docs 4. Docker (Deploy)

Para construir e rodar a aplicação em container:

```bash
# Constrói a imagem
make docker-build

# Roda o container na porta 8000
make docker-run
```

🧪 Testes

Para executar os testes unitários da API e validação de dados:

```bash
make test
```

🛠 Tecnologias

Linguagem: Python 3.11

ML: XGBoost, Scikit-Learn, Pandas

API: FastAPI, Uvicorn

Infra: Docker

Gerenciamento: Makefile, UV

---

### Passo a Passo Final para Corrigir o Erro 500

1.  **Atualize o arquivo `src/api/main.py`** com o código do passo 1 acima.
2.  **Garanta que o modelo existe:** Rode `make train` (ou os scripts python manualmente) na sua máquina **antes** de construir o Docker. O Dockerfile copia os arquivos da sua pasta atual. Se a pasta `models/` estiver vazia na sua máquina, ela estará vazia no Docker.
    - _Verifique:_ Você deve ter um arquivo `models/xgb_model.pkl` e `models/target_encoder.pkl`.
3.  **Reconstrua o Docker:**
    ```bash
    docker build -t fish-predictor .
    ```
4.  **Rode novamente:**
    ```bash
    docker run -p 8000:8000 fish-predictor
    ```

Agora, ao enviar o `curl` ou usar o Swagger, a API deve encontrar o modelo e funcionar corret

---

---

---

## Housing ML end2end Project

## Project Overview

Housing Regression MLE is an end-to-end machine learning pipeline for predicting housing prices using XGBoost. The project follows ML engineering best practices with modular pipelines, experiment tracking via MLflow, containerization, AWS cloud deployment, and comprehensive testing. The system includes both a REST API and a Streamlit dashboard for interactive predictions.

## Architecture

The codebase is organized into distinct pipelines following the flow:
`Load → Preprocess → Feature Engineering → Train → Tune → Evaluate → Inference → Batch → Serve`

### Core Modules

- **`src/feature_pipeline/`**: Data loading, preprocessing, and feature engineering

  - `load.py`: Time-aware data splitting (train <2020, eval 2020-21, holdout ≥2022)
  - `preprocess.py`: City normalization, deduplication, outlier removal
  - `feature_engineering.py`: Date features, frequency encoding (zipcode), target encoding (city_full)

- **`src/training_pipeline/`**: Model training and hyperparameter optimization

  - `train.py`: Baseline XGBoost training with configurable parameters
  - `tune.py`: Optuna-based hyperparameter tuning with MLflow integration
  - `eval.py`: Model evaluation and metrics calculation

- **`src/inference_pipeline/`**: Production inference

  - `inference.py`: Applies same preprocessing/encoding transformations using saved encoders

- **`src/batch/`**: Batch prediction processing

  - `run_monthly.py`: Generates monthly predictions on holdout data

- **`src/api/`**: FastAPI web service
  - `main.py`: REST API with S3 integration, health checks, prediction endpoints, and batch processing

### Web Applications

- **`app.py`**: Streamlit dashboard for interactive housing price predictions
  - Real-time predictions via FastAPI integration
  - Interactive filtering by year, month, and region
  - Visualization of predictions vs actuals with metrics (MAE, RMSE, % Error)
  - Yearly trend analysis with highlighted selected periods

### Cloud Infrastructure & Deployment

- **AWS S3 Integration**: Data and model storage in `housing-regression-data` bucket
- **Amazon ECR**: Container registry for Docker images
- **Amazon ECS**: Container orchestration with Fargate
- **Application Load Balancer**: Traffic distribution and routing
- **CI/CD Pipeline**: Automated deployment via GitHub Actions

#### ECS Services:

- **housing-api-service**: FastAPI backend (port 8000, 1024 CPU, 3072 MB memory)
- **housing-streamlit-service**: Streamlit dashboard (port 8501, 512 CPU, 1024 MB memory)

### Data Leakage Prevention

The project implements strict data leakage prevention:

- Time-based splits (not random)
- Encoders fitted only on training data
- Leakage-prone columns dropped before training
- Schema alignment enforced between train/eval/inference

## Common Commands

### Environment Setup

```bash
# Install dependencies using uv
uv sync
```

### Testing

```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/test_features.py
pytest tests/test_training.py
pytest tests/test_inference.py

# Run with verbose output
pytest -v
```

### Data Pipeline

```bash
# 1. Load and split raw data
python src/feature_pipeline/load.py

# 2. Preprocess splits
python -m src.feature_pipeline.preprocess

# 3. Feature engineering
python -m src.feature_pipeline.feature_engineering
```

### Training Pipeline

```bash
# Train baseline model
python src/training_pipeline/train.py

# Hyperparameter tuning with MLflow
python src/training_pipeline/tune.py

# Model evaluation
python src/training_pipeline/eval.py
```

### Inference

```bash
# Single inference
python src/inference_pipeline/inference.py --input data/raw/holdout.csv --output predictions.csv

# Batch monthly predictions
python src/batch/run_monthly.py
```

### API Service

```bash
# Start FastAPI server locally
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Streamlit Dashboard

```bash
# Start Streamlit dashboard locally
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### Docker

```bash
# Build API container
docker build -t housing-regression .

# Build Streamlit container
docker build -t housing-streamlit -f Dockerfile.streamlit .

# Run API container
docker run -p 8000:8000 housing-regression

# Run Streamlit container
docker run -p 8501:8501 housing-streamlit
```

### MLflow Tracking

```bash
# Start MLflow UI (view experiments)
mlflow ui
```

## Key Design Patterns

### Pipeline Modularity

Each pipeline component can be run independently with consistent interfaces. All modules accept configurable input/output paths for testing isolation.

### Cloud-Native Architecture

- **S3-First Storage**: Models and data automatically sync from S3 buckets
- **Containerized Services**: Both API and dashboard run in Docker containers
- **Auto-scaling Infrastructure**: ECS Fargate provides serverless container scaling
- **Environment-based Configuration**: Separate configs for local development and production

### Encoder Persistence

Frequency and target encoders are saved as pickle files during training and loaded during inference to ensure consistent transformations.

### Configuration Management

Model parameters, file paths, and pipeline settings use sensible defaults but can be overridden through function parameters or environment variables. Production deployments use AWS environment variables.

### Testing Strategy

- Unit tests for individual pipeline components
- Integration tests for end-to-end pipeline flows
- Smoke tests for inference pipeline
- All tests use temporary directories to avoid touching production data

## Dependencies

Key production dependencies (see `pyproject.toml`):

- **ML/Data**: `xgboost==3.0.4`, `scikit-learn`, `pandas==2.1.1`, `numpy==1.26.4`
- **API**: `fastapi`, `uvicorn`
- **Dashboard**: `streamlit`, `plotly`
- **Cloud**: `boto3` (AWS integration)
- **Experimentation**: `mlflow`, `optuna`
- **Quality**: `great-expectations`, `evidently`

## File Structure Notes

- **`data/`**: Raw, processed, and prediction data (time-structured, S3-synced)
- **`models/`**: Trained models and encoders (pkl files, S3-synced)
- **`mlruns/`**: MLflow experiment tracking data
- **`configs/`**: YAML configuration files
- **`notebooks/`**: Jupyter notebooks for EDA and experimentation
- **`tests/`**: Comprehensive test suite with sample data
- **AWS Task Definitions**: `housing-api-task-def.json`, `streamlit-task-def.json`
- **CI/CD**: `.github/workflows/ci.yml` for automated deployment

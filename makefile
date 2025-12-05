# Como rodar os comandos abaixo: Make {nome do comando} (lembre-se de usar no terminal.)

# Instalação de dependências usando uv
install:
	pip install uv
	uv sync

# Pipeline completo de treino (do carregamento ao modelo final)
train:
	uv run python src/feature_pipeline/load.py
	uv run python src/feature_pipeline/preprocess.py
	uv run python src/feature_pipeline/feature_engineering.py
	uv run python src/training_pipeline/train.py

# Rodar a API localmente
run-api:
	uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Rodar o Streamlit localmente
run-app:
	uv run streamlit run app.py

# Build e Run do Docker (com limpeza de containers antigos para evitar conflito de porta)
docker-auto:
	docker build -t fish-predictor .
	# O comando abaixo tenta parar e remover container antigo se existir (ignora erro se não existir)
	-docker rm -f fish-predictor-container 2> NUL || true
	docker run --name fish-predictor-container -p 8000:8000 fish-predictor

run-mlflow:
	uv run mlflow ui

# Rodar testes (usando python -m para resolver imports)
test:
	uv run python -m pytest
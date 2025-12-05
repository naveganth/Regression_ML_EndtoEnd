# 🐟 Fish Weight Prediction - End-to-End MLOps

Este projeto é uma solução completa de Machine Learning **para prever o peso de peixes com base em medidas físicas**. O objetivo foi demonstrar boas práticas de MLOps, desde a engenharia de dados até ao deploy de uma API escalável e interface de utilizador.

## 🏆 Requisitos

### 1\. Python + Machine Learning (Obrigatório)

- **Modelo:** Utilizei **XGBoost Regressor**, escolhido pela sua performance em dados tabulares.
- **Separação:** Separação de responsabilidades em módulos Python:
  - `src/feature_pipeline`: Ingestão e limpeza.
  - `src/training_pipeline`: Treino e avaliação.
  - `src/inference_pipeline`: Lógica de predição para produção.

### 2\. Pipeline de MLOps (Obrigatório)

- **Versionamento:** Integração completa com **MLflow** para registar parâmetros, métricas (MAE, RMSE, R²) e artefatos do modelo (`.pkl`).
- **Orquestração:** Scripts organizados que podem ser executados individualmente ou encadeados via Makefile.

### 3\. Deploy em Container (Obrigatório)

- **API:** Desenvolvida em **FastAPI** para alta performance.
- **Docker:** A solução é entregue containerizada. O Dockerfile constrói um ambiente isolado com todas as dependências geridas pelo `uv`.

### 4\. Diferenciais Implementados (Opcionais)

- ✅ **Testes Unitários:** Cobertura de testes com `pytest` para garantir a integridade da API e do schema de dados.
- ✅ **CI/CD:** Pipeline no GitHub Actions que roda testes e build automático a cada push.
- ✅ **Makefile:** Automação de comandos complexos para facilitar a execução.
- ✅ **Visualização:** Aplicação Fullstack com Streamlit para demonstração interativa.
- ✅ **Model Registry:** Uso de **MLflow** para registrar e versionar oficialmente o modelo como `FishWeightPredictor`.

## 🏗 Arquitetura da Solução

O projeto está modularizado em diretórios específicos:

- **Feature Pipeline:** Ingestão, limpeza e transformação dos dados (`src/feature_pipeline`).
- **Training Pipeline:** Treino do modelo XGBoost com rastreamento via MLflow (`src/training_pipeline`).
- **Inference:** API REST (`src/api`) e lógica de inferência (`src/inference_pipeline`).
- **Frontend:** Interface com Streamlit (`src/app.py`).
- **DevOps:** Configurações de Docker, Makefile e CI/CD.

## 📂 Estrutura do Projeto

```text
├── .github/workflows  # Pipeline de CI (Testes e Build)
├── data/              # Dados brutos e processados
├── models/            # Artefatos do modelo (.pkl)
├── src/
│   ├── api/           # Código da API (FastAPI)
│   ├── feature_.../   # Scripts de processamento
│   ├── training_.../  # Scripts de treino e tuning
│   └── app.py         # Frontend Streamlit
├── tests/             # Testes unitários e de integração
├── Dockerfile         # Configuração da imagem da API
├── Makefile           # Comandos rápidos de execução
└── pyproject.toml     # Dependências (gerenciado pelo uv)
```

## 🚀 Como Executar

### Pré-requisitos

- **Docker** (Recomendado para execução isolada)
- Ou **Python 3.11+** com `uv` instalado para execução local.

---

### Opção 1: Via Docker (Recomendado)

Esta opção sobe a API pronta para uso sem instalar nada no seu Python local.

**1. Construir e Rodar a API:**
Isso irá construir a imagem, remover containers antigos e iniciar a API na porta 8000.

```bash
make docker-auto
```

**2. MLflow**

O projeto utiliza MLflow não apenas para rastreamento de métricas, mas também como **Model Registry**.

Para visualizar o catálogo de modelos:

1. Execute o comando de interface:

   ```bash
   make run-mlflow
   ```

2. Acesse a http://127.0.0.1:5000.

3. Clique na aba "Models" no topo da página.

4. Verá o modelo FishWeightPredictor com todas as suas versões (v1, v2, etc.) e estágios de produção.

**3. Testar a API:**

- Acesse a documentação interativa (Swagger): [http://localhost:8000/docs](https://www.google.com/search?q=http://localhost:8000/docs)
- Ou veja a secção **"Como Realizar a Inferência"** abaixo.

---

### Opção 2: Execução Local (Desenvolvimento)

Se preferir rodar os scripts manualmente:

**1. Instalar dependências:**

```bash
pip install uv
make install
```

**2. Treinar o Modelo:**
Executa o pipeline completo (Load -\> Preprocess -\> Feature Eng -\> Train). O modelo será salvo em `models/xgb_model.pkl` e as métricas registadas no MLflow.

```bash
make train
```

**3. Rodar a API:**

```bash
make run-api
```

**4. Rodar o Dashboard (Streamlit):**
Para visualizar uma interface gráfica amigável:

```bash
make run-app
```

- Acesse em: [http://localhost:8501](https://www.google.com/search?q=http://localhost:8501)

## 📡 Como Realizar a Inferência

A API aceita requisições POST no endpoint `/predict`.

**Exemplo de Payload (JSON):**

```json
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
```

**Comando cURL:**

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '[{"Species": "Perch", "Length1": 20.0, "Length2": 22.0, "Length3": 23.5, "Height": 5.5, "Width": 3.3}]'
```

**Resposta Esperada:**

```json
{
  "predictions": [245.32]
}
```

## 🔮 Possíveis Melhorias

Pontos identificados para evolução futura do projeto:

- **Monitorização de Drift:** Integração com EvidentlyAI para alertar se os peixes na inferência tiverem medidas muito diferentes do treino.
- **Deploy em Cloud:** Configuração de deploy contínuo (CD) para AWS ECS ou Lambda utilizando Terraform.
- **Feature Store:** Para um cenário com milhões de registos, implementar uma Feature Store (ex: Feast) para servir features pré-calculadas.
- **Autenticação:** Adicionar camada de segurança (OAuth2) na API.

---

**Autor:** Lucas Paulo de Souza Navegante
**Créditos:** _anesriad/Regression_ML_EndtoEnd_ que foi o modelo base para este projeto.
**Data:** 04/12/2025

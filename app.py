import streamlit as st
import requests
import pandas as pd

# Configuração da Página
st.set_page_config(page_title="Fish Weight Predictor 🐟", layout="centered")

st.title("🐟 Previsão de Peso de Peixes")
st.markdown("Insira as medidas do peixe abaixo para estimar o seu peso.")

# URL da API (assumindo que está a correr no Docker/Local na porta 8000)
API_URL = "http://localhost:8000/predict"

# Formulário de Input
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        species = st.selectbox(
            "Espécie",
            ["Bream", "Roach", "Whitefish", "Parkki", "Perch", "Pike", "Smelt"],
        )
        length1 = st.number_input(
            "Comprimento Vertical (cm)", min_value=0.0, value=20.0
        )
        length2 = st.number_input(
            "Comprimento Diagonal (cm)", min_value=0.0, value=22.0
        )

    with col2:
        length3 = st.number_input("Comprimento Cruzado (cm)", min_value=0.0, value=23.5)
        height = st.number_input("Altura (cm)", min_value=0.0, value=5.5)
        width = st.number_input("Largura (cm)", min_value=0.0, value=3.3)

    submitted = st.form_submit_button("Calcular Peso ⚖️")

if submitted:
    # Payload igual ao esperado pela API
    payload = [
        {
            "Species": species,
            "Length1": length1,
            "Length2": length2,
            "Length3": length3,
            "Height": height,
            "Width": width,
        }
    ]

    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            weight = response.json()["predictions"][0]
            st.success(f"🐟 Peso Estimado: **{weight:.2f} g**")
        else:
            st.error(f"Erro na API: {response.text}")
    except Exception as e:
        st.error(
            f"Não foi possível conectar à API. Verifique se o Docker está a correr. Erro: {e}"
        )

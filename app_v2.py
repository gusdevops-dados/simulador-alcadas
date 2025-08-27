import streamlit as st
import pandas as pd

st.set_page_config(page_title="Simulador de Backtest de Alçadas", layout="wide")
st.title("📊 Simulador de Backtest de Alçadas")

# =====================
# Upload da base
# =====================
uploaded_file = st.file_uploader("Suba a base histórica (CSV ou Excel)", type=["csv", "xlsx"])

if uploaded_file:
    # Ler base
    if uploaded_file.name.endswith("csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("✅ Base carregada com sucesso!")
    st.write("Prévia da base:", df.head())

    # =====================
    # Parâmetros de regras
    # =====================
    st.subheader("⚙️ Defina os parâmetros da alçada")

    min_invest = st.slider("Investimento mínimo (vlr_invest)", 0, int(df["vlr_invest"].max()), 500)
    max_valor = st.slider("Valor máximo contestado (vlr_contest)", 0, int(df["vlr_contest"].max()), 1000)
    vip_limite = st.slider("Limite especial para cliente VIP", 0, int(df["vlr_contest"].max()), 2000)

    # =====================
    # Função de aprovação
    # =====================
    def aplica_regras(row):
        # Regra de vulnerabilidade prioritária
        if row["vulnerabilidade"] == "alta":
            return True

        # Regra cliente VIP
        if row["client_vip"] == "s" and row["vlr_contest"] <= vip_limite:
            return True

        # Regras gerais
        if (
            row["vlr_invest"] >= min_invest and
            row["vlr_contest"] <= max_valor and
            row["bom_pagador"] == "s"
        ):
            return True

        return False

    # Aplicar regras
    df["aprovado"] = df.apply(aplica_regras, axis=1)

    # =====================
    # Resultados
    # =====================
    taxa_aprov = df["aprovado"].mean() * 100
    st.metric("Taxa de aprovação", f"{taxa_aprov:.2f}%")

    st.write("📋 Resultados detalhados:", df.head(20))

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Simulador de Alçadas", layout="wide")
st.title("📊 Simulador Genérico de Políticas e Alçadas")

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

    st.success("Base carregada com sucesso!")
    st.write("Prévia da base:", df.head())

    # =====================
    # Seleção das variáveis
    # =====================
    variaveis_escolhidas = st.multiselect(
        "Escolha as variáveis que deseja usar nas regras da alçada:",
        options=df.columns
    )

    parametros = {}
    if variaveis_escolhidas:
        st.subheader("⚙️ Defina os parâmetros para cada variável")

        for var in variaveis_escolhidas:
            col = df[var]

            if pd.api.types.is_numeric_dtype(col):
                min_val, max_val = float(col.min()), float(col.max())
                parametros[var] = st.slider(
                    f"Valor mínimo para {var}",
                    min_val, max_val, min_val
                )

            elif pd.api.types.is_bool_dtype(col):
                parametros[var] = st.checkbox(f"Considerar apenas {var}=True?")

            else:  # categórica (string/object)
                valores_unicos = col.dropna().unique().tolist()
                parametros[var] = st.multiselect(
                    f"Valores permitidos para {var}",
                    valores_unicos,
                    default=valores_unicos
                )

        # =====================
        # Aplicação das regras
        # =====================
        def aplica_regras(row, parametros):
            for var, regra in parametros.items():
                valor = row[var]

                if isinstance(regra, (int, float)):  # numérico
                    if valor < regra:
                        return False

                elif isinstance(regra, list):  # categórico
                    if valor not in regra:
                        return False

                elif isinstance(regra, bool):  # booleano
                    if regra and not valor:
                        return False

            return True

        df["aprovado"] = df.apply(lambda r: aplica_regras(r, parametros), axis=1)

        # =====================
        # Resultados
        # =====================
        taxa_aprov = df["aprovado"].mean() * 100
        st.metric("Taxa de aprovação", f"{taxa_aprov:.2f}%")

        st.write("📋 Resultados detalhados:", df.head(20))

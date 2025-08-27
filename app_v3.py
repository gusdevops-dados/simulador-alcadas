# app.py
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Simulador Genérico de Alçadas", layout="wide")
st.title("📊 Simulador Genérico de Políticas e Alçadas")

st.markdown(
    "Carregue uma base histórica, escolha as variáveis e defina regras sem código. "
    "Você pode combinar condições numéricas, categóricas e booleanas, alterar agregação (AND/OR) "
    "e aplicar overrides (auto-negar / auto-aprovar)."
)

# =========================
# Upload
# =========================
uploaded_file = st.file_uploader("Suba a base (CSV ou Excel)", type=["csv", "xlsx"])
if not uploaded_file:
    st.info("⏫ Envie um arquivo para começar.")
    st.stop()

# Leitura
try:
    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Erro ao ler arquivo: {e}")
    st.stop()

if df.empty:
    st.warning("A base está vazia.")
    st.stop()

st.success("✅ Base carregada!")
with st.expander("🔍 Prévia da base", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)

# =========================
# Inferência de tipos (simples) e utilitários
# =========================
def is_bool_series(s: pd.Series) -> bool:
    # Verdadeiro booleano
    if pd.api.types.is_bool_dtype(s):
        return True
    # Strings representando booleanos (ex: 's'/'n', 'true'/'false', 'sim'/'nao')
    if pd.api.types.is_object_dtype(s):
        uniques = set(str(x).strip().lower() for x in s.dropna().unique())
        candidates = {"s", "n", "sim", "não", "nao", "true", "false", "0", "1"}
        # Se todos os valores (ignorando NaN) estão contidos no conjunto candidato e há <= 2 valores distintos
        if len(uniques) <= 2 and uniques.issubset(candidates):
            return True
    return False

def normalize_bool(v):
    if isinstance(v, bool):
        return v
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip().lower()
    if s in {"s", "sim", "true", "1"}:
        return True
    if s in {"n", "nao", "não", "false", "0"}:
        return False
    return None  # não forçamos conversão

numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
bool_cols    = [c for c in df.columns if is_bool_series(df[c])]
# categóricas: object ou category que não foram classificadas como booleanas
cat_cols     = [c for c in df.columns
                if (pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c]))
                and c not in bool_cols]

# =========================
# Seletor de variáveis e agregação
# =========================
st.sidebar.header("⚙️ Configuração de Regras")
selected_cols = st.sidebar.multiselect(
    "Quais variáveis entram nas regras?",
    options=numeric_cols + bool_cols + cat_cols,
    default=[]
)

logic_mode = st.sidebar.radio("Agregação entre condições", ["AND (todas)", "OR (qualquer)"], index=0)

st.sidebar.caption(
    "Dica: adicione condições para variáveis numéricas, selecione valores permitidos para categóricas "
    "e escolha como tratar booleanas."
)

# =========================
# Overrides opcionais
# =========================
st.sidebar.subheader("🚦 Overrides (opcional)")
ov_cat_candidates = bool_cols + cat_cols  # só faz sentido para colunas discretas
override_deny_col = st.sidebar.selectbox("Auto-NEGAR por coluna (opcional)", ["(nenhum)"] + ov_cat_candidates)
override_deny_vals = []
if override_deny_col != "(nenhum)":
    vals = df[override_deny_col].dropna().astype(str).unique().tolist()
    override_deny_vals = st.sidebar.multiselect(f"Valores que serão sempre NEGADOS em {override_deny_col}", vals)

override_allow_col = st.sidebar.selectbox("Auto-APROVAR por coluna (opcional)", ["(nenhum)"] + ov_cat_candidates)
override_allow_vals = []
if override_allow_col != "(nenhum)":
    vals = df[override_allow_col].dropna().astype(str).unique().tolist()
    override_allow_vals = st.sidebar.multiselect(f"Valores que serão sempre APROVADOS em {override_allow_col}", vals)

st.sidebar.caption("Prioridade: Auto-NEGA > Auto-APROVA > Regras.")

if not selected_cols and not override_deny_vals and not override_allow_vals:
    st.info("Selecione ao menos uma variável nas regras ou configure um override para continuar.")
    st.stop()

# =========================
# Controles por variável
# =========================
st.subheader("🧩 Parâmetros das Regras")

rule_widgets = {}

for col in selected_cols:
    with st.container(border=True):
        st.markdown(f"**{col}**")
        s = df[col]

        if col in numeric_cols:
            col_min = float(np.nanmin(s.values))
            col_max = float(np.nanmax(s.values))
            op = st.radio(
                f"Operador para `{col}`",
                ["≥ (mínimo)", "≤ (máximo)", "entre"],
                horizontal=True,
                key=f"op_{col}"
            )
            if op == "≥ (mínimo)":
                thr = st.slider(f"Valor mínimo", col_min, col_max, col_min, key=f"thr_ge_{col}")
                rule_widgets[col] = ("num_ge", thr)
            elif op == "≤ (máximo)":
                thr = st.slider(f"Valor máximo", col_min, col_max, col_max, key=f"thr_le_{col}")
                rule_widgets[col] = ("num_le", thr)
            else:  # entre
                rng = st.slider(
                    "Intervalo permitido",
                    col_min, col_max, (col_min, col_max),
                    key=f"thr_between_{col}"
                )
                rule_widgets[col] = ("num_between", rng)

        elif col in bool_cols:
            # mapa: exigir True, exigir False, ignorar? (como já selecionou a coluna, não faz sentido ignorar aqui)
            choice = st.radio(
                f"Como tratar `{col}`?",
                ["Exigir True", "Exigir False"],
                horizontal=True,
                key=f"bool_{col}"
            )
            rule_widgets[col] = ("bool_true" if choice == "Exigir True" else "bool_false", None)

        else:  # categórica
            # Listar valores únicos (com limite para performance)
            uniques = s.dropna().astype(str).unique().tolist()
            if len(uniques) > 200:
                st.warning(f"{col} possui muitos valores únicos ({len(uniques)}). Mostrando uma amostra de 200.")
                uniques = uniques[:200]
            sel = st.multiselect(
                f"Valores permitidos em `{col}`",
                options=sorted(uniques),
                default=sorted(uniques),
                key=f"cat_{col}"
            )
            rule_widgets[col] = ("cat_in", sel)

# =========================
# Avaliação das regras
# =========================
def row_matches_rules(row) -> bool:
    # 1) Overrides com prioridade
    if override_deny_col != "(nenhum)":
        v = row[override_deny_col]
        v = str(v) if pd.notna(v) else ""
        if v in set(override_deny_vals):
            return False

    if override_allow_col != "(nenhum)":
        v = row[override_allow_col]
        v = str(v) if pd.notna(v) else ""
        if v in set(override_allow_vals):
            return True

    # 2) Regras comuns
    results = []
    for col, (rtype, param) in rule_widgets.items():
        val = row[col]

        if rtype.startswith("num_"):
            if pd.isna(val):
                results.append(False)
                continue
            x = float(val)
            if rtype == "num_ge":
                results.append(x >= float(param))
            elif rtype == "num_le":
                results.append(x <= float(param))
            elif rtype == "num_between":
                lo, hi = param
                results.append((x >= float(lo)) & (x <= float(hi)))

        elif rtype.startswith("bool_"):
            nb = normalize_bool(val)
            if nb is None:
                results.append(False)  # valor ambíguo não passa
            else:
                if rtype == "bool_true":
                    results.append(nb is True)
                else:
                    results.append(nb is False)

        elif rtype == "cat_in":
            v = str(val) if pd.notna(val) else ""
            results.append(v in set(param))

        else:
            results.append(False)

    if not results:
        return True  # se não houver regra, mas havia override, já teria retornado acima

    if logic_mode.startswith("AND"):
        return all(results)
    else:
        return any(results)

with st.spinner("Aplicando regras..."):
    aprovado_series = df.apply(row_matches_rules, axis=1)
    out = df.copy()
    out["aprovado"] = aprovado_series

# =========================
# Resultados
# =========================
st.subheader("📈 Resultados")
taxa = float(aprovado_series.mean() * 100.0)
st.metric("Taxa de aprovação", f"{taxa:.2f}%")

cols = st.columns(2)
with cols[0]:
    st.caption("✅ Aprovados (amostra)")
    st.dataframe(out[out["aprovado"] == True].head(20), use_container_width=True)
with cols[1]:
    st.caption("❌ Negados (amostra)")
    st.dataframe(out[out["aprovado"] == False].head(20), use_container_width=True)

# Agregação opcional por uma coluna (ex.: impacto por segmento)
with st.expander("📊 Agregar resultados por coluna (opcional)"):
    group_col = st.selectbox("Agrupar por", ["(nenhum)"] + list(df.columns))
    if group_col != "(nenhum)":
        agg_df = out.groupby(group_col)["aprovado"].mean().mul(100).reset_index()
        agg_df.rename(columns={"aprovado": "taxa_aprovacao_%"}, inplace=True)
        st.dataframe(agg_df, use_container_width=True)

# Download
csv_bytes = out.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Baixar resultado (CSV)",
    data=csv_bytes,
    file_name="resultado_backtest.csv",
    mime="text/csv"
)

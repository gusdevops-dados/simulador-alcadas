# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Simulador de Alçadas – Cenários", layout="wide")
st.title("📊 Simulador de Políticas e Alçadas — Cenários A/B")

# -----------------------------
# Upload
# -----------------------------
uploaded_file = st.file_uploader("Suba a base (CSV ou Excel)", type=["csv", "xlsx"])
if not uploaded_file:
    st.info("⏫ Envie um arquivo para começar.")
    st.stop()

# Leitura
try:
    if uploaded_file.name.lower().endswith(".csv"):
        base = pd.read_csv(uploaded_file)
    else:
        base = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Erro ao ler arquivo: {e}")
    st.stop()

if base.empty:
    st.warning("A base está vazia.")
    st.stop()

st.success("✅ Base carregada!")
with st.expander("🔍 Prévia da base", expanded=False):
    st.dataframe(base.head(20), use_container_width=True)

if "vlr_contest" not in base.columns:
    st.warning("A coluna 'vlr_contest' não foi encontrada. O impacto financeiro será 0.")
    base["vlr_contest"] = 0.0

# -----------------------------
# Helpers de tipo
# -----------------------------
def is_bool_series(s: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(s):  # bool nativo
        return True
    if pd.api.types.is_object_dtype(s):
        vals = set(str(x).strip().lower() for x in s.dropna().unique())
        candidates = {"s", "n", "sim", "não", "nao", "true", "false", "0", "1"}
        return len(vals) <= 2 and vals.issubset(candidates)
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
    return None

numeric_cols = [c for c in base.columns if pd.api.types.is_numeric_dtype(base[c])]
bool_cols    = [c for c in base.columns if is_bool_series(base[c])]
cat_cols     = [c for c in base.columns
                if (pd.api.types.is_object_dtype(base[c]) or pd.api.types.is_categorical_dtype(base[c]))
                and c not in bool_cols]

# -----------------------------
# Componentes de UI por cenário
# -----------------------------
def build_rule_ui(df: pd.DataFrame, key_prefix: str):
    st.subheader(f"🧩 Regras — {key_prefix}")
    selected = st.multiselect(
        "Escolha as variáveis que entram nas regras:",
        options=numeric_cols + bool_cols + cat_cols,
        key=f"sel_{key_prefix}"
    )
    logic_mode = st.radio(
        "Agregação entre condições:",
        ["AND (todas)", "OR (qualquer)"],
        index=0, horizontal=True, key=f"logic_{key_prefix}"
    )

    rule_widgets = []
    for col in selected:
        with st.container(border=True):
            st.markdown(f"**{col}**")
            s = df[col]
            # Numéricas
            if col in numeric_cols:
                col_min = float(np.nanmin(s.values))
                col_max = float(np.nanmax(s.values))
                op = st.radio(
                    f"Operador para `{col}`",
                    ["≥ (mínimo)", "≤ (máximo)", "entre"],
                    horizontal=True,
                    key=f"{key_prefix}_op_{col}"
                )
                if op == "≥ (mínimo)":
                    thr = st.slider(f"Valor mínimo", col_min, col_max, col_min, key=f"{key_prefix}_thr_ge_{col}")
                    rule_widgets.append((col, "num_ge", thr))
                elif op == "≤ (máximo)":
                    thr = st.slider(f"Valor máximo", col_min, col_max, col_max, key=f"{key_prefix}_thr_le_{col}")
                    rule_widgets.append((col, "num_le", thr))
                else:
                    rng = st.slider(
                        "Intervalo permitido",
                        col_min, col_max, (col_min, col_max),
                        key=f"{key_prefix}_thr_between_{col}"
                    )
                    rule_widgets.append((col, "num_between", rng))

            # Booleanas
            elif col in bool_cols:
                choice = st.radio(
                    f"Como tratar `{col}`?",
                    ["Exigir True", "Exigir False"],
                    horizontal=True, key=f"{key_prefix}_bool_{col}"
                )
                rule_widgets.append((col, "bool_true" if choice == "Exigir True" else "bool_false", None))

            # Categóricas
            else:
                uniques = s.dropna().astype(str).unique().tolist()
                uniques = sorted(uniques[:500])  # corta pra não pesar
                sel = st.multiselect(
                    f"Valores permitidos em `{col}`",
                    options=uniques, default=uniques,
                    key=f"{key_prefix}_cat_{col}"
                )
                rule_widgets.append((col, "cat_in", sel))

    return rule_widgets, logic_mode

def apply_rules_row(row, rules, logic_mode):
    if not rules:
        return True
    results = []
    for (col, rtype, param) in rules:
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
                results.append((x >= float(lo)) and (x <= float(hi)))
        elif rtype.startswith("bool_"):
            nb = normalize_bool(val)
            if nb is None:
                results.append(False)
            else:
                results.append(nb is True if rtype == "bool_true" else nb is False)
        elif rtype == "cat_in":
            v = str(val) if pd.notna(val) else ""
            results.append(v in set(param))
        else:
            results.append(False)
    return all(results) if logic_mode.startswith("AND") else any(results)

def evaluate_scenario(df: pd.DataFrame, rules, logic_mode, rule_order_cols):
    out = df.copy()
    aprovado = out.apply(lambda r: apply_rules_row(r, rules, logic_mode), axis=1)
    out["aprovado"] = aprovado

    # Métricas
    pote_total = int(aprovado.sum())  # público que passou
    taxa = float(aprovado.mean() * 100.0)
    vlr_aprov = float(out.loc[out["aprovado"]==True, "vlr_contest"].sum())
    vlr_neg   = float(out.loc[out["aprovado"]==False, "vlr_contest"].sum())

    # Motivo principal de negativa (primeira regra que falha na ordem escolhida)
    # Apenas se AND; em OR o conceito de "primeira regra que falha" perde sentido
    motivo_col = None
    if rules and rule_order_cols and df.shape[0] > 0 and logic_mode.startswith("AND"):
        def first_fail(row):
            if apply_rules_row(row, rules, logic_mode):  # aprovado
                return "Aprovado"
            for col in rule_order_cols:
                # localiza regra(s) dessa coluna
                sub = [r for r in rules if r[0] == col]
                if sub:
                    if not apply_rules_row(row, sub, "AND (todas)"):
                        return f"Falha em {col}"
            return "Falha (outra)"
        motivo_col = out.apply(first_fail, axis=1)
        out["motivo_negativa"] = motivo_col
    else:
        out["motivo_negativa"] = np.where(out["aprovado"], "Aprovado", "Negado")

    return out, {
        "pote_total": pote_total,
        "taxa_aprovacao_pct": taxa,
        "vlr_aprovado": vlr_aprov,
        "vlr_negado": vlr_neg
    }

def plots_for_scenario(out_df: pd.DataFrame, scenario_name: str, rule_paths_df: pd.DataFrame|None):
    c1, c2 = st.columns(2)
    # Barra Aprovados x Negados
    with c1:
        vc = out_df["aprovado"].value_counts().rename({True: "Aprovado", False: "Negado"}).reset_index()
        vc.columns = ["status", "qtd"]
        fig = px.bar(vc, x="status", y="qtd", title=f"{scenario_name}: Aprovados vs Negados")
        st.plotly_chart(fig, use_container_width=True)

    # Histograma valores
    with c2:
        tmp = out_df.copy()
        tmp["status"] = np.where(tmp["aprovado"], "Aprovado", "Negado")
        fig2 = px.histogram(tmp, x="vlr_contest", color="status", barmode="overlay",
                            nbins=30, title=f"{scenario_name}: Distribuição de vlr_contest")
        st.plotly_chart(fig2, use_container_width=True)

    # “Árvore” (sunburst) — precisa de paths de regras
    if rule_paths_df is not None and not rule_paths_df.empty:
        st.caption("🌳 Visualização tipo árvore (sunburst) — como as regras segmentam a base")
        fig3 = px.sunburst(
            rule_paths_df,
            path=[c for c in rule_paths_df.columns if c.startswith("reg_")] + ["status_final"],
            values="count",
            title=f"{scenario_name}: Regras → Status final"
        )
        st.plotly_chart(fig3, use_container_width=True)

def build_rule_paths(df: pd.DataFrame, rules, rule_order_cols, logic_mode):
    """Constrói um DataFrame com colunas reg_{col} = 'Passa'/'Não passa' na ordem,
       e status_final = 'Aprovado'/'Negado', para um sunburst."""
    if not rules or not rule_order_cols:
        return None
    # Para cada coluna na ordem, avalia só aquela(s) regra(s) (AND entre regras da mesma coluna)
    parts = {}
    for col in rule_order_cols:
        sub = [r for r in rules if r[0] == col]
        if not sub:
            continue
        parts[f"reg_{col}"] = df.apply(lambda r: "Passa" if apply_rules_row(r, sub, "AND (todas)") else "Não passa", axis=1)
    if not parts:
        return None
    tmp = pd.DataFrame(parts)
    final = df.apply(lambda r: "Aprovado" if apply_rules_row(r, rules, logic_mode) else "Negado", axis=1)
    tmp["status_final"] = final
    tmp["count"] = 1
    # agrega para reduzir cardinalidade
    group_cols = [c for c in tmp.columns if c.startswith("reg_")] + ["status_final"]
    return tmp.groupby(group_cols, as_index=False)["count"].sum()

# -----------------------------
# Tabs de Cenários
# -----------------------------
tabA, tabB, tabCmp = st.tabs(["🅰️ Cenário A", "🅱️ Cenário B", "🔀 Comparar A vs B"])

with tabA:
    rulesA, logicA = build_rule_ui(base, "Cenário A")
    if rulesA:
        order_cols_A = [c for c in (st.session_state.get("sel_Cenário A") or [])]
        with st.spinner("Calculando cenário A..."):
            outA, metricsA = evaluate_scenario(base, rulesA, logicA, order_cols_A)
            pathsA = build_rule_paths(base, rulesA, order_cols_A, logicA)

        # KPIs
        st.subheader("📈 Indicadores — Cenário A")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Pote Total (aprovados)", f"{metricsA['pote_total']:,}".replace(",", "."))
        k2.metric("Taxa de aprovação", f"{metricsA['taxa_aprovacao_pct']:.2f}%")
        k3.metric("Impacto (aprovados) R$", f"{metricsA['vlr_aprovado']:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
        k4.metric("Impacto (negados) R$", f"{metricsA['vlr_negado']:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
        plots_for_scenario(outA, "Cenário A", pathsA)

        with st.expander("📋 Amostra de resultados (A)", expanded=False):
            st.dataframe(outA.head(30), use_container_width=True)
        st.session_state["outA"] = outA
        st.session_state["metricsA"] = metricsA
        st.session_state["rulesA"] = rulesA
        st.session_state["logicA"] = logicA
        st.session_state["orderA"] = order_cols_A
    else:
        st.info("Selecione variáveis e defina as regras para o Cenário A.")

with tabB:
    rulesB, logicB = build_rule_ui(base, "Cenário B")
    if rulesB:
        order_cols_B = [c for c in (st.session_state.get("sel_Cenário B") or [])]
        with st.spinner("Calculando cenário B..."):
            outB, metricsB = evaluate_scenario(base, rulesB, logicB, order_cols_B)
            pathsB = build_rule_paths(base, rulesB, order_cols_B, logicB)

        st.subheader("📈 Indicadores — Cenário B")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Pote Total (aprovados)", f"{metricsB['pote_total']:,}".replace(",", "."))
        k2.metric("Taxa de aprovação", f"{metricsB['taxa_aprovacao_pct']:.2f}%")
        k3.metric("Impacto (aprovados) R$", f"{metricsB['vlr_aprovado']:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
        k4.metric("Impacto (negados) R$", f"{metricsB['vlr_negado']:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
        plots_for_scenario(outB, "Cenário B", pathsB)

        with st.expander("📋 Amostra de resultados (B)", expanded=False):
            st.dataframe(outB.head(30), use_container_width=True)
        st.session_state["outB"] = outB
        st.session_state["metricsB"] = metricsB
        st.session_state["rulesB"] = rulesB
        st.session_state["logicB"] = logicB
        st.session_state["orderB"] = order_cols_B
    else:
        st.info("Selecione variáveis e defina as regras para o Cenário B.")

with tabCmp:
    mA = st.session_state.get("metricsA")
    mB = st.session_state.get("metricsB")
    if not (mA and mB):
        st.info("Configure e calcule os dois cenários para comparar.")
        st.stop()

    st.subheader("🧮 Comparação A vs B")
    cmp_df = pd.DataFrame([
        ["Pote Total (aprovados)", mA["pote_total"], mB["pote_total"],
         mB["pote_total"] - mA["pote_total"],
         ( (mB["pote_total"] - mA["pote_total"]) / mA["pote_total"] * 100 ) if mA["pote_total"] != 0 else np.nan],
        ["Taxa de aprovação (%)", mA["taxa_aprovacao_pct"], mB["taxa_aprovacao_pct"],
         mB["taxa_aprovacao_pct"] - mA["taxa_aprovacao_pct"],
         ( (mB["taxa_aprovacao_pct"] - mA["taxa_aprovacao_pct"]) / mA["taxa_aprovacao_pct"] * 100 ) if mA["taxa_aprovacao_pct"] != 0 else np.nan],
        ["Impacto Aprovados (R$)", mA["vlr_aprovado"], mB["vlr_aprovado"],
         mB["vlr_aprovado"] - mA["vlr_aprovado"],
         ( (mB["vlr_aprovado"] - mA["vlr_aprovado"]) / mA["vlr_aprovado"] * 100 ) if mA["vlr_aprovado"] != 0 else np.nan],
        ["Impacto Negados (R$)", mA["vlr_negado"], mB["vlr_negado"],
         mB["vlr_negado"] - mA["vlr_negado"],
         ( (mB["vlr_negado"] - mA["vlr_negado"]) / mA["vlr_negado"] * 100 ) if mA["vlr_negado"] != 0 else np.nan],
    ], columns=["Métrica", "Cenário A", "Cenário B", "Δ Abs (B - A)", "Δ % (B vs A)"])

    # Formatação amigável
    fmt_cols_money = {"Cenário A", "Cenário B", "Δ Abs (B - A)"}
    def fmt(row):
        if "R$" in row["Métrica"]:
            for c in ["Cenário A", "Cenário B", "Δ Abs (B - A)"]:
                row[c] = f"{row[c]:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        elif "(%)" in row["Métrica"]:
            row["Cenário A"] = f"{row['Cenário A']:.2f}%"
            row["Cenário B"] = f"{row['Cenário B']:.2f}%"
            row["Δ Abs (B - A)"] = f"{row['Δ Abs (B - A)']:.2f} p.p."
        else:
            for c in ["Cenário A", "Cenário B", "Δ Abs (B - A)"]:
                row[c] = f"{int(row[c])}"
        row["Δ % (B vs A)"] = "-" if pd.isna(row["Δ % (B vs A)"]) else f"{row['Δ % (B vs A)']:.2f}%"
        return row

    cmp_df_fmt = cmp_df.apply(fmt, axis=1)
    st.dataframe(cmp_df_fmt, use_container_width=True)

    # Gráfico side-by-side de taxa aprovação
    chart_df = pd.DataFrame({
        "Cenário": ["A", "B"],
        "Taxa de aprovação (%)": [mA["taxa_aprovacao_pct"], mB["taxa_aprovacao_pct"]]
    })
    fig_cmp = px.bar(chart_df, x="Cenário", y="Taxa de aprovação (%)", title="Taxa de aprovação — A vs B")
    st.plotly_chart(fig_cmp, use_container_width=True)

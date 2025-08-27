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
    base["vlr_contest"] = 0.0  # ainda usamos pra impacto financeiro

# -----------------------------
# Helpers de tipo
# -----------------------------
def is_bool_series(s: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(s):
        return True
    if pd.api.types.is_object_dtype(s):
        vals = set(str(x).strip().lower() for x in s.dropna().unique())
        cand = {"s", "n", "sim", "nao", "não", "true", "false", "0", "1"}
        return len(vals) <= 2 and vals.issubset(cand)
    return False

def normalize_bool(v):
    if isinstance(v, bool): return v
    if v is None or (isinstance(v, float) and np.isnan(v)): return None
    s = str(v).strip().lower()
    if s in {"s", "sim", "true", "1"}: return True
    if s in {"n", "nao", "não", "false", "0"}: return False
    return None

numeric_cols = [c for c in base.columns if pd.api.types.is_numeric_dtype(base[c])]
bool_cols    = [c for c in base.columns if is_bool_series(base[c])]
cat_cols     = [c for c in base.columns
                if (pd.api.types.is_object_dtype(base[c]) or pd.api.types.is_categorical_dtype(base[c]))
                and c not in bool_cols]

# -----------------------------
# UI de regras por cenário
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

    rules = []
    for col in selected:
        with st.container(border=True):
            st.markdown(f"**{col}**")
            s = df[col]
            if col in numeric_cols:
                cmin = float(np.nanmin(s.values)); cmax = float(np.nanmax(s.values))
                op = st.radio(
                    f"Operador para `{col}`",
                    ["≥ (mínimo)", "≤ (máximo)", "entre"],
                    horizontal=True, key=f"{key_prefix}_op_{col}"
                )
                if op == "≥ (mínimo)":
                    thr = st.slider("Valor mínimo", cmin, cmax, cmin, key=f"{key_prefix}_thr_ge_{col}")
                    rules.append((col, "num_ge", thr))
                elif op == "≤ (máximo)":
                    thr = st.slider("Valor máximo", cmin, cmax, cmax, key=f"{key_prefix}_thr_le_{col}")
                    rules.append((col, "num_le", thr))
                else:
                    rng = st.slider("Intervalo permitido", cmin, cmax, (cmin, cmax), key=f"{key_prefix}_thr_between_{col}")
                    rules.append((col, "num_between", rng))
            elif col in bool_cols:
                choice = st.radio(
                    f"Como tratar `{col}`?",
                    ["Exigir True", "Exigir False"],
                    horizontal=True, key=f"{key_prefix}_bool_{col}"
                )
                rules.append((col, "bool_true" if choice == "Exigir True" else "bool_false", None))
            else:
                uniques = s.dropna().astype(str).unique().tolist()
                uniques = sorted(uniques[:500])  # corta pra não pesar
                sel = st.multiselect(
                    f"Valores permitidos em `{col}`",
                    options=uniques, default=uniques, key=f"{key_prefix}_cat_{col}"
                )
                rules.append((col, "cat_in", sel))

    return rules, logic_mode

def apply_rules_row(row, rules, logic_mode):
    if not rules: return True
    results = []
    for (col, rtype, param) in rules:
        val = row[col]
        if rtype.startswith("num_"):
            if pd.isna(val): results.append(False); continue
            x = float(val)
            if rtype == "num_ge": results.append(x >= float(param))
            elif rtype == "num_le": results.append(x <= float(param))
            elif rtype == "num_between":
                lo, hi = param; results.append((x >= float(lo)) and (x <= float(hi)))
        elif rtype.startswith("bool_"):
            nb = normalize_bool(val)
            if nb is None: results.append(False)
            else: results.append(nb is True if rtype == "bool_true" else nb is False)
        elif rtype == "cat_in":
            v = str(val) if pd.notna(val) else ""
            results.append(v in set(param))
    return all(results) if logic_mode.startswith("AND") else any(results)

def evaluate_scenario(df: pd.DataFrame, rules, logic_mode, rule_order_cols):
    out = df.copy()
    out["aprovado"] = out.apply(lambda r: apply_rules_row(r, rules, logic_mode), axis=1)

    pote_total = int(out["aprovado"].sum())
    taxa = float(out["aprovado"].mean() * 100.0)
    vlr_aprov = float(out.loc[out["aprovado"], "vlr_contest"].sum())
    vlr_neg   = float(out.loc[~out["aprovado"], "vlr_contest"].sum())

    # Motivo (para AND, primeira coluna cuja regra não passa)
    if rules and rule_order_cols and logic_mode.startswith("AND"):
        def first_fail(row):
            if row["aprovado"]: return "Aprovado"
            for col in rule_order_cols:
                sub = [r for r in rules if r[0] == col]
                if sub and not apply_rules_row(row, sub, "AND (todas)"):
                    return f"Falha em {col}"
            return "Falha (outra)"
        out["motivo_negativa"] = out.apply(first_fail, axis=1)
    else:
        out["motivo_negativa"] = np.where(out["aprovado"], "Aprovado", "Negado")

    return out, {
        "pote_total": pote_total,
        "taxa_aprovacao_pct": taxa,
        "vlr_aprovado": vlr_aprov,
        "vlr_negado": vlr_neg
    }

# -------- Árvore de decisão (treemap com % por nó) --------
def build_decision_tree_df(df: pd.DataFrame, rules, rule_order_cols, logic_mode):
    """
    Gera um DF para treemap: colunas reg_{col} com 'Passa'/'Não passa' condicionais na ordem,
    e métricas por nó: count e aprov_rate (% de aprovados condicionado ao nó).
    """
    if not rules or not rule_order_cols:
        return None

    # 1) marca por coluna (AND dentro da mesma coluna, seguindo a ordem escolhida)
    parts = {}
    for col in rule_order_cols:
        sub = [r for r in rules if r[0] == col]
        if not sub: continue
        parts[f"reg_{col}"] = df.apply(lambda r: "Passa" if apply_rules_row(r, sub, "AND (todas)") else "Não passa", axis=1)
    if not parts: return None

    tmp = pd.DataFrame(parts)
    tmp["aprovado"] = df.apply(lambda r: apply_rules_row(r, rules, logic_mode), axis=1)
    tmp["__n__"] = 1

    # 2) agrega em todos os níveis (prefixos) para ter taxa condicional por nó
    group_cols = [c for c in tmp.columns if c.startswith("reg_")]
    agg_rows = []
    for depth in range(1, len(group_cols)+1):
        gcols = group_cols[:depth]
        grp = tmp.groupby(gcols, dropna=False).agg(
            n=("__n__", "sum"),
            aprov_rate=("aprovado", "mean")
        ).reset_index()
        grp["depth"] = depth
        agg_rows.append(grp)
    agg = pd.concat(agg_rows, ignore_index=True)

    # 3) formata labels amigáveis (ex.: "vlr_invest: Passa — 72% (n=120)")
    def label_from_col(col_name):
        # reg_{col} -> {col}
        return col_name.replace("reg_", "")

    for c in group_cols:
        base_col = label_from_col(c)
        agg[c] = agg[c].fillna("NaN").astype(str).map(lambda v: f"{base_col}: {v}")

    # 4) valores finais pro treemap
    agg["aprov_pct_txt"] = (agg["aprov_rate"] * 100).round(1).astype(str) + "%"

    # rótulo do nó = último nível + pct + n
    last_level = group_cols[-1]
    agg["node_label"] = (
        agg.apply(lambda r: f"{r[[gc for gc in group_cols][:r['depth']]].tolist()[-1]} — {r['aprov_pct_txt']} (n={int(r['n'])})", axis=1)
    )

    return agg, group_cols

def show_decision_tree(agg_df, group_cols, title):
    if agg_df is None or agg_df.empty:
        st.info("Defina a ordem/variáveis de regras para ver a árvore.")
        return
    # filtra apenas a linha do nível máximo para size, mas precisamos do path completo
    max_depth = agg_df["depth"].max()
    top = agg_df[agg_df["depth"] == max_depth].copy()

    # path = todos os níveis (reg_col1, reg_col2, ...)
    path_cols = group_cols
    # Para exibir o rótulo custom em cada nível, criamos colunas textuais por nível
    label_cols = []
    for i, gc in enumerate(path_cols, start=1):
        lvl_col = f"lvl_{i}"
        label_cols.append(lvl_col)
        # pega o label daquele nível a partir do agg no depth correspondente
        # simplificação: usa o próprio texto gc presente em top
        top[lvl_col] = top[gc]

    # colorir pelo % aprovado no nó (no nível máximo). (0-100)
    top["aprov_pct"] = top["aprov_pct_txt"].str.replace("%", "", regex=False).astype(float)

    fig = px.treemap(
        top,
        path=label_cols,
        values="n",
        color="aprov_pct",
        color_continuous_scale="RdYlGn",
        range_color=(0, 100),
        title=title
    )
    fig.update_traces(
        hovertemplate="<b>%{label}</b><br>Casos: %{value}<br>% Aprov.: %{color:.1f}%<extra></extra>"
    )
    st.plotly_chart(fig, use_container_width=True)

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
            aggA, path_cols_A = build_decision_tree_df(base, rulesA, order_cols_A, logicA)

        # KPIs
        st.subheader("📈 Indicadores — Cenário A")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Pote Total (aprovados)", f"{metricsA['pote_total']:,}".replace(",", "."))
        k2.metric("Taxa de aprovação", f"{metricsA['taxa_aprovacao_pct']:.2f}%")
        k3.metric("Impacto (aprovados) R$", f"{metricsA['vlr_aprovado']:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
        k4.metric("Impacto (negados) R$", f"{metricsA['vlr_negado']:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

        # Gráfico: Aprovados vs Negados
        vc = outA["aprovado"].value_counts().rename({True: "Aprovado", False: "Negado"}).reset_index()
        vc.columns = ["status", "qtd"]
        fig = px.bar(vc, x="status", y="qtd", title="Aprovados vs Negados — Cenário A")
        st.plotly_chart(fig, use_container_width=True)

        # Árvore de decisão (percentual condicional por nó)
        st.caption("🌳 Árvore de decisão — percentuais condicionais por regra (em ordem).")
        show_decision_tree(aggA, path_cols_A, "Árvore — Cenário A")

        with st.expander("📋 Amostra de resultados (A)", expanded=False):
            st.dataframe(outA.head(30), use_container_width=True)

        st.session_state["metricsA"], st.session_state["outA"] = metricsA, outA
    else:
        st.info("Selecione variáveis e defina as regras para o Cenário A.")

with tabB:
    rulesB, logicB = build_rule_ui(base, "Cenário B")
    if rulesB:
        order_cols_B = [c for c in (st.session_state.get("sel_Cenário B") or [])]
        with st.spinner("Calculando cenário B..."):
            outB, metricsB = evaluate_scenario(base, rulesB, logicB, order_cols_B)
            aggB, path_cols_B = build_decision_tree_df(base, rulesB, order_cols_B, logicB)

        st.subheader("📈 Indicadores — Cenário B")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Pote Total (aprovados)", f"{metricsB['pote_total']:,}".replace(",", "."))
        k2.metric("Taxa de aprovação", f"{metricsB['taxa_aprovacao_pct']:.2f}%")
        k3.metric("Impacto (aprovados) R$", f"{metricsB['vlr_aprovado']:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
        k4.metric("Impacto (negados) R$", f"{metricsB['vlr_negado']:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

        vc = outB["aprovado"].value_counts().rename({True: "Aprovado", False: "Negado"}).reset_index()
        vc.columns = ["status", "qtd"]
        fig = px.bar(vc, x="status", y="qtd", title="Aprovados vs Negados — Cenário B")
        st.plotly_chart(fig, use_container_width=True)

        st.caption("🌳 Árvore de decisão — percentuais condicionais por regra (em ordem).")
        show_decision_tree(aggB, path_cols_B, "Árvore — Cenário B")

        with st.expander("📋 Amostra de resultados (B)", expanded=False):
            st.dataframe(outB.head(30), use_container_width=True)

        st.session_state["metricsB"], st.session_state["outB"] = metricsB, outB
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

    st.dataframe(cmp_df.apply(fmt, axis=1), use_container_width=True)

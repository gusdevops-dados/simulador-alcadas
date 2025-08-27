# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Simulador de Alçadas – Cenários", layout="wide")
st.title("📊 Simulador de Políticas e Alçadas — Múltiplos Cenários")

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

# Garantir coluna de valor para impacto
if "vlr_contest" not in base.columns:
    base["vlr_contest"] = 0.0

PUBLICO_TOTAL = int(len(base))

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
# Estado de múltiplos cenários
# -----------------------------
if "scenarios" not in st.session_state:
    st.session_state.scenarios = ["Cenário A", "Cenário B"]  # default
if "scenario_results" not in st.session_state:
    st.session_state.scenario_results = {}  # name -> dict(metrics/out)

with st.sidebar:
    st.header("🗂️ Cenários")
    # adicionar cenário
    new_name = st.text_input("Nome do novo cenário", value="Cenário C")
    cols_btn = st.columns(2)
    with cols_btn[0]:
        if st.button("➕ Adicionar", use_container_width=True):
            name = new_name.strip() or f"Cenário {len(st.session_state.scenarios)+1}"
            if name not in st.session_state.scenarios:
                st.session_state.scenarios.append(name)
    # remover cenário
    with cols_btn[1]:
        rem = st.selectbox("Remover cenário", ["(nenhum)"] + st.session_state.scenarios, index=0)
        if st.button("🗑️ Remover", use_container_width=True) and rem != "(nenhum)":
            st.session_state.scenarios = [s for s in st.session_state.scenarios if s != rem]
            st.session_state.scenario_results.pop(rem, None)

# -----------------------------
# Construtores de UI e avaliação
# -----------------------------
def build_rule_ui(df: pd.DataFrame, key_prefix: str):
    st.subheader(f"🧩 Regras — {key_prefix}")
    selected = st.multiselect(
        "Variáveis que entram nas regras:",
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
                uniques = sorted(uniques[:500])
                sel = st.multiselect(
                    f"Valores permitidos em `{col}`",
                    options=uniques, default=uniques, key=f"{key_prefix}_cat_{col}"
                )
                rules.append((col, "cat_in", sel))
    return rules, logic_mode

def apply_rules_row(row, rules, logic_mode):
    if not rules: return True
    res = []
    for (col, rtype, param) in rules:
        val = row[col]
        if rtype.startswith("num_"):
            if pd.isna(val): res.append(False); continue
            x = float(val)
            if rtype == "num_ge": res.append(x >= float(param))
            elif rtype == "num_le": res.append(x <= float(param))
            elif rtype == "num_between":
                lo, hi = param; res.append((x >= float(lo)) and (x <= float(hi)))
        elif rtype.startswith("bool_"):
            nb = normalize_bool(val)
            if nb is None: res.append(False)
            else: res.append(nb is True if rtype == "bool_true" else nb is False)
        elif rtype == "cat_in":
            v = str(val) if pd.notna(val) else ""
            res.append(v in set(param))
    return all(res) if logic_mode.startswith("AND") else any(res)

def evaluate_scenario(df: pd.DataFrame, rules, logic_mode, rule_order_cols):
    out = df.copy()
    out["aprovado"] = out.apply(lambda r: apply_rules_row(r, rules, logic_mode), axis=1)

    pote_total = int(out["aprovado"].sum())                 # público que passou
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

    metrics = {
        "publico_total": int(len(df)),
        "pote_total": pote_total,
        "taxa_aprovacao_pct": taxa,
        "vlr_aprovado": vlr_aprov,
        "vlr_negado": vlr_neg
    }
    return out, metrics

def build_decision_tree_df(df: pd.DataFrame, rules, rule_order_cols, logic_mode):
    if not rules or not rule_order_cols: return None, None
    parts = {}
    for col in rule_order_cols:
        sub = [r for r in rules if r[0] == col]
        if not sub: continue
        parts[f"reg_{col}"] = df.apply(lambda r: "Passa" if apply_rules_row(r, sub, "AND (todas)") else "Não passa", axis=1)
    if not parts: return None, None

    tmp = pd.DataFrame(parts)
    tmp["aprovado"] = df.apply(lambda r: apply_rules_row(r, rules, logic_mode), axis=1)
    tmp["__n__"] = 1

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

    def label_from_col(c): return c.replace("reg_", "")
    for c in group_cols:
        base_col = label_from_col(c)
        agg[c] = agg[c].fillna("NaN").astype(str).map(lambda v: f"{base_col}: {v}")

    agg["aprov_pct_txt"] = (agg["aprov_rate"] * 100).round(1).astype(str) + "%"
    last_level = group_cols[-1]
    agg["node_label"] = (
        agg.apply(lambda r: f"{r[[gc for gc in group_cols][:r['depth']]].tolist()[-1]} — {r['aprov_pct_txt']} (n={int(r['n'])})", axis=1)
    )
    return agg, group_cols

def show_decision_tree(agg_df, group_cols, title):
    if agg_df is None or agg_df.empty:
        st.info("Defina a ordem/variáveis de regras para ver a árvore.")
        return
    max_depth = agg_df["depth"].max()
    top = agg_df[agg_df["depth"] == max_depth].copy()

    label_cols = []
    for i, gc in enumerate(group_cols, start=1):
        lvl_col = f"lvl_{i}"
        label_cols.append(lvl_col)
        top[lvl_col] = top[gc]

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
    fig.update_traces(hovertemplate="<b>%{label}</b><br>Casos: %{value}<br>% Aprov.: %{color:.1f}%<extra></extra>")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Render dinâmico: uma aba por cenário
# -----------------------------
tabs = st.tabs(st.session_state.scenarios + ["📊 Comparar"])
scenario_outputs = {}  # name -> (out_df, metrics)

for i, name in enumerate(st.session_state.scenarios):
    with tabs[i]:
        rules, logic = build_rule_ui(base, name)
        if not rules:
            st.info("Selecione variáveis e defina as regras para este cenário.")
            continue

        order_cols = [c for c in (st.session_state.get(f"sel_{name}") or [])]
        with st.spinner(f"Calculando {name}..."):
            out, metrics = evaluate_scenario(base, rules, logic, order_cols)
            agg, path_cols = build_decision_tree_df(base, rules, order_cols, logic)

        # KPIs
        st.subheader(f"📈 Indicadores — {name}")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Público total (N)", f"{metrics['publico_total']:,}".replace(",", "."))
        k2.metric("Pote Total (aprovados)", f"{metrics['pote_total']:,}".replace(",", "."))
        k3.metric("Taxa de aprovação", f"{metrics['taxa_aprovacao_pct']:.2f}%")
        k4.metric("Impacto (aprovados) R$", f"{metrics['vlr_aprovado']:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
        k5.metric("Impacto (negados) R$", f"{metrics['vlr_negado']:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

        # Aprovados vs Negados
        vc = out["aprovado"].value_counts().rename({True: "Aprovado", False: "Negado"}).reset_index()
        vc.columns = ["status", "qtd"]
        fig = px.bar(vc, x="status", y="qtd", title=f"Aprovados vs Negados — {name}")
        st.plotly_chart(fig, use_container_width=True)

        # Árvore de decisão
        st.caption("🌳 Árvore de decisão — percentuais condicionais por regra (na ordem selecionada).")
        show_decision_tree(agg, path_cols, f"Árvore — {name}")

        with st.expander(f"📋 Amostra de resultados ({name})", expanded=False):
            st.dataframe(out.head(30), use_container_width=True)

        scenario_outputs[name] = (out, metrics)

# -----------------------------
# Comparação de todos os cenários
# -----------------------------
with tabs[-1]:
    if not scenario_outputs:
        st.info("Configure ao menos um cenário para comparar.")
        st.stop()

    st.subheader("🧮 Comparação de Indicadores (todos os cenários)")
    rows = []
    for name, (_, m) in scenario_outputs.items():
        rows.append({
            "Cenário": name,
            "Público total (N)": m["publico_total"],
            "Pote Total (aprovados)": m["pote_total"],
            "Taxa de aprovação (%)": m["taxa_aprovacao_pct"],
            "Impacto Aprovados (R$)": m["vlr_aprovado"],
            "Impacto Negados (R$)": m["vlr_negado"],
        })
    cmp_df = pd.DataFrame(rows)

    # formatações amigáveis
    fmt = cmp_df.copy()
    fmt["Público total (N)"] = fmt["Público total (N)"].map(lambda x: f"{int(x):,}".replace(",", "."))
    fmt["Pote Total (aprovados)"] = fmt["Pote Total (aprovados)"].map(lambda x: f"{int(x):,}".replace(",", "."))
    fmt["Taxa de aprovação (%)"] = fmt["Taxa de aprovação (%)"].map(lambda x: f"{x:.2f}%")
    for col in ["Impacto Aprovados (R$)", "Impacto Negados (R$)"]:
        fmt[col] = fmt[col].map(lambda x: f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

    st.dataframe(fmt, use_container_width=True)

    # opcional: gráfico comparativo de taxa de aprovação
    fig_cmp = px.bar(
        cmp_df.sort_values("Taxa de aprovação (%)", ascending=False),
        x="Cenário", y="Taxa de aprovação (%)",
        title="Taxa de aprovação por cenário"
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    # download do comparativo bruto
    st.download_button(
        "⬇️ Baixar comparativo (CSV)",
        data=cmp_df.to_csv(index=False).encode("utf-8"),
        file_name="comparativo_cenarios.csv",
        mime="text/csv"
    )

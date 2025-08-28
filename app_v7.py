# app.py
import streamlit as st
import pandas as pd
import numpy as np
import json
import io
import plotly.express as px

st.set_page_config(page_title="Simulador de Alçadas — Cenários", layout="wide")
st.title("📊 Simulador de Políticas e Alçadas — Múltiplos Cenários")

# -----------------------------
# Upload
# -----------------------------
uploaded_file = st.file_uploader("Suba a base (CSV ou Excel)", type=["csv", "xlsx"])
if not uploaded_file:
    st.info("⏫ Envie um arquivo para começar.")
    st.stop()

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

# Garantir coluna financeira usada nos KPIs
if "vlr_contest" not in base.columns:
    base["vlr_contest"] = 0.0

# -----------------------------
# Helpers de tipo
# -----------------------------
def is_bool_series(s: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(s): return True
    if pd.api.types.is_object_dtype(s):
        vals = set(str(x).strip().lower() for x in s.dropna().unique())
        cand = {"s","n","sim","nao","não","true","false","0","1"}
        return len(vals) <= 2 and vals.issubset(cand)
    return False

def normalize_bool(v):
    if isinstance(v, bool): return v
    if v is None or (isinstance(v, float) and np.isnan(v)): return None
    s = str(v).strip().lower()
    if s in {"s","sim","true","1"}: return True
    if s in {"n","nao","não","false","0"}: return False
    return None

numeric_cols = [c for c in base.columns if pd.api.types.is_numeric_dtype(base[c])]
bool_cols    = [c for c in base.columns if is_bool_series(base[c])]
cat_cols     = [c for c in base.columns
                if (pd.api.types.is_object_dtype(base[c]) or pd.api.types.is_categorical_dtype(base[c]))
                and c not in bool_cols]

# -----------------------------
# Estado global
# -----------------------------
if "scenarios" not in st.session_state:
    st.session_state.scenarios = ["Cenário A", "Cenário B"]
if "scenario_cfg" not in st.session_state:
    # name -> {"logic": str, "rules": [(col, rtype, param), ...], "order": [col1, col2, ...]}
    st.session_state.scenario_cfg = {}
if "scenario_outputs" not in st.session_state:
    # name -> (out_df, metrics)
    st.session_state.scenario_outputs = {}

PUBLICO_TOTAL = int(len(base))

# -----------------------------
# Funções de avaliação e UI de regras
# -----------------------------
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

    pote_total = int(out["aprovado"].sum())
    taxa = float(out["aprovado"].mean() * 100.0)
    vlr_aprov = float(out.loc[out["aprovado"], "vlr_contest"].sum())
    vlr_neg   = float(out.loc[~out["aprovado"], "vlr_contest"].sum())

    # Ticket médio (úteis para finanças)
    tk_aprov = float(out.loc[out["aprovado"], "vlr_contest"].mean()) if pote_total > 0 else 0.0
    tk_neg   = float(out.loc[~out["aprovado"], "vlr_contest"].mean()) if pote_total < len(out) else 0.0

    # Motivo (para AND: primeira coluna que falha na ordem)
    if rules and rule_order_cols and logic_mode.startswith("AND"):
        def first_fail(row):
            if row["aprovado"]: return "Aprovado"
            for col in rule_order_cols:
                sub = [r for r in rules if r[0] == col]
                # avalia apenas as regras daquela coluna
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
        "vlr_negado": vlr_neg,
        "ticket_medio_aprov": tk_aprov,
        "ticket_medio_neg": tk_neg
    }
    return out, metrics

def build_rule_ui(df: pd.DataFrame, key_prefix: str, preset=None):
    st.subheader(f"🧩 Regras — {key_prefix}")
    # seleção de colunas
    selected = st.multiselect(
        "Variáveis que entram nas regras:",
        options=numeric_cols + bool_cols + cat_cols,
        default=(preset.get("order") if preset else []),
        key=f"sel_{key_prefix}"
    )
    logic_mode = st.radio(
        "Agregação entre condições:",
        ["AND (todas)", "OR (qualquer)"],
        index=0 if not preset else (0 if preset.get("logic","AND").startswith("AND") else 1),
        horizontal=True, key=f"logic_{key_prefix}"
    )

    rules = []
    for col in selected:
        with st.container(border=True):
            st.markdown(f"**{col}**")
            s = df[col]
            # resgatar preset de regra específica, se existir
            preset_rule = None
            if preset and preset.get("rules"):
                for (pc, prt, pp) in preset["rules"]:
                    if pc == col:
                        preset_rule = (prt, pp)
                        break

            if col in numeric_cols:
                cmin = float(np.nanmin(s.values)); cmax = float(np.nanmax(s.values))
                default_op = "≥ (mínimo)"
                default_thr_ge = cmin
                default_thr_le = cmax
                default_between = (cmin, cmax)
                if preset_rule:
                    prt, pp = preset_rule
                    if prt == "num_ge": default_op, default_thr_ge = "≥ (mínimo)", float(pp)
                    elif prt == "num_le": default_op, default_thr_le = "≤ (máximo)", float(pp)
                    elif prt == "num_between": default_op, default_between = "entre", tuple(pp)

                op = st.radio(
                    f"Operador para `{col}`",
                    ["≥ (mínimo)", "≤ (máximo)", "entre"],
                    horizontal=True, key=f"{key_prefix}_op_{col}", index=["≥ (mínimo)","≤ (máximo)","entre"].index(default_op)
                )
                if op == "≥ (mínimo)":
                    thr = st.slider("Valor mínimo", cmin, cmax, default_thr_ge, key=f"{key_prefix}_thr_ge_{col}")
                    rules.append((col, "num_ge", thr))
                elif op == "≤ (máximo)":
                    thr = st.slider("Valor máximo", cmin, cmax, default_thr_le, key=f"{key_prefix}_thr_le_{col}")
                    rules.append((col, "num_le", thr))
                else:
                    rng = st.slider("Intervalo permitido", cmin, cmax, default_between, key=f"{key_prefix}_thr_between_{col}")
                    rules.append((col, "num_between", rng))

            elif col in bool_cols:
                default_choice = "Exigir True"
                if preset_rule:
                    prt, _ = preset_rule
                    default_choice = "Exigir True" if prt == "bool_true" else "Exigir False"
                choice = st.radio(
                    f"Como tratar `{col}`?",
                    ["Exigir True", "Exigir False"],
                    horizontal=True, key=f"{key_prefix}_bool_{col}",
                    index=0 if default_choice == "Exigir True" else 1
                )
                rules.append((col, "bool_true" if choice == "Exigir True" else "bool_false", None))

            else:  # categórica
                uniques = s.dropna().astype(str).unique().tolist()
                uniques = sorted(uniques[:500])
                default_sel = uniques
                if preset_rule:
                    prt, pp = preset_rule
                    if prt == "cat_in":
                        default_sel = [u for u in uniques if u in set(pp)]
                sel = st.multiselect(
                    f"Valores permitidos em `{col}`",
                    options=uniques, default=default_sel, key=f"{key_prefix}_cat_{col}"
                )
                rules.append((col, "cat_in", sel))

    return rules, logic_mode, selected

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
    return agg, group_cols

def show_decision_tree(agg_df, group_cols, title):
    if agg_df is None or agg_df.empty:
        st.info("Defina a ordem/variáveis de regras para ver a árvore.")
        return
    max_depth = agg_df["depth"].max()
    top = agg_df[agg_df["depth"] == max_depth].copy()
    # labels por nível
    label_cols = []
    for i, gc in enumerate(group_cols, start=1):
        lvl_col = f"lvl_{i}"
        label_cols.append(lvl_col)
        top[lvl_col] = top[gc]
    top["aprov_pct"] = top["aprov_pct_txt"].str.replace("%","",regex=False).astype(float)
    fig = px.treemap(
        top, path=label_cols, values="n",
        color="aprov_pct", color_continuous_scale="RdYlGn",
        range_color=(0,100), title=title
    )
    fig.update_traces(hovertemplate="<b>%{label}</b><br>Casos: %{value}<br>% Aprov.: %{color:.1f}%<extra></extra>")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Sidebar: Gerir cenários + Salvar/Carregar/Duplicar
# -----------------------------
with st.sidebar:
    st.header("🗂️ Cenários")
    # adicionar
    new_name = st.text_input("Nome do novo cenário", value="Cenário C")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("➕ Adicionar"):
            name = new_name.strip() or f"Cenário {len(st.session_state.scenarios)+1}"
            if name not in st.session_state.scenarios:
                st.session_state.scenarios.append(name)
    with c2:
        rem = st.selectbox("Remover", ["(nenhum)"] + st.session_state.scenarios, index=0)
        if st.button("🗑️ Remover") and rem != "(nenhum)":
            st.session_state.scenarios = [s for s in st.session_state.scenarios if s != rem]
            st.session_state.scenario_cfg.pop(rem, None)
            st.session_state.scenario_outputs.pop(rem, None)
    with c3:
        src = st.selectbox("Duplicar de", ["(nenhum)"] + st.session_state.scenarios, index=0, key="dup_src")
        dst = st.text_input("Como", value="Cenário (cópia)")
        if st.button("📄 Duplicar") and src != "(nenhum)":
            target = dst.strip() or f"{src} (cópia)"
            if target not in st.session_state.scenarios:
                st.session_state.scenarios.append(target)
                st.session_state.scenario_cfg[target] = json.loads(json.dumps(st.session_state.scenario_cfg.get(src, {})))

    st.divider()
    st.subheader("💾 Salvar/Carregar cenários")
    # Salvar (export JSON de todas as configs)
    export_btn = st.button("⬇️ Exportar cenários (JSON)")
    if export_btn:
        payload = json.dumps(st.session_state.scenario_cfg, ensure_ascii=False, indent=2)
        st.download_button("Baixar arquivo", data=payload.encode("utf-8"),
                           file_name="cenarios.json", mime="application/json")

    # Carregar
    imp = st.file_uploader("Importar cenários (JSON)", type=["json"], key="import_json")
    if imp is not None:
        try:
            cfg = json.load(io.StringIO(imp.getvalue().decode("utf-8")))
            if isinstance(cfg, dict):
                st.session_state.scenario_cfg.update(cfg)
                for name in cfg.keys():
                    if name not in st.session_state.scenarios:
                        st.session_state.scenarios.append(name)
                st.success("Configurações importadas!")
        except Exception as e:
            st.error(f"Falha ao importar: {e}")

# -----------------------------
# Render: abas de cenários
# -----------------------------
tabs = st.tabs(st.session_state.scenarios + ["📊 Comparar"])
for i, name in enumerate(st.session_state.scenarios):
    with tabs[i]:
        preset = st.session_state.scenario_cfg.get(name, {})
        rules, logic, selected_cols = build_rule_ui(base, name, preset=preset)

        # persistir config na sessão (para salvar/duplicar)
        st.session_state.scenario_cfg[name] = {
            "logic": logic,
            "rules": rules,
            "order": selected_cols
        }

        if not rules:
            st.info("Selecione variáveis e defina as regras para este cenário.")
            continue

        with st.spinner(f"Calculando {name}..."):
            out, metrics = evaluate_scenario(base, rules, logic, selected_cols)
            agg, path_cols = build_decision_tree_df(base, rules, selected_cols, logic)

        # KPIs (inclui ticket médio)
        st.subheader(f"📈 Indicadores — {name}")
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Público total (N)", f"{metrics['publico_total']:,}".replace(",", "."))
        k2.metric("Pote Total (aprovados)", f"{metrics['pote_total']:,}".replace(",", "."))
        k3.metric("Taxa de aprovação", f"{metrics['taxa_aprovacao_pct']:.2f}%")
        k4.metric("Impacto (aprovados) R$", f"{metrics['vlr_aprovado']:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
        k5.metric("Impacto (negados) R$", f"{metrics['vlr_negado']:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
        k6.metric("Ticket médio aprov. R$", f"{metrics['ticket_medio_aprov']:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

        # Aprovados vs Negados
        vc = out["aprovado"].value_counts().rename({True: "Aprovado", False: "Negado"}).reset_index()
        vc.columns = ["status", "qtd"]
        st.plotly_chart(px.bar(vc, x="status", y="qtd", title=f"Aprovados vs Negados — {name}"),
                        use_container_width=True)

        # Motivos da negativa (somente se AND)
        if logic.startswith("AND"):
            st.caption("🧭 Motivos de negativa (1ª regra que falha, ordem das variáveis).")
            neg = out.loc[out["motivo_negativa"] != "Aprovado", "motivo_negativa"].value_counts().reset_index()
            neg.columns = ["motivo", "qtd"]
            if not neg.empty:
                st.plotly_chart(px.bar(neg, x="motivo", y="qtd", title=f"Motivos da negativa — {name}"),
                                use_container_width=True)

        # Árvore de decisão
        st.caption("🌳 Árvore de decisão — % de aprovação condicionais por regra (na ordem selecionada).")
        show_decision_tree(agg, path_cols, f"Árvore — {name}")

        with st.expander(f"📋 Amostra de resultados ({name})", expanded=False):
            st.dataframe(out.head(30), use_container_width=True)

        # armazenar para comparação
        st.session_state.scenario_outputs[name] = (out, metrics)

# -----------------------------
# Comparação de cenários + agregações configuráveis
# -----------------------------
with tabs[-1]:
    st.subheader("🧮 Comparação de Indicadores")
    outs = st.session_state.scenario_outputs
    if not outs:
        st.info("Configure ao menos um cenário para comparar.")
        st.stop()

    # tabela base de KPIs
    rows = []
    for name, (_, m) in outs.items():
        rows.append({
            "Cenário": name,
            "Público total (N)": m["publico_total"],
            "Pote Total (aprovados)": m["pote_total"],
            "Taxa de aprovação (%)": m["taxa_aprovacao_pct"],
            "Impacto Aprovados (R$)": m["vlr_aprovado"],
            "Impacto Negados (R$)": m["vlr_negado"],
            "Ticket médio aprov. (R$)": m["ticket_medio_aprov"],
            "Ticket médio neg. (R$)": m["ticket_medio_neg"],
        })
    cmp_df = pd.DataFrame(rows)

    # ---- Agregações configuráveis por variável ----
    st.markdown("### 🔧 Agregações extras por cenário (configuráveis)")
    num_opts = [c for c in numeric_cols if c != "aprovado"]  # evitar coluna derivada
    agg_vars = st.multiselect("Variáveis numéricas para agregar (ex.: 'vlr_invest')", options=num_opts)
    agg_funcs = st.multiselect("Funções", options=["sum","mean","median","max","min"], default=["sum"])
    pop = st.selectbox("População", ["Aprovados", "Negados", "Todos"], index=0)

    extra_cols = []
    if agg_vars and agg_funcs:
        for scen_name, (out_df, _) in outs.items():
            if pop == "Aprovados":
                sub = out_df[out_df["aprovado"] == True]
                suffix = "aprov"
            elif pop == "Negados":
                sub = out_df[out_df["aprovado"] == False]
                suffix = "neg"
            else:
                sub = out_df
                suffix = "todos"

            agg_res = {}
            for v in agg_vars:
                if v in sub.columns and pd.api.types.is_numeric_dtype(sub[v]):
                    for f in agg_funcs:
                        try:
                            val = getattr(sub[v], f)()
                        except Exception:
                            val = np.nan
                        agg_res[f"{v}__{f}__{suffix}"] = val
            # anexa ao cmp_df
            for k, v in agg_res.items():
                colname = f"{k}"
                if colname not in cmp_df.columns:
                    cmp_df[colname] = np.nan
                    extra_cols.append(colname)
                cmp_df.loc[cmp_df["Cenário"] == scen_name, colname] = v

    # formatação amigável
    fmt = cmp_df.copy()
    # números inteiros
    for c in ["Público total (N)", "Pote Total (aprovados)"]:
        fmt[c] = fmt[c].map(lambda x: f"{int(x):,}".replace(",", "."))
    # percentuais
    fmt["Taxa de aprovação (%)"] = fmt["Taxa de aprovação (%)"].map(lambda x: f"{x:.2f}%")
    # dinheiro
    money_cols = ["Impacto Aprovados (R$)", "Impacto Negados (R$)", "Ticket médio aprov. (R$)", "Ticket médio neg. (R$)"]
    for c in money_cols:
        fmt[c] = fmt[c].map(lambda x: f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    # extra agregações (não sabemos se são dinheiro, então formatamos com 2 casas)
    for c in extra_cols:
        fmt[c] = fmt[c].map(lambda x: "-" if pd.isna(x) else f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

    st.dataframe(fmt, use_container_width=True)

    # ordenação opcional por alguma coluna
    sort_by = st.selectbox("Ordenar por", options=fmt.columns.tolist(), index=0)
    ascending = st.checkbox("Ordem crescente?", value=False)
    st.plotly_chart(
        px.bar(cmp_df.sort_values(sort_by.replace(" — formatado",""), ascending=ascending),
               x="Cenário", y="Taxa de aprovação (%)", title="Taxa de aprovação por cenário"),
        use_container_width=True
    )

    # download
    st.download_button(
        "⬇️ Baixar comparativo (CSV)",
        data=cmp_df.to_csv(index=False).encode("utf-8"),
        file_name="comparativo_cenarios.csv",
        mime="text/csv"
    )

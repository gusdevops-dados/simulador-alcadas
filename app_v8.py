# app.py
import streamlit as st
import pandas as pd
import numpy as np
import json
import io
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

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

# -----------------------------
# Engine de regras
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

    # Ticket médio
    tk_aprov = float(out.loc[out["aprovado"], "vlr_contest"].mean()) if pote_total > 0 else 0.0
    tk_neg   = float(out.loc[~out["aprovado"], "vlr_contest"].mean()) if pote_total < len(out) else 0.0

    # Motivo (para AND, primeira coluna que falha)
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
        "vlr_negado": vlr_neg,
        "ticket_medio_aprov": tk_aprov,
        "ticket_medio_neg": tk_neg
    }
    return out, metrics

# -----------------------------
# UI de regras por cenário
# -----------------------------
def build_rule_ui(df: pd.DataFrame, key_prefix: str, preset=None):
    st.subheader(f"🧩 Regras — {key_prefix}")
    selected = st.multiselect(
        "Variáveis que entram nas regras (ordem define os níveis da árvore):",
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
            # preset de regra específica, se houver
            preset_rule = None
            if preset and preset.get("rules"):
                for (pc, prt, pp) in preset["rules"]:
                    if pc == col:
                        preset_rule = (prt, pp); break

            if col in numeric_cols:
                cmin = float(np.nanmin(s.values)); cmax = float(np.nanmax(s.values))
                default_op = "≥ (mínimo)"; default_thr_ge = cmin; default_thr_le = cmax; default_between = (cmin, cmax)
                if preset_rule:
                    prt, pp = preset_rule
                    if prt == "num_ge": default_op, default_thr_ge = "≥ (mínimo)", float(pp)
                    elif prt == "num_le": default_op, default_thr_le = "≤ (máximo)", float(pp)
                    elif prt == "num_between": default_op, default_between = "entre", tuple(pp)

                op = st.radio(
                    f"Operador para `{col}`",
                    ["≥ (mínimo)", "≤ (máximo)", "entre"],
                    horizontal=True, key=f"{key_prefix}_op_{col}",
                    index=["≥ (mínimo)","≤ (máximo)","entre"].index(default_op)
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

# -----------------------------
# Árvore simples (Matplotlib)
# -----------------------------
def _col_pass_mask_for_rules(df, col, rules):
    """Avalia apenas as regras da coluna (AND entre regras da mesma coluna)."""
    sub = [r for r in rules if r[0] == col]
    if not sub:
        return pd.Series([True] * len(df), index=df.index)
    return df.apply(lambda r: apply_rules_row(r, sub, "AND (todas)"), axis=1)

def build_simple_tree_layers(df, rules, rule_order_cols, logic_mode, max_depth=None, min_samples=1):
    """
    Constrói camadas da árvore: lista de níveis; cada nível é lista de nós:
    nó = {"mask": Series[bool], "label": str, "edge": str ('Passa'/'Não passa' ou 'Raiz'), "depth": int}
    """
    if not rules or not rule_order_cols:
        return []

    if max_depth is None:
        max_depth = len(rule_order_cols)
    else:
        max_depth = max(0, min(max_depth, len(rule_order_cols)))

    def node_metrics(mask):
        sub = df[mask]
        n = int(len(sub))
        if n == 0:
            return 0, 0.0, 0.0
        aprovado_mask = sub.apply(lambda r: apply_rules_row(r, rules, logic_mode), axis=1)
        n_ap = int(aprovado_mask.sum())
        n_ng = n - n_ap
        pct_ap = (n_ap / n) * 100.0
        pct_ng = (n_ng / n) * 100.0
        return n, pct_ap, pct_ng

    # raiz
    root_mask = pd.Series([True] * len(df), index=df.index)
    layers = []
    n, p_ap, p_ng = node_metrics(root_mask)
    root_label = f"Raiz\nn={n} | %aprov={p_ap:.1f}% | %neg={p_ng:.1f}%"
    layers.append([{"mask": root_mask, "label": root_label, "edge": "Raiz", "depth": 0}])

    # níveis
    for depth in range(max_depth):
        col = rule_order_cols[depth]
        current_nodes = layers[-1]
        next_nodes = []

        for node in current_nodes:
            parent_mask = node["mask"]
            subset = df[parent_mask]
            if subset.empty:
                continue

            pass_mask_local = _col_pass_mask_for_rules(subset, col, rules)
            fail_mask_local = ~pass_mask_local

            for edge_txt, local_mask in [("Passa", pass_mask_local), ("Não passa", fail_mask_local)]:
                if not local_mask.any():
                    continue
                child_mask = parent_mask.copy()
                child_mask.loc[subset.index] = local_mask

                n_child, p_ap_child, p_ng_child = node_metrics(child_mask)
                if n_child < min_samples:
                    continue

                child_label = f"{col}: {edge_txt}\n" \
                              f"n={n_child} | %aprov={p_ap_child:.1f}% | %neg={p_ng_child:.1f}%"
                next_nodes.append({
                    "mask": child_mask,
                    "label": child_label,
                    "edge": edge_txt,
                    "depth": depth + 1
                })

        if next_nodes:
            layers.append(next_nodes)
        else:
            break

    return layers

def draw_simple_tree(layers, title=None, figsize=(12, 6)):
    """
    Desenha a árvore com Matplotlib: colunas por nível, nós como caixas e arestas.
    """
    if not layers:
        st.info("Defina variáveis e ordem de regras para ver a árvore.")
        return

    num_levels = len(layers)
    x_positions = list(range(num_levels))

    coords = {}
    for depth, nodes in enumerate(layers):
        k = len(nodes)
        ys = [0.5] if k == 1 else np.linspace(0.1, 0.9, k)
        for i, node in enumerate(nodes):
            coords[id(node)] = (x_positions[depth], ys[i])

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()
    ax.set_xlim(-0.5, num_levels - 0.5)
    ax.set_ylim(0.0, 1.0)

    # arestas
    for depth in range(num_levels - 1):
        parents = layers[depth]
        children = layers[depth + 1]
        for child in children:
            cx, cy = coords[id(child)]
            best_parent = None
            best_overlap = -1
            for parent in parents:
                overlap = int((child["mask"] & parent["mask"]).sum())
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_parent = parent
            if best_parent is None:
                continue
            px, py = coords[id(best_parent)]
            ax.plot([px + 0.18, cx - 0.18], [py, cy], linewidth=1.2)

    # nós
    for nodes in layers:
        for node in nodes:
            x, y = coords[id(node)]
            width, height = 0.36, 0.14
            box = FancyBboxPatch(
                (x - width/2, y - height/2),
                width, height,
                boxstyle="round,pad=0.02,rounding_size=0.02",
                linewidth=1.0
            )
            ax.add_patch(box)
            ax.text(x, y, node["label"], ha="center", va="center", fontsize=9)

    if title:
        ax.set_title(title, fontsize=12, pad=10)

    st.pyplot(fig, clear_figure=True)

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
    # Exportar (todas as configs)
    export_btn = st.button("⬇️ Exportar cenários (JSON)")
    if export_btn:
        payload = json.dumps(st.session_state.scenario_cfg, ensure_ascii=False, indent=2)
        st.download_button("Baixar arquivo", data=payload.encode("utf-8"),
                           file_name="cenarios.json", mime="application/json")
    # Importar
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

        # persistir config (para salvar/duplicar)
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

        # KPIs
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
            st.caption("🧭 Motivos da negativa (1ª regra que falha, ordem das variáveis).")
            neg = out.loc[out["motivo_negativa"] != "Aprovado", "motivo_negativa"].value_counts().reset_index()
            neg.columns = ["motivo", "qtd"]
            if not neg.empty:
                st.plotly_chart(px.bar(neg, x="motivo", y="qtd", title=f"Motivos da negativa — {name}"),
                                use_container_width=True)

        # Árvore simples (Matplotlib)
        st.caption("🌳 Árvore de decisões — visual simples (n, % aprov., % neg. por nó).")
        max_depth = st.slider(
            "Profundidade máx.", 1, len(selected_cols) if selected_cols else 1,
            min(3, len(selected_cols) if selected_cols else 1),
            key=f"depth_{name}"
        )
        min_samples = st.number_input(
            "Tamanho mínimo do nó (poda)", min_value=1, value=5, step=1, key=f"min_{name}"
        )
        layers = build_simple_tree_layers(
            base, rules, selected_cols, logic,
            max_depth=max_depth, min_samples=min_samples
        )
        draw_simple_tree(layers, title=f"Árvore — {name}", figsize=(12, 6))

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

    # ---- Agregações extras por cenário (configuráveis) ----
    st.markdown("### 🔧 Agregações extras por cenário (configuráveis)")
    num_opts = [c for c in numeric_cols if c != "aprovado"]
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
            for k, v in agg_res.items():
                colname = f"{k}"
                if colname not in cmp_df.columns:
                    cmp_df[colname] = np.nan
                    extra_cols.append(colname)
                cmp_df.loc[cmp_df["Cenário"] == scen_name, colname] = v

    # formatação amigável
    fmt = cmp_df.copy()
    for c in ["Público total (N)", "Pote Total (aprovados)"]:
        fmt[c] = fmt[c].map(lambda x: f"{int(x):,}".replace(",", "."))
    fmt["Taxa de aprovação (%)"] = fmt["Taxa de aprovação (%)"].map(lambda x: f"{x:.2f}%")
    money_cols = ["Impacto Aprovados (R$)", "Impacto Negados (R$)", "Ticket médio aprov. (R$)", "Ticket médio neg. (R$)"]
    for c in money_cols:
        fmt[c] = fmt[c].map(lambda x: f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    for c in extra_cols:
        fmt[c] = fmt[c].map(lambda x: "-" if pd.isna(x) else f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

    st.dataframe(fmt, use_container_width=True)

    # gráfico comparativo de taxa
    fig_cmp = px.bar(
        cmp_df.sort_values("Taxa de aprovação (%)", ascending=False),
        x="Cenário", y="Taxa de aprovação (%)",
        title="Taxa de aprovação por cenário"
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    st.download_button(
        "⬇️ Baixar comparativo (CSV)",
        data=cmp_df.to_csv(index=False).encode("utf-8"),
        file_name="comparativo_cenarios.csv",
        mime="text/csv"
    )

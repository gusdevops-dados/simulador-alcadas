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
st.title("📊 Simulador de Políticas e Alçadas — Múltiplos Cenários (grupos de regras)")

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

# Coluna financeira padrão (se não existir)
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

# Configurações atuais (UI) — agora com grupos
# scenario_cfg[name] = { "top_logic": "AND"/"OR", "groups": [ { "logic": "AND"/"OR", "rules": [(col, rtype, param)...], "order": [cols...] }, ... ] }
if "scenario_cfg" not in st.session_state:
    st.session_state.scenario_cfg = {}

# Resultados (após clicar no botão)
if "scenario_runs" not in st.session_state:
    st.session_state.scenario_runs = {}

# -----------------------------
# Engine de regras (com grupos)
# -----------------------------
def apply_rule_atom(val, rtype, param):
    """Aplica uma regra atômica a um único valor."""
    if rtype.startswith("num_"):
        if pd.isna(val): return False
        x = float(val)
        if rtype == "num_ge": return x >= float(param)
        if rtype == "num_le": return x <= float(param)
        if rtype == "num_between":
            lo, hi = param
            return (x >= float(lo)) and (x <= float(hi))
    elif rtype.startswith("bool_"):
        nb = normalize_bool(val)
        if nb is None: return False
        return (nb is True) if rtype == "bool_true" else (nb is False)
    elif rtype == "cat_in":
        v = str(val) if pd.notna(val) else ""
        return v in set(param)
    return False

def apply_rules_row_group(row, rules, group_logic):
    """Aplica as regras de UM grupo (AND/OR interno)."""
    if not rules:  # grupo vazio = passa
        return True
    results = []
    for (col, rtype, param) in rules:
        results.append(apply_rule_atom(row[col], rtype, param))
    return all(results) if group_logic == "AND" else any(results)

def apply_rules_row_groups(row, groups, top_logic):
    """Agrega os grupos sob o operador de topo (AND/OR)."""
    if not groups:  # sem grupos = passa
        return True
    group_results = []
    for g in groups:
        glog = g.get("logic", "AND")
        rules = g.get("rules", [])
        group_results.append(apply_rules_row_group(row, rules, glog))
    return all(group_results) if top_logic == "AND" else any(group_results)

def evaluate_scenario_grouped(df: pd.DataFrame, top_logic, groups):
    """Avalia cenário com grupos | retorna (out_df, metrics, ordem_para_arvore, motivo_negativa se aplicável)."""
    out = df.copy()
    out["aprovado"] = out.apply(lambda r: apply_rules_row_groups(r, groups, top_logic), axis=1)

    pote_total = int(out["aprovado"].sum())
    taxa = float(out["aprovado"].mean() * 100.0)
    vlr_aprov = float(out.loc[out["aprovado"], "vlr_contest"].sum())
    vlr_neg   = float(out.loc[~out["aprovado"], "vlr_contest"].sum())
    tk_aprov  = float(out.loc[out["aprovado"], "vlr_contest"].mean()) if pote_total > 0 else 0.0
    tk_neg    = float(out.loc[~out["aprovado"], "vlr_contest"].mean()) if pote_total < len(out) else 0.0

    # motivo da negativa (determinístico apenas se topo = AND)
    if groups and top_logic == "AND":
        # Primeiro grupo que falha; dentro dele, primeira coluna que falha
        def first_fail_reason(row):
            if row["aprovado"]: return "Aprovado"
            for gi, g in enumerate(groups, start=1):
                rules = g.get("rules", [])
                order = g.get("order", [r[0] for r in rules])
                glog = g.get("logic", "AND")
                # se grupo passa, continua
                if apply_rules_row_group(row, rules, glog):
                    continue
                # grupo falhou — achar 1ª coluna que falha na ordem declarada
                for col in order:
                    sub_rules = [r for r in rules if r[0] == col]
                    if not sub_rules:
                        continue
                    # avalia "AND" entre as regras da mesma coluna para explicar falha
                    # (se glog==OR, ainda assim usamos "AND por coluna" para diagnosticar o não atendimento da coluna)
                    passed_col = all(apply_rule_atom(row[col], rtype, param) for _, rtype, param in sub_rules)
                    if not passed_col:
                        return f"Falha no Grupo {gi}: {col}"
                return f"Falha no Grupo {gi}"
            return "Falha (outra)"
        out["motivo_negativa"] = out.apply(first_fail_reason, axis=1)
    else:
        out["motivo_negativa"] = np.where(out["aprovado"], "Aprovado", "Negado")

    # ordem de variáveis para árvore = concat das ordens de cada grupo (sem repetir, preservando ordem)
    order_for_tree = []
    seen = set()
    for g in groups:
        for col in g.get("order", []):
            if col not in seen:
                seen.add(col)
                order_for_tree.append(col)

    metrics = {
        "publico_total": int(len(df)),
        "pote_total": pote_total,
        "taxa_aprovacao_pct": taxa,
        "vlr_aprovado": vlr_aprov,
        "vlr_negado": vlr_neg,
        "ticket_medio_aprov": tk_aprov,
        "ticket_medio_neg": tk_neg
    }
    return out, metrics, order_for_tree

# -----------------------------
# UI de grupos de regras (com number_input)
# -----------------------------
def build_grouped_rules_ui(df: pd.DataFrame, scen_name: str, preset=None):
    """Retorna top_logic, groups (lista de dicts)"""
    st.subheader(f"🧩 Regras — {scen_name}")
    # lógica de topo
    top_logic = st.radio(
        "Agregação entre GRUPOS:",
        ["AND", "OR"],
        horizontal=True,
        index=0 if not preset else (0 if preset.get("top_logic","AND") == "AND" else 1),
        key=f"toplogic_{scen_name}"
    )

    # quantidade de grupos
    default_groups = preset.get("groups") if preset else None
    ng_default = len(default_groups) if default_groups else 2
    num_groups = st.number_input("Quantidade de grupos de regras", min_value=1, max_value=10, value=ng_default, step=1, key=f"ng_{scen_name}")

    groups = []
    for gi in range(int(num_groups)):
        grp_preset = default_groups[gi] if (default_groups and gi < len(default_groups)) else {}
        with st.container(border=True):
            st.markdown(f"**Grupo {gi+1}**")
            g_logic = st.radio(
                "Agregação DENTRO do grupo:",
                ["AND", "OR"],
                horizontal=True,
                index=0 if not grp_preset else (0 if grp_preset.get("logic","AND")=="AND" else 1),
                key=f"glogic_{scen_name}_{gi}"
            )
            # seleção de colunas deste grupo (ordem = níveis locais)
            selected = st.multiselect(
                "Variáveis deste grupo (ordem define precedência para motivo/árvore):",
                options=numeric_cols + bool_cols + cat_cols,
                default=(grp_preset.get("order") if grp_preset else []),
                key=f"gsel_{scen_name}_{gi}"
            )

            rules = []
            for col in selected:
                with st.container():
                    st.caption(f"• {col}")
                    s = df[col]
                    # preset da coluna
                    preset_rule = None
                    if grp_preset and grp_preset.get("rules"):
                        for (pc, prt, pp) in grp_preset["rules"]:
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
                            horizontal=True, key=f"{scen_name}_g{gi}_op_{col}",
                            index=["≥ (mínimo)","≤ (máximo)","entre"].index(default_op)
                        )
                        if op == "≥ (mínimo)":
                            thr = st.number_input(
                                "Valor mínimo", value=float(default_thr_ge), step=1.0,
                                key=f"{scen_name}_g{gi}_thr_ge_{col}",
                                help=f"Sugestão: mín={cmin}, máx={cmax}"
                            )
                            rules.append((col, "num_ge", thr))
                        elif op == "≤ (máximo)":
                            thr = st.number_input(
                                "Valor máximo", value=float(default_thr_le), step=1.0,
                                key=f"{scen_name}_g{gi}_thr_le_{col}",
                                help=f"Sugestão: mín={cmin}, máx={cmax}"
                            )
                            rules.append((col, "num_le", thr))
                        else:
                            c1, c2 = st.columns(2)
                            with c1:
                                lo = st.number_input(
                                    "Intervalo: mín", value=float(default_between[0]), step=1.0,
                                    key=f"{scen_name}_g{gi}_thr_between_lo_{col}",
                                    help=f"Observado: {cmin}–{cmax}"
                                )
                            with c2:
                                hi = st.number_input(
                                    "Intervalo: máx", value=float(default_between[1]), step=1.0,
                                    key=f"{scen_name}_g{gi}_thr_between_hi_{col}"
                                )
                            if hi < lo:
                                st.warning("⚠️ Máx < mín — ajustando.", icon="⚠️")
                                hi = lo
                            rules.append((col, "num_between", (lo, hi)))

                    elif col in bool_cols:
                        default_choice = "Exigir True"
                        if preset_rule:
                            prt, _ = preset_rule
                            default_choice = "Exigir True" if prt == "bool_true" else "Exigir False"
                        choice = st.radio(
                            f"Como tratar `{col}`?",
                            ["Exigir True", "Exigir False"],
                            horizontal=True, key=f"{scen_name}_g{gi}_bool_{col}",
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
                        sel_vals = st.multiselect(
                            f"Valores permitidos em `{col}`",
                            options=uniques, default=default_sel, key=f"{scen_name}_g{gi}_cat_{col}"
                        )
                        rules.append((col, "cat_in", sel_vals))

            groups.append({"logic": g_logic, "rules": rules, "order": selected})

    return top_logic, groups

# -----------------------------
# Árvore simples (Matplotlib) — reaproveitada
# -----------------------------
def _col_pass_mask_for_rules(df, col, rules):
    """Avalia apenas as regras daquela COLUNA (AND entre regras da mesma coluna)."""
    sub = [r for r in rules if r[0] == col]
    if not sub:
        return pd.Series([True] * len(df), index=df.index)
    # aplica AND entre as regras daquela coluna
    def pass_col(row):
        for _, rtype, param in sub:
            if not apply_rule_atom(row[col], rtype, param):
                return False
        return True
    return df.apply(pass_col, axis=1)

def build_simple_tree_layers(df, groups, order_for_tree, top_logic, max_depth=None, min_samples=1):
    """
    Constrói camadas da árvore considerando a ordem_for_tree (concat das ordens dos grupos).
    Cada nó mostra métricas condicionais calculadas com TODAS as regras (grupos + top_logic).
    """
    if not order_for_tree:
        return []

    if max_depth is None:
        max_depth = len(order_for_tree)
    else:
        max_depth = max(0, min(max_depth, len(order_for_tree)))

    def node_metrics(mask):
        sub = df[mask]
        n = int(len(sub))
        if n == 0:
            return 0, 0.0, 0.0
        aprovado_mask = sub.apply(lambda r: apply_rules_row_groups(r, groups, top_logic), axis=1)
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

    # níveis (seguindo order_for_tree)
    for depth in range(max_depth):
        col = order_for_tree[depth]
        current_nodes = layers[-1]
        next_nodes = []

        for node in current_nodes:
            parent_mask = node["mask"]
            subset = df[parent_mask]
            if subset.empty:
                continue

            # avalia somente as regras da COLUNA (AND por coluna)
            pass_mask_local = _col_pass_mask_for_rules(subset, col, sum((g["rules"] for g in groups), []))
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
    if not layers:
        st.info("Defina variáveis nas regras (ordem) para ver a árvore.")
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
            st.session_state.scenario_runs.pop(rem, None)
    with c3:
        src = st.selectbox("Duplicar de", ["(nenhum)"] + st.session_state.scenarios, index=0, key="dup_src")
        dst = st.text_input("Como", value="Cenário (cópia)")
        if st.button("📄 Duplicar") and src != "(nenhum)":
            target = dst.strip() or f"{src} (cópia)"
            if target not in st.session_state.scenarios:
                st.session_state.scenarios.append(target)
                st.session_state.scenario_cfg[target] = json.loads(json.dumps(st.session_state.scenario_cfg.get(src, {})))
                st.session_state.scenario_runs.pop(target, None)

    st.divider()
    st.subheader("💾 Salvar/Carregar cenários")
    export_btn = st.button("⬇️ Exportar cenários (JSON)")
    if export_btn:
        payload = json.dumps(st.session_state.scenario_cfg, ensure_ascii=False, indent=2)
        st.download_button("Baixar arquivo", data=payload.encode("utf-8"),
                           file_name="cenarios.json", mime="application/json")
    imp = st.file_uploader("Importar cenários (JSON)", type=["json"], key="import_json")
    if imp is not None:
        try:
            cfg = json.load(io.StringIO(imp.getvalue().decode("utf-8")))
            if isinstance(cfg, dict):
                st.session_state.scenario_cfg.update(cfg)
                for name in cfg.keys():
                    if name not in st.session_state.scenarios:
                        st.session_state.scenarios.append(name)
                st.success("Configurações importadas! (clique 'Gerar/Atualizar cenário' em cada aba)")
        except Exception as e:
            st.error(f"Falha ao importar: {e}")

# -----------------------------
# Abas de cenários
# -----------------------------
tabs = st.tabs(st.session_state.scenarios + ["📊 Comparar"])

for i, name in enumerate(st.session_state.scenarios):
    with tabs[i]:
        preset = st.session_state.scenario_cfg.get(name, {})
        top_logic, groups = build_grouped_rules_ui(base, name, preset=preset)

        current_cfg = {"top_logic": top_logic, "groups": groups}
        st.session_state.scenario_cfg[name] = current_cfg

        col_run, col_state = st.columns([1, 2])
        with col_run:
            run_clicked = st.button("⚙️ Gerar/Atualizar cenário", key=f"run_{name}", type="primary")
        with col_state:
            last_run = st.session_state.scenario_runs.get(name)
            if last_run is not None:
                if json.dumps(last_run["cfg_snapshot"], sort_keys=True, ensure_ascii=False) != json.dumps(current_cfg, sort_keys=True, ensure_ascii=False):
                    st.warning("Configuração alterada. Clique **Gerar/Atualizar cenário** para recomputar.", icon="⚠️")
                else:
                    st.success("Cenário está atualizado com a configuração atual.", icon="✅")
            else:
                st.info("Defina os grupos/regras e clique **Gerar/Atualizar cenário**.", icon="ℹ️")

        if run_clicked:
            with st.spinner(f"Calculando {name}..."):
                out, metrics, order_for_tree = evaluate_scenario_grouped(base, top_logic, groups)
            st.session_state.scenario_runs[name] = {
                "out": out, "metrics": metrics, "cfg_snapshot": current_cfg, "order_for_tree": order_for_tree
            }
            st.success("Cenário gerado com sucesso!", icon="✅")

        if name not in st.session_state.scenario_runs:
            continue

        pack = st.session_state.scenario_runs[name]
        out, metrics, order_for_tree = pack["out"], pack["metrics"], pack["order_for_tree"]

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

        # Motivos da negativa (somente se Topo = AND)
        if current_cfg["top_logic"] == "AND":
            st.caption("🧭 Motivos da negativa (1º grupo/variável que falha na ordem definida).")
            neg = out.loc[out["motivo_negativa"] != "Aprovado", "motivo_negativa"].value_counts().reset_index()
            neg.columns = ["motivo", "qtd"]
            if not neg.empty:
                st.plotly_chart(px.bar(neg, x="motivo", y="qtd", title=f"Motivos da negativa — {name}"),
                                use_container_width=True)

        # Árvore simples (usa ordem concatenada dos grupos)
        st.caption("🌳 Árvore de decisões — visual simples (n, % aprov., % neg. por nó).")
        max_depth = st.slider(
            "Profundidade máx.", 1, len(order_for_tree) if order_for_tree else 1,
            min(3, len(order_for_tree) if order_for_tree else 1),
            key=f"depth_{name}"
        )
        min_samples = st.number_input(
            "Tamanho mínimo do nó (poda)", min_value=1, value=5, step=1, key=f"min_{name}"
        )
        layers = build_simple_tree_layers(
            base, current_cfg["groups"], order_for_tree, current_cfg["top_logic"],
            max_depth=max_depth, min_samples=min_samples
        )
        draw_simple_tree(layers, title=f"Árvore — {name}", figsize=(12, 6))

        with st.expander(f"📋 Amostra de resultados ({name})", expanded=False):
            st.dataframe(out.head(30), use_container_width=True)

# -----------------------------
# Comparação de cenários + agregações configuráveis
# -----------------------------
with tabs[-1]:
    st.subheader("🧮 Comparação de Indicadores")
    outs = st.session_state.scenario_runs
    if not outs:
        st.info("Nenhum cenário processado ainda. Vá em uma aba e clique **Gerar/Atualizar cenário**.")
        st.stop()

    # KPIs base
    rows = []
    for name, pack in outs.items():
        m = pack["metrics"]
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

    # Agregações extras
    st.markdown("### 🔧 Agregações extras por cenário (configuráveis)")
    num_opts = [c for c in numeric_cols if c != "aprovado"]
    agg_vars = st.multiselect("Variáveis numéricas para agregar (ex.: 'vlr_invest')", options=num_opts)
    agg_funcs = st.multiselect("Funções", options=["sum","mean","median","max","min"], default=["sum"])
    pop = st.selectbox("População", ["Aprovados", "Negados", "Todos"], index=0)

    extra_cols = []
    if agg_vars and agg_funcs:
        for scen_name, pack in outs.items():
            out_df = pack["out"]
            if pop == "Aprovados":
                sub = out_df[out_df["aprovado"] == True]; suffix = "aprov"
            elif pop == "Negados":
                sub = out_df[out_df["aprovado"] == False]; suffix = "neg"
            else:
                sub = out_df; suffix = "todos"

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
                    cmp_df[colname] = np.nan; extra_cols.append(colname)
                cmp_df.loc[cmp_df["Cenário"] == scen_name, colname] = v

    # formatação
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

# app.py
import streamlit as st
import pandas as pd
import numpy as np
import json
import io
import plotly.express as px
import matplotlib.pyplot as plt  # opcional, caso use algum gráfico simples

st.set_page_config(page_title="Simulador de Alçadas — Cenários", layout="wide")
st.title("📊 Simulador de Políticas e Alçadas — Múltiplos Cenários (grupos + anualização)")

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

# ---------------------------------
# Helpers de tipo e datas
# ---------------------------------
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

def compute_annualization_factor(df: pd.DataFrame, date_col: str = "data_contest"):
    """
    Retorna (factor, info_str, valid):
      - factor = 365.25 / dias_cobertos (min..max), se possível; senão 1.0
      - info_str = texto amigável com período e fator
      - valid = True se anualização é aplicável
    """
    if date_col not in df.columns:
        return 1.0, "Coluna 'data_contest' não encontrada.", False
    dates = pd.to_datetime(df[date_col], errors="coerce")
    dates = dates.dropna()
    if dates.empty:
        return 1.0, "Não foi possível interpretar datas em 'data_contest'.", False
    dmin, dmax = dates.min(), dates.max()
    days = (dmax - dmin).days + 1
    if days <= 0:
        return 1.0, "Período inválido em 'data_contest'.", False
    factor = 365.25 / days
    info = f"Período detectado: {dmin.date()} → {dmax.date()} ({days} dias). Fator anualização ≈ {factor:.2f}x"
    return float(factor), info, True

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

# scenario_cfg[name] = { "top_logic": "AND"/"OR", "groups": [ { "logic": "AND"/"OR", "rules": [(col, rtype, param)...], "order": [cols...] }, ... ] }
if "scenario_cfg" not in st.session_state:
    st.session_state.scenario_cfg = {}

# name -> {"out": df, "metrics": dict, "cfg_snapshot": dict}
if "scenario_runs" not in st.session_state:
    st.session_state.scenario_runs = {}

# -----------------------------
# Engine de regras (com grupos)
# -----------------------------
def apply_rule_atom(val, rtype, param):
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
    if not rules:
        return True
    results = []
    for (col, rtype, param) in rules:
        results.append(apply_rule_atom(row[col], rtype, param))
    return all(results) if group_logic == "AND" else any(results)

def apply_rules_row_groups(row, groups, top_logic):
    if not groups:
        return True
    group_results = []
    for g in groups:
        glog = g.get("logic", "AND")
        rules = g.get("rules", [])
        group_results.append(apply_rules_row_group(row, rules, glog))
    return all(group_results) if top_logic == "AND" else any(group_results)

def evaluate_scenario_grouped(df: pd.DataFrame, top_logic, groups):
    out = df.copy()
    out["aprovado"] = out.apply(lambda r: apply_rules_row_groups(r, groups, top_logic), axis=1)

    # motivo da negativa (determinístico apenas se topo = AND)
    if groups and top_logic == "AND":
        def first_fail_reason(row):
            if row["aprovado"]: return "Aprovado"
            for gi, g in enumerate(groups, start=1):
                rules = g.get("rules", [])
                order = g.get("order", [r[0] for r in rules])
                glog = g.get("logic", "AND")
                if apply_rules_row_group(row, rules, glog):
                    continue
                for col in order:
                    sub_rules = [r for r in rules if r[0] == col]
                    if not sub_rules:
                        continue
                    passed_col = all(apply_rule_atom(row[col], rtype, param) for _, rtype, param in sub_rules)
                    if not passed_col:
                        return f"Falha no Grupo {gi}: {col}"
                return f"Falha no Grupo {gi}"
            return "Falha (outra)"
        out["motivo_negativa"] = out.apply(first_fail_reason, axis=1)
    else:
        out["motivo_negativa"] = np.where(out["aprovado"], "Aprovado", "Negado")

    # KPIs (nomenclaturas solicitadas)
    publico_total = int(len(df))                                  # N
    pote_total_money = float(df["vlr_contest"].sum())             # R$ total
    aprovados_n = int(out["aprovado"].sum())                      # N aprovados
    taxa_aprovacao_pct = float(out["aprovado"].mean() * 100.0)    # %

    perdas_aprovado_money = float(out.loc[out["aprovado"], "vlr_contest"].sum())   # R$ aprovados
    saving_negado_money   = float(out.loc[~out["aprovado"], "vlr_contest"].sum())  # R$ negados

    # Percentuais sobre o dinheiro total
    if pote_total_money > 0:
        perdas_aprovado_pct = perdas_aprovado_money / pote_total_money * 100.0
        saving_negado_pct   = saving_negado_money / pote_total_money * 100.0
    else:
        perdas_aprovado_pct = 0.0
        saving_negado_pct   = 0.0

    metrics = {
        "publico_total": publico_total,
        "pote_total": pote_total_money,
        "aprovados": aprovados_n,
        "taxa_aprovacao_pct": taxa_aprovacao_pct,
        "perdas_valor_aprovado_money": perdas_aprovado_money,
        "perdas_valor_aprovado_pct": perdas_aprovado_pct,
        "saving_valor_negado_money": saving_negado_money,
        "saving_valor_negado_pct": saving_negado_pct,
    }
    return out, metrics

# -----------------------------
# Anualização (aplicação em R$ e CONTAGENS)
# -----------------------------
MONEY_KEYS = [
    "pote_total",
    "perdas_valor_aprovado_money",
    "saving_valor_negado_money",
]
COUNT_KEYS = [
    "publico_total",
    "aprovados",
]

def apply_annualization_if_needed(metrics: dict, factor: float, enabled: bool):
    """
    Aplica anualização a valores monetários E contagens.
    Percentuais NÃO são anualizados.
    """
    if not enabled or factor is None or factor <= 0:
        return metrics, False
    m2 = metrics.copy()
    for k in MONEY_KEYS + COUNT_KEYS:
        if k in m2 and pd.notna(m2[k]):
            m2[k] = float(m2[k]) * float(factor)
    return m2, True

# -----------------------------
# UI de grupos de regras (com number_input)
# -----------------------------
def build_grouped_rules_ui(df: pd.DataFrame, scen_name: str, preset=None):
    st.subheader(f"🧩 Regras — {scen_name}")
    top_logic = st.radio(
        "Agregação entre GRUPOS:",
        ["AND", "OR"],
        horizontal=True,
        index=0 if not preset else (0 if preset.get("top_logic","AND") == "AND" else 1),
        key=f"toplogic_{scen_name}"
    )

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
            selected = st.multiselect(
                "Variáveis deste grupo (ordem local):",
                options=numeric_cols + bool_cols + cat_cols,
                default=(grp_preset.get("order") if grp_preset else []),
                key=f"gsel_{scen_name}_{gi}"
            )

            rules = []
            for col in selected:
                with st.container():
                    st.caption(f"• {col}")
                    s = df[col]
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

        # Anualização — fator com base na base inteira
        factor, factor_info, factor_ok = compute_annualization_factor(base, "data_contest")

        col_run, col_state, col_ann, col_dl = st.columns([1, 2, 1.8, 1.6])
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
        with col_ann:
            ann_enabled = st.checkbox("📈 Anualizar valores (R$ e contagens)", value=False, key=f"ann_{name}")
            if ann_enabled:
                if factor_ok:
                    st.caption(f"✅ {factor_info}")
                else:
                    st.caption(f"⚠️ {factor_info}")
        with col_dl:
            if name in st.session_state.scenario_runs:
                out_df = st.session_state.scenario_runs[name]["out"]
                csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Base analítica (CSV)",
                    data=csv_bytes,
                    file_name=f"base_analitica_{name}.csv",
                    mime="text/csv",
                    help="Exporta toda a base com as colunas originais + aprovado + motivo_negativa."
                )

        if run_clicked:
            with st.spinner(f"Calculando {name}..."):
                out, metrics = evaluate_scenario_grouped(base, top_logic, groups)
            st.session_state.scenario_runs[name] = {
                "out": out, "metrics": metrics, "cfg_snapshot": current_cfg
            }
            st.success("Cenário gerado com sucesso!", icon="✅")

        if name not in st.session_state.scenario_runs:
            continue

        pack = st.session_state.scenario_runs[name]
        out, metrics = pack["out"], pack["metrics"]

        # aplicar anualização (R$ + contagens) para exibição
        display_metrics, ann_applied = apply_annualization_if_needed(metrics, factor, ann_enabled and factor_ok)

        # ---------------- KPIs com nomenclaturas solicitadas ----------------
        st.subheader(f"📈 Indicadores — {name}" + (" (Anualizado)" if ann_applied else ""))
        k1, k2, k3, k4 = st.columns(4)
        k5, k6, k7, k8 = st.columns(4)

        # contagens: arredondar para inteiro na exibição
        k1.metric("Público total (N)", f"{int(round(display_metrics['publico_total'])):,}".replace(",", "."))
        k2.metric("Pote total (R$)", f"{display_metrics['pote_total']:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
        k3.metric("Aprovados (N)", f"{int(round(display_metrics['aprovados'])):,}".replace(",", "."))
        k4.metric("Taxa de aprovação (%)", f"{display_metrics['taxa_aprovacao_pct']:.2f}%")

        k5.metric("Perdas (valor aprovado) R$", f"{display_metrics['perdas_valor_aprovado_money']:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
        k6.metric("Perdas (valor aprovado) %", f"{display_metrics['perdas_valor_aprovado_pct']:.2f}%")
        k7.metric("Saving (valor negado) R$", f"{display_metrics['saving_valor_negado_money']:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
        k8.metric("Saving (valor negado) %", f"{display_metrics['saving_valor_negado_pct']:.2f}%")

        # Aprovados vs Negados (contagem) — sempre sem anualização (é distribuição)
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

        with st.expander(f"📋 Amostra da base analítica ({name})", expanded=False):
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

    # anualização global opcional (para a tabela)
    factor, factor_info, factor_ok = compute_annualization_factor(base, "data_contest")
    ann_global = st.checkbox("📈 Anualizar valores (R$ e contagens) na comparação", value=False)
    if ann_global:
        if factor_ok:
            st.caption(f"✅ {factor_info}")
        else:
            st.caption(f"⚠️ {factor_info}")

    rows = []
    for name, pack in outs.items():
        m = pack["metrics"].copy()
        if ann_global and factor_ok:
            m, _ = apply_annualization_if_needed(m, factor, True)
        rows.append({
            "Cenário": name,
            "Público total (N)": m["publico_total"],
            "Pote total (R$)": m["pote_total"],
            "Aprovados (N)": m["aprovados"],
            "Taxa de aprovação (%)": m["taxa_aprovacao_pct"],
            "Perdas (valor aprovado) R$": m["perdas_valor_aprovado_money"],
            "Perdas (valor aprovado) %": m["perdas_valor_aprovado_pct"],
            "Saving (valor negado) R$": m["saving_valor_negado_money"],
            "Saving (valor negado) %": m["saving_valor_negado_pct"],
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

    # formatação amigável
    fmt = cmp_df.copy()
    # contagens → arredonda para inteiro
    for c in ["Público total (N)", "Aprovados (N)"]:
        fmt[c] = fmt[c].map(lambda x: f"{int(round(x)):,}".replace(",", ".") if pd.notna(x) else "-")
    # dinheiro
    money_cols = ["Pote total (R$)", "Perdas (valor aprovado) R$", "Saving (valor negado) R$"]
    for c in money_cols:
        fmt[c] = fmt[c].map(lambda x: f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") if pd.notna(x) else "-")
    # percentuais
    pct_cols = ["Taxa de aprovação (%)", "Perdas (valor aprovado) %", "Saving (valor negado) %"]
    for c in pct_cols:
        fmt[c] = fmt[c].map(lambda x: f"{x:.2f}%" if pd.notna(x) else "-")
    # extras
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

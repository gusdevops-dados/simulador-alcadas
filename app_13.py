# app_optimized.py
import json
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ============================
# CONFIG INICIAL
# ============================
st.set_page_config(page_title="Simulador de Alçadas — Otimizado", layout="wide")
st.title("📊 Simulador de Políticas e Alçadas — Otimizado (cache + execução só no botão)")

# ============================
# UPLOAD
# ============================
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

# Coluna financeira padrão (se não existir)
if "vlr_contest" not in base.columns:
    base["vlr_contest"] = 0.0

# ============================
# HELPERS — TIPO & DATAS
# ============================
@st.cache_data
def detect_columns_once(df: pd.DataFrame):
    def is_bool_series(s: pd.Series) -> bool:
        if pd.api.types.is_bool_dtype(s): return True
        if pd.api.types.is_object_dtype(s):
            vals = set(str(x).strip().lower() for x in s.dropna().unique())
            cand = {"s","n","sim","nao","não","true","false","0","1"}
            return len(vals) <= 2 and vals.issubset(cand)
        return False

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    bool_cols    = [c for c in df.columns if is_bool_series(df[c])]
    cat_cols     = [c for c in df.columns
                    if (pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c]))
                    and c not in bool_cols]
    return numeric_cols, bool_cols, cat_cols

numeric_cols, bool_cols, cat_cols = detect_columns_once(base)

def parse_dates_series(s: pd.Series, mode: str, custom_fmt: str | None = None) -> pd.Series:
    if mode == "auto":
        return pd.to_datetime(s, errors="coerce", utc=False)
    elif mode == "dayfirst":
        return pd.to_datetime(s, errors="coerce", dayfirst=True, utc=False)
    elif mode == "monthfirst":
        return pd.to_datetime(s, errors="coerce", dayfirst=False, utc=False)
    elif mode == "custom" and custom_fmt:
        return pd.to_datetime(s, errors="coerce", format=custom_fmt, utc=False)
    else:
        return pd.to_datetime(s, errors="coerce", utc=False)

def compute_annualization_factor_with_mode(df: pd.DataFrame, date_col: str, mode: str, custom_fmt: str | None = None):
    if date_col not in df.columns:
        return 1.0, "Coluna 'data_contest' não encontrada.", False, 0
    dates = parse_dates_series(df[date_col], mode=mode, custom_fmt=custom_fmt)
    n_invalid = int(dates.isna().sum())
    dates = dates.dropna()
    if dates.empty:
        return 1.0, "Não foi possível interpretar datas em 'data_contest'.", False, n_invalid
    dmin, dmax = dates.min(), dates.max()
    days = (dmax - dmin).days + 1
    if days <= 0:
        return 1.0, "Período inválido em 'data_contest'.", False, n_invalid
    factor = 365.25 / days
    info = f"Período detectado: {dmin.date()} → {dmax.date()} ({days} dias). Fator ≈ {factor:.2f}x"
    if n_invalid > 0:
        info += f" — {n_invalid} registros sem data válida ignorados."
    return float(factor), info, True, n_invalid

# ============================
# ENGINE — VETORIZADO + CACHE
# ============================
def normalize_bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s
    s_str = s.astype(str).str.strip().str.lower()
    mapping = {
        "s": True, "sim": True, "true": True, "1": True,
        "n": False, "nao": False, "não": False, "false": False, "0": False
    }
    out = s_str.map(mapping)
    return out

def build_mask_for_rule(df: pd.DataFrame, col: str, rtype: str, param):
    s = df[col]
    if rtype.startswith("num_"):
        s_num = pd.to_numeric(s, errors="coerce")
        if rtype == "num_ge":
            return s_num.ge(float(param)).fillna(False)
        if rtype == "num_le":
            return s_num.le(float(param)).fillna(False)
        if rtype == "num_between":
            lo, hi = param
            return s_num.ge(float(lo)) & s_num.le(float(hi))
    elif rtype.startswith("bool_"):
        s_bool = normalize_bool_series(s)
        if rtype == "bool_true":
            return s_bool.eq(True).fillna(False)
        else:
            return s_bool.eq(False).fillna(False)
    elif rtype == "cat_in":
        vals = set(param or [])
        s_str = s.astype(str)
        return s_str.isin(vals)
    return pd.Series(False, index=df.index)

def build_group_mask(df: pd.DataFrame, group: dict):
    logic = group.get("logic", "AND")
    rules = group.get("rules", [])
    if not rules:
        return pd.Series(True, index=df.index)
    masks = [build_mask_for_rule(df, col, rtype, param) for (col, rtype, param) in rules]
    if logic == "AND":
        m = masks[0].copy()
        for mk in masks[1:]:
            m &= mk
        return m
    else:
        m = masks[0].copy()
        for mk in masks[1:]:
            m |= mk
        return m

def evaluate_vectorized(df: pd.DataFrame, top_logic: str, groups: list):
    """Retorna (out_df, metrics) — 100% vetorizado."""
    if not groups:
        aprovado_mask = pd.Series(True, index=df.index)
    else:
        group_masks = [build_group_mask(df, g) for g in groups]
        if top_logic == "AND":
            m = group_masks[0].copy()
            for gm in group_masks[1:]:
                m &= gm
            aprovado_mask = m
        else:
            m = group_masks[0].copy()
            for gm in group_masks[1:]:
                m |= gm
            aprovado_mask = m

    out = df.copy()
    out["aprovado"] = aprovado_mask.fillna(False)

    # Motivo da negativa (determinístico só quando topo=AND) — explicação simplificada e rápida.
    if groups and top_logic == "AND":
        motivo = np.full(len(out), "Aprovado", dtype=object)
        neg_idx = out.index[~out["aprovado"]]
        if len(neg_idx) > 0:
            # Para negativos, procura 1º grupo que falha e 1ª coluna daquele grupo que falha
            for gi, g in enumerate(groups, start=1):
                gm = build_group_mask(out.loc[neg_idx], g)
                # linhas que FALHARAM neste grupo
                fail_here = neg_idx[~gm]
                if len(fail_here) == 0:
                    continue
                # tenta primeiro por ordem declarada do grupo
                order = g.get("order", [r[0] for r in g.get("rules", [])])
                for col in order:
                    # coluna falhou? => testa as regras desta coluna
                    sub_rules = [r for r in g.get("rules", []) if r[0] == col]
                    if not sub_rules:
                        continue
                    col_mask = pd.Series(True, index=fail_here)
                    for (_, rtype, param) in sub_rules:
                        col_mask &= build_mask_for_rule(out.loc[fail_here], col, rtype, param)
                    fail_col = fail_here[~col_mask]
                    motivo_idx = out.index.get_indexer(fail_col)
                    motivo[motivo_idx] = f"Falha no Grupo {gi}: {col}"
                # qualquer resto não explicado recebe o grupo
                motivo_idx = out.index.get_indexer(fail_here)
                for idx in motivo_idx:
                    if motivo[idx] == "Aprovado":
                        motivo[idx] = f"Falha no Grupo {gi}"
        out["motivo_negativa"] = motivo
    else:
        out["motivo_negativa"] = np.where(out["aprovado"], "Aprovado", "Negado")

    # KPIs
    publico_total = int(len(df))
    pote_total_money = float(df["vlr_contest"].sum())
    aprovados_n = int(out["aprovado"].sum())
    taxa_aprovacao_pct = float(out["aprovado"].mean() * 100.0)
    perdas_aprovado_money = float(out.loc[out["aprovado"], "vlr_contest"].sum())
    saving_negado_money = float(out.loc[~out["aprovado"], "vlr_contest"].sum())
    if pote_total_money > 0:
        perdas_aprovado_pct = perdas_aprovado_money / pote_total_money * 100.0
        saving_negado_pct = saving_negado_money / pote_total_money * 100.0
    else:
        perdas_aprovado_pct = 0.0
        saving_negado_pct = 0.0

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

@st.cache_data(show_spinner=False)
def cached_evaluate(df: pd.DataFrame, cfg_str: str):
    """Empacota o cálculo para cachear por configuração (hashável)."""
    cfg = json.loads(cfg_str)
    top_logic = cfg["top_logic"]
    groups = cfg["groups"]
    return evaluate_vectorized(df, top_logic, groups)

# ============================
# ESTADO GLOBAL
# ============================
if "scenarios" not in st.session_state:
    st.session_state.scenarios = ["Cenário A", "Cenário B"]
if "scenario_cfg" not in st.session_state:
    st.session_state.scenario_cfg = {}
if "scenario_runs" not in st.session_state:
    # name -> {"out": df, "metrics": dict, "cfg_snapshot": dict}
    st.session_state.scenario_runs = {}

# ============================
# UI — FUNÇÕES
# ============================
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
    num_groups = st.number_input("Quantidade de grupos de regras", min_value=1, max_value=10,
                                 value=ng_default, step=1, key=f"ng_{scen_name}")

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
                st.caption(f"• {col}")
                s = df[col]
                preset_rule = None
                if grp_preset and grp_preset.get("rules"):
                    for (pc, prt, pp) in grp_preset["rules"]:
                        if pc == col:
                            preset_rule = (prt, pp); break

                if col in numeric_cols:
                    cmin = float(pd.to_numeric(s, errors="coerce").min())
                    cmax = float(pd.to_numeric(s, errors="coerce").max())
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
                        thr = st.number_input("Valor mínimo", value=float(default_thr_ge), step=1.0,
                                              key=f"{scen_name}_g{gi}_thr_ge_{col}")
                        rules.append((col, "num_ge", thr))
                    elif op == "≤ (máximo)":
                        thr = st.number_input("Valor máximo", value=float(default_thr_le), step=1.0,
                                              key=f"{scen_name}_g{gi}_thr_le_{col}")
                        rules.append((col, "num_le", thr))
                    else:
                        c1, c2 = st.columns(2)
                        with c1:
                            lo = st.number_input("Intervalo: mín", value=float(default_between[0]), step=1.0,
                                                 key=f"{scen_name}_g{gi}_thr_between_lo_{col}")
                        with c2:
                            hi = st.number_input("Intervalo: máx", value=float(default_between[1]), step=1.0,
                                                 key=f"{scen_name}_g{gi}_thr_between_hi_{col}")
                        if hi < lo:
                            st.warning("⚠️ Máx < mín — ajustando.")
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
                    sel_vals = st.multiselect(f"Valores permitidos em `{col}`", options=uniques,
                                              default=default_sel, key=f"{scen_name}_g{gi}_cat_{col}")
                    rules.append((col, "cat_in", sel_vals))

            groups.append({"logic": g_logic, "rules": rules, "order": selected})
    return top_logic, groups

# ============================
# SIDEBAR — GERENCIAR CENÁRIOS
# ============================
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

# ============================
# ABAS DE CENÁRIOS
# ============================
tabs = st.tabs(st.session_state.scenarios + ["📊 Comparar"])

for i, name in enumerate(st.session_state.scenarios):
    with tabs[i]:
        preset = st.session_state.scenario_cfg.get(name, {})
        top_logic, groups = build_grouped_rules_ui(base, name, preset=preset)
        current_cfg = {"top_logic": top_logic, "groups": groups}
        st.session_state.scenario_cfg[name] = current_cfg

        # UI de anualização — seletor de parsing + toggle
        st.markdown("### 🗓️ Anualização")
        opt = st.selectbox(
            "Como interpretar `data_contest`?",
            ["Detectar automaticamente", "DD/MM/AAAA (Brasil)", "MM/DD/AAAA (EUA)", "Formato personalizado"],
            index=1, key=f"date_mode_{name}"
        )
        mode_map = {
            "Detectar automaticamente": "auto",
            "DD/MM/AAAA (Brasil)": "dayfirst",
            "MM/DD/AAAA (EUA)": "monthfirst",
            "Formato personalizado": "custom"
        }
        mode = mode_map[opt]
        custom_fmt = None
        if mode == "custom":
            custom_fmt = st.text_input("Informe o formato strptime (ex.: %d/%m/%Y)", value="%d/%m/%Y", key=f"custom_fmt_{name}")

        factor, factor_info, factor_ok, _ = compute_annualization_factor_with_mode(base, "data_contest", mode=mode, custom_fmt=custom_fmt)

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
                st.caption(("✅ " if factor_ok else "⚠️ ") + factor_info)
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

        # >>> EXECUÇÃO SOMENTE QUANDO CLICA NO BOTÃO <<<
        if run_clicked:
            with st.spinner(f"Calculando {name}..."):
                cfg_str = json.dumps(current_cfg, ensure_ascii=False)
                out, metrics = cached_evaluate(base, cfg_str)
            st.session_state.scenario_runs[name] = {
                "out": out, "metrics": metrics, "cfg_snapshot": current_cfg
            }
            st.success("Cenário gerado com sucesso!", icon="✅")

        # Se ainda não rodou, nada é recalculado/mostrado
        if name not in st.session_state.scenario_runs:
            st.stop()

        pack = st.session_state.scenario_runs[name]
        out, metrics = pack["out"], pack["metrics"]

        # anualização (R$ + contagens)
        MONEY_KEYS = ["pote_total", "perdas_valor_aprovado_money", "saving_valor_negado_money"]
        COUNT_KEYS = ["publico_total", "aprovados"]

        display_metrics = metrics.copy()
        ann_applied = False
        if ann_enabled and factor_ok and factor > 0:
            for k in MONEY_KEYS + COUNT_KEYS:
                display_metrics[k] = float(display_metrics[k]) * float(factor)
            ann_applied = True

        # KPIs
        st.subheader(f"📈 Indicadores — {name}" + (" (Anualizado)" if ann_applied else ""))
        k1, k2, k3, k4 = st.columns(4)
        k5, k6, k7, k8 = st.columns(4)
        k1.metric("Público total (N)", f"{int(round(display_metrics['publico_total'])):,}".replace(",", "."))
        k2.metric("Pote total (R$)", f"{display_metrics['pote_total']:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
        k3.metric("Aprovados (N)", f"{int(round(display_metrics['aprovados'])):,}".replace(",", "."))
        k4.metric("Taxa de aprovação (%)", f"{display_metrics['taxa_aprovacao_pct']:.2f}%")
        k5.metric("Perdas (valor aprovado) R$", f"{display_metrics['perdas_valor_aprovado_money']:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
        k6.metric("Perdas (valor aprovado) %", f"{display_metrics['perdas_valor_aprovado_pct']:.2f}%")
        k7.metric("Saving (valor negado) R$", f"{display_metrics['saving_valor_negado_money']:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
        k8.metric("Saving (valor negado) %", f"{display_metrics['saving_valor_negado_pct']:.2f}%")

        # Gráficos (leves, só usam 'out' já pronto)
        vc = out["aprovado"].value_counts().rename({True: "Aprovado", False: "Negado"}).reset_index()
        vc.columns = ["status", "qtd"]
        st.plotly_chart(px.bar(vc, x="status", y="qtd", title=f"Aprovados vs Negados — {name}"), use_container_width=True)

        if pack["cfg_snapshot"]["top_logic"] == "AND":
            st.caption("🧭 Motivos da negativa (1º grupo/variável que falha).")
            neg = out.loc[out["motivo_negativa"] != "Aprovado", "motivo_negativa"].value_counts().reset_index()
            neg.columns = ["motivo", "qtd"]
            if not neg.empty:
                st.plotly_chart(px.bar(neg, x="motivo", y="qtd", title=f"Motivos da negativa — {name}"),
                                use_container_width=True)

        with st.expander(f"📋 Amostra da base analítica ({name})", expanded=False):
            st.dataframe(out.head(30), use_container_width=True)

# ============================
# COMPARAR CENÁRIOS
# ============================
with tabs[-1]:
    st.subheader("🧮 Comparação de Indicadores")
    outs = st.session_state.scenario_runs
    if not outs:
        st.info("Nenhum cenário processado ainda. Vá em uma aba e clique **Gerar/Atualizar cenário**.")
        st.stop()

    # anualização global
    opt_global = st.selectbox(
        "Como interpretar `data_contest` na comparação?",
        ["Detectar automaticamente", "DD/MM/AAAA (Brasil)", "MM/DD/AAAA (EUA)", "Formato personalizado"],
        index=1, key="date_mode_global"
    )
    mode_global = {"Detectar automaticamente": "auto",
                   "DD/MM/AAAA (Brasil)": "dayfirst",
                   "MM/DD/AAAA (EUA)": "monthfirst",
                   "Formato personalizado": "custom"}[opt_global]
    custom_fmt_global = st.text_input("Formato customizado (comparação)", value="%d/%m/%Y") if mode_global == "custom" else None

    factor, factor_info, factor_ok, _ = compute_annualization_factor_with_mode(
        base, "data_contest", mode=mode_global, custom_fmt=custom_fmt_global
    )
    ann_global = st.checkbox("📈 Anualizar valores (R$ e contagens) na comparação", value=False)
    if ann_global:
        st.caption(("✅ " if factor_ok else "⚠️ ") + factor_info)

    rows = []
    for name, pack in outs.items():
        m = pack["metrics"].copy()
        if ann_global and factor_ok and factor > 0:
            for key in ["pote_total", "perdas_valor_aprovado_money", "saving_valor_negado_money",
                        "publico_total", "aprovados"]:
                m[key] = float(m[key]) * float(factor)
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

    # Agregações extras
    st.markdown("### 🔧 Agregações extras por cenário (configuráveis)")
    num_opts = [c for c in numeric_cols if c not in {"aprovado"}]
    agg_vars = st.multiselect("Variáveis numéricas para agregar (ex.: 'vlr_invest')", options=num_opts)
    agg_funcs = st.multiselect("Funções", options=["sum","mean","median","max","min"], default=["sum"])
    pop = st.selectbox("População", ["Aprovados", "Negados", "Todos"], index=0)

    if agg_vars and agg_funcs:
        extra_cols = []
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

    # Formatação amigável
    fmt = cmp_df.copy()
    for c in ["Público total (N)", "Aprovados (N)"]:
        fmt[c] = fmt[c].map(lambda x: f"{int(round(x)):,}".replace(",", ".") if pd.notna(x) else "-")
    money_cols = ["Pote total (R$)", "Perdas (valor aprovado) R$", "Saving (valor negado) R$"]
    for c in money_cols:
        if c in fmt:
            fmt[c] = fmt[c].map(lambda x: f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") if pd.notna(x) else "-")
    pct_cols = ["Taxa de aprovação (%)", "Perdas (valor aprovado) %", "Saving (valor negado) %"]
    for c in pct_cols:
        if c in fmt:
            fmt[c] = fmt[c].map(lambda x: f"{x:.2f}%" if pd.notna(x) else "-")

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

# ============================
# PRÉVIA DA BASE (LAZY)
# ============================
with st.expander("🔍 Prévia da base (20 linhas)", expanded=False):
    st.dataframe(base.head(20), use_container_width=True)

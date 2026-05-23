# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo",
#   "pandas>=2.0",
#   "numpy>=1.26",
#   "pyarrow>=14.0",
#   "scipy>=1.11",
#   "statsmodels>=0.14",
#   "matplotlib>=3.8",
#   "seaborn>=0.13",
#   "scikit-learn>=1.4",
# ]
# ///_

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium", app_title="02 · Panel Logit — PD Estimation")


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # 📐 Panel Logit — PD Estimation (SBS Ecuador)
        Estimación de la Probabilidad de Default con modelos de panel:
        **Logit Pooled**, **Fixed Effects (LSDV)** y **Random Effects (GEE)**.
        Selección mediante **Hausman test**.
        """
    )
    return


@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import warnings
    from pathlib import Path
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import acorr_breusch_godfrey
    from statsmodels.stats.stattools import durbin_watson
    from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
    from scipy import stats

    warnings.filterwarnings("ignore")
    ROOT = Path(".").resolve().parents[1] if "notebooks" in str(Path(".").resolve()) else Path(".")
    DATA_DIR = ROOT / "02_pd_estimation" / "data"
    REPORTS_DIR = ROOT / "02_pd_estimation" / "reports"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    PALETTE = {"primary": "#01696f", "secondary": "#437a22", "warning": "#d19900", "error": "#a12c7b"}
    print("✅ Imports OK")
    return (
        DATA_DIR, PALETTE, REPORTS_DIR, ROOT, acorr_breusch_godfrey,
        brier_score_loss, durbin_watson, gridspec, mo, np, pd, plt,
        roc_auc_score, roc_curve, sm, sns, stats, warnings,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md("## 1 · Cargar panel de features")
    return


@app.cell
def __(DATA_DIR, pd):
    df = pd.read_parquet(DATA_DIR / "panel_features.parquet")
    print(f"Panel features: {df.shape}")
    return (df,)


@app.cell(hide_code=True)
def __(mo):
    mo.md("## 2 · Especificación del modelo")
    return


@app.cell
def __(df, np, sm):
    # Features seleccionadas — parsimonia + teoría económica
    FEATURES = [
        "tasa_activa_lag1",     # H1: costo financiero rezagado
        "pib_gap_hp",           # H2: ciclo económico
        "desempleo_lag1",       # H3: mercado laboral
        "inflacion_lag1",       # H4: erosión ingreso real
        "morosidad_lag1",       # AR(1): persistencia
        "dummy_covid",          # H5: shock exógeno
        "dummy_oilshock",       # H6: shock externo
        "seg_consumo",          # FE: segmento consumo
        "seg_inmobiliario",     # FE: segmento inmobiliario
        "seg_microcredito",     # FE: segmento microcrédito
    ]
    TARGET = "default_bin"

    # Split temporal: train 2015-2021, OOT 2022-2024
    train = df[df["fecha"] < "2022-01"].copy()
    oot   = df[df["fecha"] >= "2022-01"].copy()

    X_train = sm.add_constant(train[FEATURES])
    y_train = train[TARGET]
    X_oot   = sm.add_constant(oot[FEATURES])
    y_oot   = oot[TARGET]

    print(f"Train: {train.shape} | OOT: {oot.shape}")
    print(f"Train default rate: {y_train.mean():.3f} | OOT: {y_oot.mean():.3f}")
    return FEATURES, TARGET, X_oot, X_train, oot, train, y_oot, y_train


@app.cell(hide_code=True)
def __(mo):
    mo.md("## 3 · Logit Pooled (statsmodels — con inferencia completa)")
    return


@app.cell
def __(X_train, sm, y_train):
    logit_pooled = sm.Logit(y_train, X_train).fit(method="bfgs", disp=False)
    print(logit_pooled.summary())
    return (logit_pooled,)


@app.cell(hide_code=True)
def __(mo):
    mo.md("## 4 · Logit Fixed Effects (LSDV — Least Squares Dummy Variables)")
    return


@app.cell
def __(FEATURES, pd, sm, train, y_train):
    # LSDV: incluye dummies de tiempo (año) como FE adicionales
    time_dummies = pd.get_dummies(train["fecha"].dt.year, prefix="yr", drop_first=True).astype(int)
    X_fe = pd.concat([
        pd.DataFrame({"const": 1}, index=train.index),
        train[FEATURES],
        time_dummies
    ], axis=1)

    logit_fe = sm.Logit(y_train, X_fe).fit(method="bfgs", maxiter=200, disp=False)
    print("\n=== LOGIT FIXED EFFECTS (LSDV) ===")
    # Mostrar solo coeficientes principales (no time dummies)
    main_coefs = logit_fe.params[logit_fe.params.index.str.startswith("yr") == False]
    print(main_coefs.round(4))
    print(f"\nLog-Likelihood FE: {logit_fe.llf:.2f}")
    print(f"McFadden R²: {logit_fe.prsquared:.4f}")
    return X_fe, logit_fe, main_coefs, time_dummies


@app.cell(hide_code=True)
def __(mo):
    mo.md("## 5 · Hausman-like test (LR test Pooled vs FE)")
    return


@app.cell
def __(logit_fe, logit_pooled, stats):
    # LR test: 2*(llf_unrestricto - llf_restricto) ~ chi2(df)
    lr_stat = 2 * (logit_fe.llf - logit_pooled.llf)
    df_diff = logit_fe.df_model - logit_pooled.df_model
    p_hausman = 1 - stats.chi2.cdf(lr_stat, df=max(df_diff, 1))

    verdict = "✅ FIXED EFFECTS preferido (FE consistente)" if p_hausman < 0.05 else "✅ POOLED aceptable (efectos no correlacionados)"
    print("=== LR TEST: Pooled vs Fixed Effects ===")
    print(f"LR statistic: {lr_stat:.4f}")
    print(f"Grados de libertad: {df_diff}")
    print(f"p-value: {p_hausman:.4f}")
    print(f"Decisión: {verdict}")
    return df_diff, lr_stat, p_hausman, verdict


@app.cell(hide_code=True)
def __(mo):
    mo.md("## 6 · Métricas de discriminación — Train + OOT")
    return


@app.cell
def __(X_oot, X_train, brier_score_loss, logit_fe, roc_auc_score, y_oot, y_train):
    def eval_model(model, X, y, split_name):
        pred_prob = model.predict(X)
        auc  = roc_auc_score(y, pred_prob)
        gini = 2 * auc - 1
        brier = brier_score_loss(y, pred_prob)
        # KS
        from scipy.stats import ks_2samp
        ks = ks_2samp(pred_prob[y == 1], pred_prob[y == 0]).statistic
        mcfadden = model.prsquared

        thresholds = {
            "KS":       (ks,       0.30, "≥30%"),
            "Gini":     (gini,     0.40, "≥40%"),
            "AUC":      (auc,      0.70, "≥0.70"),
            "McFadden": (mcfadden, 0.20, "≥0.20"),
            "Brier":    (brier,    0.25, "≤0.25"),
        }
        print(f"\n{'='*40}")
        print(f"  {split_name}")
        print(f"{'='*40}")
        for name, (val, thr, label) in thresholds.items():
            if name == "Brier":
                flag = "✅" if val <= thr else "⚠️"
            else:
                flag = "✅" if val >= thr else "⚠️"
            print(f"  {flag} {name:10s}: {val:.4f}  (Basel threshold: {label})")
        return pred_prob

    pred_train = eval_model(logit_fe, X_train, y_train, "TRAIN (2015-2021)")
    pred_oot   = eval_model(logit_fe, X_oot,   y_oot,   "OOT  (2022-2024)")
    return eval_model, pred_oot, pred_train


@app.cell(hide_code=True)
def __(mo):
    mo.md("## 7 · ROC Curve + Score Distribution + Coeficientes")
    return


@app.cell
def __(PALETTE, logit_fe, np, plt, pred_oot, pred_train, roc_curve, y_oot, y_train):
    fig = plt.figure(figsize=(16, 12), facecolor="white")
    gs  = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35)
    fig.suptitle("Panel Logit — PD Estimation · SBS Ecuador", fontsize=14, fontweight="bold")

    # ── A) ROC Curves
    ax_roc = fig.add_subplot(gs[0, 0])
    for preds, y, lbl, col in [
        (pred_train, y_train, "Train 2015-21", PALETTE["primary"]),
        (pred_oot,   y_oot,   "OOT  2022-24",  PALETTE["error"]),
    ]:
        from sklearn.metrics import roc_auc_score
        fpr, tpr, _ = roc_curve(y, preds)
        auc = roc_auc_score(y, preds)
        ax_roc.plot(fpr, tpr, color=col, lw=2, label=f"{lbl} (AUC={auc:.3f})")
    ax_roc.plot([0,1],[0,1],"--",color="gray",lw=1)
    ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR")
    ax_roc.set_title("ROC Curve", fontweight="bold")
    ax_roc.legend(fontsize=9)
    ax_roc.spines[["top","right"]].set_visible(False)

    # ── B) Score distributions
    ax_dist = fig.add_subplot(gs[0, 1])
    ax_dist.hist(pred_train[y_train==0], bins=25, alpha=0.6, color=PALETTE["secondary"], label="No-Default")
    ax_dist.hist(pred_train[y_train==1], bins=25, alpha=0.6, color=PALETTE["error"],    label="Default")
    ax_dist.set_title("Score Distribution (Train)", fontweight="bold")
    ax_dist.set_xlabel("PD estimada")
    ax_dist.legend(fontsize=9)
    ax_dist.spines[["top","right"]].set_visible(False)

    # ── C) Coeficientes con IC95%
    ax_coef = fig.add_subplot(gs[0, 2])
    main_idx = [i for i in logit_fe.params.index if not i.startswith("yr")]
    coefs = logit_fe.params[main_idx]
    ci    = logit_fe.conf_int().loc[main_idx]
    y_pos = range(len(coefs))
    ax_coef.barh(y_pos, coefs.values, xerr=[
        coefs.values - ci[0].values,
        ci[1].values - coefs.values
    ], color=[PALETTE["primary"] if v>0 else PALETTE["error"] for v in coefs.values],
    alpha=0.75, capsize=4)
    ax_coef.set_yticks(list(y_pos))
    ax_coef.set_yticklabels([i.replace("_","\n") for i in main_idx], fontsize=8)
    ax_coef.axvline(0, color="black", lw=0.8, linestyle="--")
    ax_coef.set_title("Coeficientes (IC 95%)", fontweight="bold")
    ax_coef.spines[["top","right"]].set_visible(False)

    # ── D) Predicted PD by segment over time
    ax_ts = fig.add_subplot(gs[1, :])
    from sklearn.metrics import roc_auc_score as rauc
    # Build full prediction series for visualization
    import pandas as pd
    pred_all = logit_fe.predict()
    train_vis = pd.DataFrame({"fecha": pd.read_parquet("02_pd_estimation/data/panel_features.parquet")["fecha"].iloc[pred_all.index], "pd_hat": pred_all.values, "segmento": pd.read_parquet("02_pd_estimation/data/panel_features.parquet")["segmento"].iloc[pred_all.index]})
    colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["warning"], PALETTE["error"]]
    for seg, col in zip(train_vis["segmento"].unique(), colors):
        s = train_vis[train_vis["segmento"]==seg]
        ax_ts.plot(s["fecha"], s["pd_hat"], label=seg, color=col, lw=1.6)
    ax_ts.axvspan(pd.Timestamp("2020-03"), pd.Timestamp("2020-12"), alpha=0.12, color="red", label="COVID")
    ax_ts.set_title("PD Estimada por Segmento (Train)", fontweight="bold")
    ax_ts.set_ylabel("PD estimada")
    ax_ts.legend(fontsize=9, loc="upper left")
    ax_ts.spines[["top","right"]].set_visible(False)
    ax_ts.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x:.1%}"))
    ax_ts.grid(axis="y", linestyle="--", alpha=0.3)

    plt.savefig("02_pd_estimation/reports/05_panel_logit_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Figura guardada → reports/05_panel_logit_results.png")
    return (
        ax_coef, ax_dist, ax_roc, ax_ts, ci, coefs, col, colors,
        fig, gs, lbl, main_idx, preds, rauc, seg, train_vis, y, y_pos,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md("## 8 · Tests de diagnóstico")
    return


@app.cell
def __(X_train, acorr_breusch_godfrey, durbin_watson, logit_fe, y_train):
    resid = y_train.values - logit_fe.predict(X_train)
    dw = durbin_watson(resid)
    bg_stat, bg_p, _, _ = acorr_breusch_godfrey(logit_fe, nlags=4)

    print("=== TESTS DE DIAGNÓSTICO ===")
    print(f"Durbin-Watson: {dw:.4f}  (ideal ≈ 2.0, rango 1.5–2.5)")
    dw_flag = "✅" if 1.5 <= dw <= 2.5 else "⚠️"
    print(f"  {dw_flag} {'Sin autocorrelación serial' if 1.5 <= dw <= 2.5 else 'AUTOCORRELACIÓN DETECTADA'}")
    print(f"\nBreusch-Godfrey (4 lags): stat={bg_stat:.4f}, p={bg_p:.4f}")
    bg_flag = "✅" if bg_p > 0.05 else "⚠️"
    print(f"  {bg_flag} {'H0 no rechazada: sin autocorrelación' if bg_p > 0.05 else 'H0 rechazada: autocorrelación serial'}")
    return bg_flag, bg_p, bg_stat, dw, dw_flag, resid


@app.cell(hide_code=True)
def __(mo):
    mo.md("## 9 · Model Card exportado")
    return


@app.cell
def __(REPORTS_DIR, brier_score_loss, logit_fe, pred_oot, pred_train, roc_auc_score, y_oot, y_train):
    from scipy.stats import ks_2samp
    from datetime import datetime

    metrics_train = {
        "AUC":  roc_auc_score(y_train, pred_train),
        "Gini": 2*roc_auc_score(y_train, pred_train)-1,
        "KS":   ks_2samp(pred_train[y_train==1], pred_train[y_train==0]).statistic,
        "Brier": brier_score_loss(y_train, pred_train),
        "McFadden": logit_fe.prsquared,
    }
    metrics_oot = {
        "AUC":  roc_auc_score(y_oot, pred_oot),
        "Gini": 2*roc_auc_score(y_oot, pred_oot)-1,
        "KS":   ks_2samp(pred_oot[y_oot==1], pred_oot[y_oot==0]).statistic,
        "Brier": brier_score_loss(y_oot, pred_oot),
        "McFadden": logit_fe.prsquared,
    }

    card = f"""# Model Card — PD Estimation Panel Logit (SBS Ecuador)

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Model
- **Type:** Logit LSDV (Fixed Effects via dummy variables)
- **Target:** `default_bin` (morosidad ≥ p75 within-segment)
- **Panel:** 4 segmentos SBS × 120 períodos mensuales (2015–2024)
- **Split:** Train 2015–2021 / OOT 2022–2024

## Features
| Variable | Fuente | Hipótesis |
|----------|--------|-----------|
| tasa_activa_lag1 | BCE | Costo financiero rezagado |
| pib_gap_hp | BCE/HP filter | Ciclo económico |
| desempleo_lag1 | INEC | Mercado laboral |
| inflacion_lag1 | BCE | Erosión ingreso real |
| morosidad_lag1 | SBS | Persistencia AR(1) |
| dummy_covid | — | Shock COVID-19 |
| dummy_oilshock | — | Shock precio petróleo 2016 |
| seg_* | SBS | Fixed Effects de segmento |

## Performance

### Train (2015–2021)
| Métrica | Valor | Threshold Basel | Status |
|---------|-------|----------------|--------|
| AUC | {metrics_train['AUC']:.4f} | ≥ 0.70 | {'✅' if metrics_train['AUC']>=0.70 else '⚠️'} |
| Gini | {metrics_train['Gini']:.4f} | ≥ 0.40 | {'✅' if metrics_train['Gini']>=0.40 else '⚠️'} |
| KS | {metrics_train['KS']:.4f} | ≥ 0.30 | {'✅' if metrics_train['KS']>=0.30 else '⚠️'} |
| Brier | {metrics_train['Brier']:.4f} | ≤ 0.25 | {'✅' if metrics_train['Brier']<=0.25 else '⚠️'} |
| McFadden R² | {metrics_train['McFadden']:.4f} | ≥ 0.20 | {'✅' if metrics_train['McFadden']>=0.20 else '⚠️'} |

### OOT (2022–2024)
| Métrica | Valor | Threshold Basel | Status |
|---------|-------|----------------|--------|
| AUC | {metrics_oot['AUC']:.4f} | ≥ 0.70 | {'✅' if metrics_oot['AUC']>=0.70 else '⚠️'} |
| Gini | {metrics_oot['Gini']:.4f} | ≥ 0.40 | {'✅' if metrics_oot['Gini']>=0.40 else '⚠️'} |
| KS | {metrics_oot['KS']:.4f} | ≥ 0.30 | {'✅' if metrics_oot['KS']>=0.30 else '⚠️'} |
| Brier | {metrics_oot['Brier']:.4f} | ≤ 0.25 | {'✅' if metrics_oot['Brier']<=0.25 else '⚠️'} |

## Limitaciones
- Datos calibrados con estadísticas agregadas SBS; no son datos individuales de clientes.
- PD binaria basada en percentil 75 within-segment: no equivale a default regulatorio Basilea.
- Ausencia de variables de balance bancario individual (CAMEL).
- No incluye riesgo de concentración geográfica (datos SBS nacionales).
- Requiere re-estimación anual por degradación de la distribución (PSI).

## Referencias
- Jiménez & Saurina (2006). Credit Cycles, Credit Risk, and Prudential Regulation. IJCB.
- Pesaran (2015). Time Series and Panel Data Econometrics. Oxford.
- BIS WP No. 14 (2005). Studies on the Validation of Internal Rating Systems.
"""

    card_path = REPORTS_DIR / "model_card_02_pd_panel.md"
    card_path.write_text(card)
    print(f"✅ Model Card exportado → {card_path}")
    print("\n🏁 Case Study 02 completado.")
    return card, card_path, datetime, ks_2samp, metrics_oot, metrics_train


if __name__ == "__main__":
    app.run()

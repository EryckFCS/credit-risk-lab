#!/usr/bin/env bash
# =============================================================================
# Case Study 02 — PD & LGD Estimation
# Dataset A: SEPS Ecuador — Colocaciones mensuales cooperativas (público)
#   Fuente: https://estadisticas.seps.gob.ec/index.php/estadisticas-sfps/
# Dataset B: BCE — Encuesta Trimestral Oferta/Demanda de Crédito
#   Fuente: https://contenido.bce.fin.ec/documentos/informacioneconomica/MonetarioFinanciero/ix_OfertaDemandaCredito.html
# Dataset C: BCE — Boletín MOA 2024 (tasas activas / operaciones)
#   Fuente: https://contenido.bce.fin.ec/documentos/Estadisticas/SectorMonFin/TasasInteres/informe_moa_2024.pdf
# Dataset D: Banco Mundial — Crédito interno al sector privado (% PIB)
#   Fuente: https://datos.bancomundial.org/indicador/FS.AST.PRVT.GD.ZS
# =============================================================================
set -euo pipefail

OUT="02_pd_lgd_estimation/data"
mkdir -p "$OUT"

echo "[credit-risk-lab] Downloading Banco Mundial credit data (Ecuador)..."

# Banco Mundial: Crédito sector privado / PIB — Ecuador (1960-2024)
# API pública sin autenticación
BM_URL="https://api.worldbank.org/v2/country/EC/indicator/FS.AST.PRVT.GD.ZS?format=json&per_page=200&mrv=30"
curl -fsSL "$BM_URL" -o "$OUT/bm_credito_privado_ec.json"
echo "[ok] Banco Mundial → $OUT/bm_credito_privado_ec.json"

# Banco Mundial: NPL ratio (Non-performing loans % total) — Ecuador
NPL_URL="https://api.worldbank.org/v2/country/EC/indicator/FB.AST.NPER.ZS?format=json&per_page=200&mrv=30"
curl -fsSL "$NPL_URL" -o "$OUT/bm_npl_ec.json"
echo "[ok] Banco Mundial NPL → $OUT/bm_npl_ec.json"

# Banco Mundial: Tasa de interés activa — Ecuador
IR_URL="https://api.worldbank.org/v2/country/EC/indicator/FR.INR.LEND?format=json&per_page=200&mrv=30"
curl -fsSL "$IR_URL" -o "$OUT/bm_tasa_activa_ec.json"
echo "[ok] Banco Mundial tasa activa → $OUT/bm_tasa_activa_ec.json"

# Convertir JSON → parquet vía Python
uv run python - <<'PYEOF'
import json
import pandas as pd
from pathlib import Path

out = Path("02_pd_lgd_estimation/data")

def wb_json_to_df(fpath, value_name):
    with open(fpath) as f:
        data = json.load(f)
    records = data[1] if data[1] else []
    rows = [{"year": int(r["date"]), value_name: r["value"]} for r in records if r["value"] is not None]
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)

df_credit = wb_json_to_df(out / "bm_credito_privado_ec.json", "credit_pct_gdp")
df_npl    = wb_json_to_df(out / "bm_npl_ec.json", "npl_ratio")
df_ir     = wb_json_to_df(out / "bm_tasa_activa_ec.json", "lending_rate")

df = df_credit.merge(df_npl, on="year", how="outer").merge(df_ir, on="year", how="outer")
df.to_parquet(out / "ecuador_macro_credit.parquet", index=False)
print(f"[ok] Macro panel saved → {out}/ecuador_macro_credit.parquet")
print(df.tail(10).to_string(index=False))
PYEOF

echo ""
echo "[note] Para datos SEPS de colocaciones cooperativas, descargar manualmente:"
echo "  https://estadisticas.seps.gob.ec/index.php/estadisticas-sfps/"
echo "  Sección: 'Colocaciones mensuales' → CSV → guardar en $OUT/seps_colocaciones.csv"
echo ""
echo "[done] Datasets macro-financieros listos en $OUT/"

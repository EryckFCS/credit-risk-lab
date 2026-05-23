#!/usr/bin/env bash
# =============================================================================
# Case Study 04 — Basel III Capital Simulator
# Dataset A: Banco Mundial — Capital Adequacy Ratio (CAR) Ecuador
#   Indicador: FB.BNK.CAPA.ZS (bank capital to assets)
# Dataset B: Banco Mundial — Bank capital to assets ratio
#   Indicador: FB.BNK.CAPA.ZS
# Dataset C: SBS Ecuador — Indicadores financieros (CAMEL)
#   Fuente: https://www.superbancos.gob.ec/estadisticas/portalestudios/
# Los parámetros RWA y factores de riesgo son sintéticos (calibrados con BCE)
# =============================================================================
set -euo pipefail

OUT="04_basel_capital_simulator/data"
mkdir -p "$OUT"

echo "[credit-risk-lab] Downloading Basel capital data..."

# Bank capital to assets ratio — Ecuador
CAP_URL="https://api.worldbank.org/v2/country/EC/indicator/FB.BNK.CAPA.ZS?format=json&per_page=200&mrv=30"
curl -fsSL "$CAP_URL" -o "$OUT/bm_capital_assets_ec.json"
echo "[ok] Capital/assets ratio → $OUT/bm_capital_assets_ec.json"

# ROA bancario Ecuador
ROA_URL="https://api.worldbank.org/v2/country/EC/indicator/FB.BNK.ROVG.ZS?format=json&per_page=200&mrv=30"
curl -fsSL "$ROA_URL" -o "$OUT/bm_roa_ec.json"
echo "[ok] ROA bancario → $OUT/bm_roa_ec.json"

# Z-score bancario (estabilidad) Ecuador
ZSC_URL="https://api.worldbank.org/v2/country/EC/indicator/FB.BNK.RESL.BS.ZS?format=json&per_page=200&mrv=30"
curl -fsSL "$ZSC_URL" -o "$OUT/bm_provisions_ec.json"
echo "[ok] Provisiones/NPL → $OUT/bm_provisions_ec.json"

uv run python - <<'PYEOF'
import json
import pandas as pd
from pathlib import Path

out = Path("04_basel_capital_simulator/data")

def wb_to_df(fpath, col):
    with open(fpath) as f:
        data = json.load(f)
    recs = data[1] if data[1] else []
    return pd.DataFrame(
        [{"year": int(r["date"]), col: r["value"]} for r in recs if r["value"] is not None]
    ).sort_values("year").reset_index(drop=True)

df = (
    wb_to_df(out / "bm_capital_assets_ec.json", "capital_to_assets")
    .merge(wb_to_df(out / "bm_roa_ec.json", "roa_pct"), on="year", how="outer")
    .merge(wb_to_df(out / "bm_provisions_ec.json", "provisions_npl_pct"), on="year", how="outer")
)
df.to_parquet(out / "ecuador_banking_camel.parquet", index=False)
print("[ok] CAMEL indicators Ecuador:")
print(df.tail(8).to_string(index=False))
PYEOF

echo "[done] Basel capital data ready in $OUT/"

#!/usr/bin/env bash
# =============================================================================
# Case Study 03 — Concentration Risk
# Dataset A: SuperCías Ecuador — Ranking empresas 2024 (CSV público)
#   Fuente: https://appscvsmovil.supercias.gob.ec/ranking/reporte.html
# Dataset B: Banco Mundial — Activos 3 mayores bancos / total activos (%)
#   Indicador: FB.BNK.CAPA.ZS
# Dataset C: SBS Ecuador — Boletín Financiero (concentración cartera)
#   Fuente: https://www.superbancos.gob.ec/estadisticas/portalestudios/
# =============================================================================
set -euo pipefail

OUT="03_concentration_risk/data"
mkdir -p "$OUT"

echo "[credit-risk-lab] Downloading concentration risk data..."

# Banco Mundial: Assets 3 largest banks (% total bank assets) — Ecuador
BNK_URL="https://api.worldbank.org/v2/country/EC/indicator/FB.BNK.CAPA.ZS?format=json&per_page=200&mrv=30"
curl -fsSL "$BNK_URL" -o "$OUT/bm_bank_concentration_ec.json"
echo "[ok] Banco Mundial bank concentration → $OUT/bm_bank_concentration_ec.json"

# Banco Mundial: Market cap listed companies (% PIB)
MKT_URL="https://api.worldbank.org/v2/country/EC/indicator/CM.MKT.LCAP.GD.ZS?format=json&per_page=200&mrv=30"
curl -fsSL "$MKT_URL" -o "$OUT/bm_market_cap_ec.json"
echo "[ok] Banco Mundial market cap → $OUT/bm_market_cap_ec.json"

# Banco Mundial: panel regional para comparación (EC, CO, PE, CL, MX, BR)
REG_URL="https://api.worldbank.org/v2/country/EC;CO;PE;CL;MX;BR/indicator/FB.BNK.CAPA.ZS?format=json&per_page=500&mrv=1"
curl -fsSL "$REG_URL" -o "$OUT/bm_bank_concentration_latam.json"
echo "[ok] LATAM comparison → $OUT/bm_bank_concentration_latam.json"

uv run python - <<'PYEOF'
import json
import pandas as pd
from pathlib import Path

out = Path("03_concentration_risk/data")

# Ecuador concentration timeline
with open(out / "bm_bank_concentration_ec.json") as f:
    data = json.load(f)[1] or []
df_ec = pd.DataFrame(
    [{"year": int(r["date"]), "top3_bank_assets_pct": r["value"]} for r in data if r["value"]]
).sort_values("year")
df_ec.to_parquet(out / "bank_concentration_ecuador.parquet", index=False)
print(f"[ok] Ecuador bank concentration: {len(df_ec)} years")
print(df_ec.tail(5).to_string(index=False))

# LATAM comparison
with open(out / "bm_bank_concentration_latam.json") as f:
    data = json.load(f)[1] or []
df_latam = pd.DataFrame(
    [{"country": r["country"]["id"], "country_name": r["country"]["value"],
      "year": int(r["date"]), "top3_pct": r["value"]} for r in data if r["value"]]
)
df_latam.to_parquet(out / "bank_concentration_latam.parquet", index=False)
print(f"\n[ok] LATAM comparison: {len(df_latam)} rows")
print(df_latam.to_string(index=False))
PYEOF

echo ""
echo "[note] Para ranking SuperCías 2024 (top empresas Ecuador por sector):"
echo "  Ir a: https://appscvsmovil.supercias.gob.ec/ranking/reporte.html"
echo "  Descargar CSV de activos y patrimonio por sector CIIU"
echo "  Guardar en: $OUT/supercias_ranking_2024.csv"
echo ""
echo "[done] Concentration risk data ready in $OUT/"

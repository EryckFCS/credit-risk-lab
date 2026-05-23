#!/usr/bin/env bash
# =============================================================================
# Case Study 05 — EDA Sistema Financiero Ecuador
# Datasets 100% públicos y descargables:
#
# A) Banco Mundial — Panel de indicadores bancarios Ecuador y LATAM
#    Indicadores: crédito/PIB, NPL, CAR, ROA, ROE, spread, Z-score
# B) Datos Abiertos Ecuador — https://www.datosabiertos.gob.ec
#    (buscar: 'crédito sistema financiero')
# C) SEPS — Estadísticas cooperativas
#    https://estadisticas.seps.gob.ec/index.php/estadisticas-sfps/
# D) BCE — Tasas de interés activas/pasivas referenciales
#    https://contenido.bce.fin.ec/documentos/informacioneconomica/MonetarioFinanciero/ix_MonetariasFinancierasPrin.html
# =============================================================================
set -euo pipefail

OUT="05_eda_sbs_ecuador/data"
mkdir -p "$OUT"

echo "[credit-risk-lab] Downloading SFN Ecuador full panel from World Bank API..."

# Lista de indicadores bancarios Ecuador
declare -A INDICATORS=(
    ["credit_gdp"]="FS.AST.PRVT.GD.ZS"
    ["npl_ratio"]="FB.AST.NPER.ZS"
    ["capital_adequacy"]="FB.BNK.CAPA.ZS"
    ["roa"]="FB.BNK.ROVG.ZS"
    ["lending_rate"]="FR.INR.LEND"
    ["deposit_rate"]="FR.INR.DPST"
    ["bank_branches_per_100k"]="FB.CBK.BRCH.P5"
)

for NAME in "${!INDICATORS[@]}"; do
    IND="${INDICATORS[$NAME]}"
    URL="https://api.worldbank.org/v2/country/EC/indicator/${IND}?format=json&per_page=100&mrv=30"
    curl -fsSL "$URL" -o "$OUT/wb_${NAME}.json"
    echo "[ok] ${NAME} (${IND}) → $OUT/wb_${NAME}.json"
done

echo "[merge] Building comprehensive SFN panel..."

uv run python - <<'PYEOF'
import json
import pandas as pd
from pathlib import Path

out = Path("05_eda_sbs_ecuador/data")

indicators = {
    "credit_gdp": "Crédito sector privado (% PIB)",
    "npl_ratio": "Cartera vencida / total cartera (%)",
    "capital_adequacy": "Capital bancario / activos (%)",
    "roa": "ROA bancario (%)",
    "lending_rate": "Tasa activa referencial (%)",
    "deposit_rate": "Tasa pasiva referencial (%)",
    "bank_branches_per_100k": "Sucursales por 100,000 adultos",
}

dfs = []
for key, label in indicators.items():
    fpath = out / f"wb_{key}.json"
    if not fpath.exists():
        continue
    with open(fpath) as f:
        data = json.load(f)
    recs = data[1] if data[1] else []
    df = pd.DataFrame(
        [{"year": int(r["date"]), key: r["value"]} for r in recs if r["value"] is not None]
    ).sort_values("year").reset_index(drop=True)
    dfs.append(df)
    print(f"  {key}: {len(df)} observations")

if dfs:
    from functools import reduce
    panel = reduce(lambda a, b: a.merge(b, on="year", how="outer"), dfs)
    panel = panel.sort_values("year").reset_index(drop=True)
    panel.to_parquet(out / "ecuador_sfn_panel.parquet", index=False)
    panel.to_csv(out / "ecuador_sfn_panel.csv", index=False)
    print(f"\n[ok] SFN panel: {len(panel)} years × {len(panel.columns)} indicators")
    print(panel.tail(8).to_string(index=False))
PYEOF

echo ""
echo "[note] Para datos de la SEPS (cooperativas), descargar manualmente:"
echo "  https://estadisticas.seps.gob.ec/index.php/estadisticas-sfps/"
echo "  Secciones: Colocaciones, Captaciones, Indicadores SFPS"
echo "  Guardar en: $OUT/seps_*.csv"
echo ""
echo "[done] Ecuador SFN panel listo en $OUT/"

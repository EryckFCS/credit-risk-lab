#!/usr/bin/env bash
# =============================================================================
# Case Study 01 — Credit Scoring
# Dataset: Default of Credit Card Clients (UCI ML Repo, id=350)
# Source : https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
# License: CC BY 4.0  |  Yeh, I. (2009). DOI: 10.24432/C55S3H
# Download method: ucimlrepo Python package (official API, no scraping)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR"

echo "[credit-risk-lab] Downloading UCI dataset 350 via ucimlrepo..."

uv run python - <<'PYEOF'
from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo

out = Path("01_credit_scoring/data")
out.mkdir(parents=True, exist_ok=True)

dst = out / "credit_card_default.parquet"
if dst.exists():
    print(f"[skip] {dst} already exists — delete to re-download.")
else:
    print("[fetch] Connecting to UCI ML Repository...")
    ds = fetch_ucirepo(id=350)
    X = ds.data.features
    y = ds.data.targets
    df = pd.concat([X, y], axis=1)
    df.columns = [c.strip().upper() for c in df.columns]
    # Rename target column standardised name
    df.rename(columns={"DEFAULT.PAYMENT.NEXT.MONTH": "DEFAULT"}, inplace=True)
    df.to_parquet(dst, index=False)
    print(f"[ok] Saved {len(df):,} rows → {dst}")
    print(f"     Columns: {list(df.columns)}")
    print(f"     Default rate: {df['DEFAULT'].mean():.2%}")
PYEOF

echo "[done] Dataset ready at 01_credit_scoring/data/credit_card_default.parquet"

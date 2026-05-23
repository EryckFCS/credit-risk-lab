"""
Download and validate the UCI German Credit dataset.
Source: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)

Usage:
    python 01_credit_scoring/data/download.py

Output:
    01_credit_scoring/data/raw/german_credit.csv   <- labeled columns
    01_credit_scoring/data/raw/german_credit_numeric.csv  <- numeric version
"""

import pathlib
import urllib.request
import pandas as pd

RAW_DIR = pathlib.Path(__file__).parent / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── UCI URLs ──────────────────────────────────────────────────────────────────
URL_CATEGORICAL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "statlog/german/german.data"
)
URL_NUMERIC = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "statlog/german/german.data-numeric"
)

# ── Column names (Hofmann 1994 codebook) ─────────────────────────────────────
COLUMNS = [
    "checking_account",     # A1x  — status of existing checking account
    "duration_months",      # A2   — duration in months
    "credit_history",       # A3x  — credit history
    "purpose",              # A4x  — purpose
    "credit_amount",        # A5   — credit amount
    "savings_account",      # A6x  — savings account / bonds
    "employment_since",     # A7x  — present employment since
    "installment_rate",     # A8   — installment rate % of disposable income
    "personal_status_sex",  # A9x  — personal status and sex
    "other_debtors",        # A10x — other debtors / guarantors
    "residence_since",      # A11  — present residence since
    "property",             # A12x — property
    "age",                  # A13  — age in years
    "other_installments",   # A14x — other installment plans
    "housing",              # A15x — housing
    "existing_credits",     # A16  — number of existing credits at this bank
    "job",                  # A17x — job
    "liable_people",        # A18  — number of people liable
    "telephone",            # A19x — telephone
    "foreign_worker",       # A20x — foreign worker
    "target",               # 1=Good, 2=Bad  → remapped to 0=Good, 1=Bad
]

NUMERIC_COLUMNS = [f"x{i}" for i in range(1, 25)] + ["target"]


def download_categorical() -> pd.DataFrame:
    dest = RAW_DIR / "german_credit_raw.data"
    if not dest.exists():
        print(f"Downloading categorical dataset → {dest}")
        urllib.request.urlretrieve(URL_CATEGORICAL, dest)
    else:
        print(f"Categorical file already exists: {dest}")

    df = pd.read_csv(dest, sep=" ", header=None, names=COLUMNS)

    # Remap target: 1=Good→0, 2=Bad→1  (industry standard: 1=default)
    df["target"] = df["target"].map({1: 0, 2: 1})

    out = RAW_DIR / "german_credit.csv"
    df.to_csv(out, index=False)
    print(f"Saved → {out}  |  shape: {df.shape}  |  defaults: {df['target'].sum()}/1000")
    return df


def download_numeric() -> pd.DataFrame:
    dest = RAW_DIR / "german_credit_numeric.data"
    if not dest.exists():
        print(f"Downloading numeric dataset → {dest}")
        urllib.request.urlretrieve(URL_NUMERIC, dest)
    else:
        print(f"Numeric file already exists: {dest}")

    df = pd.read_csv(dest, sep=r"\s+", header=None, names=NUMERIC_COLUMNS)
    df["target"] = df["target"].map({1: 0, 2: 1})

    out = RAW_DIR / "german_credit_numeric.csv"
    df.to_csv(out, index=False)
    print(f"Saved → {out}  |  shape: {df.shape}")
    return df


def validate(df: pd.DataFrame) -> None:
    print("\n── Validation ──────────────────────────────────────────")
    assert df.shape == (1000, 21), f"Expected (1000, 21), got {df.shape}"
    assert df["target"].nunique() == 2, "Target must be binary"
    missing = df.isnull().sum().sum()
    default_rate = df["target"].mean()
    print(f"  Rows         : {len(df)}")
    print(f"  Columns      : {df.shape[1]}")
    print(f"  Missing      : {missing}")
    print(f"  Default rate : {default_rate:.1%}  (expected ~30%)")
    assert 0.25 <= default_rate <= 0.35, f"Default rate out of expected range: {default_rate:.1%}"
    print("  ✓ All checks passed")


if __name__ == "__main__":
    df_cat = download_categorical()
    download_numeric()
    validate(df_cat)
    print("\nDataset ready. Next step: run notebooks/01_eda.ipynb")

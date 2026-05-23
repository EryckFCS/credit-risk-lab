#!/usr/bin/env python3
"""
credit-risk-lab CLI
====================
Command-line interface to run case study pipelines, validate models,
and generate reports.

Usage:
    python cli.py --help
    python cli.py run --case 01
    python cli.py validate --case 01 --data path/to/data.csv
    python cli.py report --case 01
    python cli.py iv --data path/to/data.csv --target default
"""

import argparse
import sys
import subprocess
from pathlib import Path

# ── ANSI colors ────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
TEAL   = "\033[36m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

CASE_MAP = {
    "01": "01_credit_scoring",
    "02": "02_pd_lgd_estimation",
    "03": "03_concentration_risk",
    "04": "04_basel_capital_simulator",
    "05": "05_eda_sbs_ecuador",
}

BANNER = f"""{TEAL}{BOLD}
  ╔══════════════════════════════════════════╗
  ║       Credit Risk Lab  ·  CLI v1.0       ║
  ║   Erick Condoy  ·  github/EryckFCS      ║
  ╚══════════════════════════════════════════╝
{RESET}"""


def _print_banner() -> None:
    print(BANNER)


def _check_case(case: str) -> Path:
    """Resolve case directory and assert it exists."""
    if case not in CASE_MAP:
        print(f"{RED}[ERROR] Unknown case '{case}'. Valid: {list(CASE_MAP.keys())}{RESET}")
        sys.exit(1)
    case_dir = Path(CASE_MAP[case])
    if not case_dir.exists():
        print(f"{RED}[ERROR] Directory not found: {case_dir}{RESET}")
        sys.exit(1)
    return case_dir


# ── Subcommand: list ────────────────────────────────────────────────────────────
def cmd_list(_args: argparse.Namespace) -> None:
    _print_banner()
    print(f"{BOLD}Available Case Studies:{RESET}\n")
    statuses = {
        "01": ("Credit Scoring — Retail Portfolio",       "🔄 In Progress"),
        "02": ("PD & LGD Estimation — SME Lending",        "📋 Planned"),
        "03": ("Concentration Risk — Sector Analysis",      "📋 Planned"),
        "04": ("Basel III Capital Requirement Simulator",   "📋 Planned"),
        "05": ("EDA — SBS Ecuador Credit Portfolio",        "📋 Planned"),
    }
    for key, (name, status) in statuses.items():
        print(f"  {TEAL}{key}{RESET}  {name:<48} {status}")
    print()


# ── Subcommand: run ─────────────────────────────────────────────────────────────
def cmd_run(args: argparse.Namespace) -> None:
    _print_banner()
    case_dir = _check_case(args.case)
    nb_dir = case_dir / "notebooks"
    notebooks = sorted(nb_dir.glob("*.ipynb"))

    if not notebooks:
        print(f"{YELLOW}[WARN] No notebooks found in {nb_dir}{RESET}")
        sys.exit(0)

    print(f"{BOLD}Running case {args.case} — {CASE_MAP[args.case]}{RESET}")
    print(f"Found {len(notebooks)} notebook(s):\n")
    for nb in notebooks:
        print(f"  → {nb.name}")
    print()

    for nb in notebooks:
        if args.notebook and nb.name not in args.notebook:
            continue
        print(f"{TEAL}[RUN]{RESET} {nb.name} ...", end=" ", flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "nbconvert",
             "--to", "notebook",
             "--execute",
             "--inplace",
             "--ExecutePreprocessor.timeout=600",
             str(nb)],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"{GREEN}✓ DONE{RESET}")
        else:
            print(f"{RED}✗ FAILED{RESET}")
            print(result.stderr[-800:] if result.stderr else "No stderr")
            if not args.continue_on_error:
                sys.exit(1)

    print(f"\n{GREEN}All notebooks executed.{RESET}")


# ── Subcommand: report ──────────────────────────────────────────────────────────
def cmd_report(args: argparse.Namespace) -> None:
    _print_banner()
    case_dir = _check_case(args.case)
    nb_dir = case_dir / "notebooks"
    report_dir = case_dir / "reports"
    report_dir.mkdir(exist_ok=True)

    notebooks = sorted(nb_dir.glob("*.ipynb"))
    if not notebooks:
        print(f"{YELLOW}[WARN] No notebooks found.{RESET}")
        sys.exit(0)

    fmt = args.format  # html or pdf
    print(f"{BOLD}Generating {fmt.upper()} reports for case {args.case}...{RESET}\n")

    for nb in notebooks:
        out_name = nb.stem + f".{fmt}"
        out_path = report_dir / out_name
        print(f"{TEAL}[EXPORT]{RESET} {nb.name} → {out_path.name} ...", end=" ", flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "nbconvert",
             "--to", fmt,
             "--output-dir", str(report_dir),
             str(nb)],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"{GREEN}✓{RESET}")
        else:
            print(f"{RED}✗ FAILED{RESET}")
            print(result.stderr[-400:])

    print(f"\n{GREEN}Reports saved to: {report_dir}{RESET}")


# ── Subcommand: iv ──────────────────────────────────────────────────────────────
def cmd_iv(args: argparse.Namespace) -> None:
    """Quick IV screen on any CSV/Parquet file."""
    _print_banner()

    try:
        import pandas as pd
        from utils.preprocessing import iv_summary
    except ImportError as e:
        print(f"{RED}[ERROR] Missing dependency: {e}{RESET}")
        sys.exit(1)

    fpath = Path(args.data)
    if not fpath.exists():
        print(f"{RED}[ERROR] File not found: {fpath}{RESET}")
        sys.exit(1)

    print(f"Loading {fpath.name} ...", end=" ", flush=True)
    if fpath.suffix == ".parquet":
        df = pd.read_parquet(fpath)
    elif fpath.suffix == ".csv":
        df = pd.read_csv(fpath)
    else:
        print(f"{RED}Unsupported format. Use .csv or .parquet{RESET}")
        sys.exit(1)
    print(f"{GREEN}✓{RESET} ({df.shape[0]:,} rows)\n")

    result = iv_summary(df, target=args.target)
    print(f"{BOLD}Information Value Report — target: '{args.target}'{RESET}\n")
    print(f"{'Feature':<30} {'IV':>8}  {'Strength'}")
    print("-" * 52)
    for _, row in result.iterrows():
        color = GREEN if row.strength == 'Strong' else (TEAL if row.strength == 'Medium' else YELLOW)
        print(f"{row.feature:<30} {row.iv:>8.4f}  {color}{row.strength}{RESET}")
    print()


# ── Subcommand: validate ────────────────────────────────────────────────────────
def cmd_validate(args: argparse.Namespace) -> None:
    """Run validation metrics on a scored dataset (requires: score, target columns)."""
    _print_banner()

    try:
        import pandas as pd
        import numpy as np
        from utils.metrics import ks_statistic, gini_coefficient, psi, cap_ratio
    except ImportError as e:
        print(f"{RED}[ERROR] {e}{RESET}")
        sys.exit(1)

    fpath = Path(args.data)
    df = pd.read_csv(fpath) if fpath.suffix == ".csv" else pd.read_parquet(fpath)

    y_true = df[args.target].values
    y_score = df[args.score_col].values

    ks_val   = ks_statistic(y_true, y_score)
    gini_val = gini_coefficient(y_true, y_score)
    auc_val  = gini_val / 2 + 0.5
    cap_val  = cap_ratio(y_true, y_score)

    print(f"{BOLD}=== Validation Report ==={RESET}\n")
    rows = [
        ("KS Statistic",     ks_val,   0.30, ks_val  > 0.30),
        ("Gini Coefficient", gini_val, 0.40, gini_val > 0.40),
        ("AUC-ROC",          auc_val,  0.70, auc_val  > 0.70),
        ("CAP Ratio",        cap_val,  0.60, cap_val  > 0.60),
    ]
    print(f"{'Metric':<22} {'Value':>8}  {'Threshold':>12}  Status")
    print("-" * 54)
    for metric, val, thr, passed in rows:
        status = f"{GREEN}✓ PASS{RESET}" if passed else f"{RED}✗ FAIL{RESET}"
        print(f"{metric:<22} {val:>8.4f}  {'> '+str(thr):>12}  {status}")
    print()


# ── Main ────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python cli.py",
        description="Credit Risk Lab — Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py list
  python cli.py run --case 01
  python cli.py run --case 01 --notebook 01_eda.ipynb
  python cli.py report --case 01 --format html
  python cli.py iv --data 01_credit_scoring/data/processed_woe.parquet --target default
  python cli.py validate --data scored.csv --target default --score-col p_default
    """
    )
    sub = parser.add_subparsers(dest="command")

    # list
    sub.add_parser("list", help="List all case studies and their status")

    # run
    p_run = sub.add_parser("run", help="Execute notebooks for a case study")
    p_run.add_argument("--case", required=True, choices=CASE_MAP.keys())
    p_run.add_argument("--notebook", nargs="+", help="Run only specific notebook(s)")
    p_run.add_argument("--continue-on-error", action="store_true",
                       help="Continue if a notebook fails")

    # report
    p_rep = sub.add_parser("report", help="Export notebooks to HTML or PDF")
    p_rep.add_argument("--case", required=True, choices=CASE_MAP.keys())
    p_rep.add_argument("--format", choices=["html", "pdf"], default="html")

    # iv
    p_iv = sub.add_parser("iv", help="Quick Information Value screen on a dataset")
    p_iv.add_argument("--data", required=True, help="Path to CSV or Parquet file")
    p_iv.add_argument("--target", required=True, help="Target column name")

    # validate
    p_val = sub.add_parser("validate", help="Run KS/Gini/AUC/CAP validation metrics")
    p_val.add_argument("--data", required=True, help="Path to scored dataset")
    p_val.add_argument("--target", required=True, help="Target column name (0/1)")
    p_val.add_argument("--score-col", default="p_default",
                       help="Column with predicted probability (default: p_default)")

    args = parser.parse_args()

    handlers = {
        "list":     cmd_list,
        "run":      cmd_run,
        "report":   cmd_report,
        "iv":       cmd_iv,
        "validate": cmd_validate,
    }

    if args.command not in handlers:
        _print_banner()
        parser.print_help()
        sys.exit(0)

    handlers[args.command](args)


if __name__ == "__main__":
    main()

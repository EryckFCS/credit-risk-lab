#!/usr/bin/env python3
"""
credit-risk-lab CLI
====================
Command-line interface for the Credit Risk Lab research portfolio.

Usage:
    python cli.py --help
    python cli.py status
    python cli.py run --case 01 --notebook eda
    python cli.py report --case 01
    python cli.py metrics --help
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

# ── Paths (agnostic — no absolute OS paths) ───────────────────────────────────
ROOT = Path(__file__).parent

CASE_MAP = {
    "01": {"dir": "01_credit_scoring",         "name": "Credit Scoring"},
    "02": {"dir": "02_pd_lgd_estimation",       "name": "PD & LGD Estimation"},
    "03": {"dir": "03_concentration_risk",       "name": "Concentration Risk"},
    "04": {"dir": "04_basel_capital_simulator",  "name": "Basel III Simulator"},
    "05": {"dir": "05_eda_sbs_ecuador",          "name": "EDA SBS Ecuador"},
}

NOTEBOOK_MAP = {
    "eda":   "01_eda.ipynb",
    "fe":    "02_feature_engineering.ipynb",
    "model": "03_model_and_validation.ipynb",
}

# ANSI colors
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


# ── Helpers ────────────────────────────────────────────────────────────────────
def _header():
    print(f"""
{BOLD}{CYAN}╔══════════════════════════════════════════╗
║       Credit Risk Lab  ·  CLI v1.0       ║
║   Erick Condoy · Economist (UNL)         ║
╚══════════════════════════════════════════╝{RESET}
""")


def _case_exists(case_id: str) -> Path:
    if case_id not in CASE_MAP:
        print(f"{RED}[ERROR]{RESET} Case '{case_id}' not found. Valid cases: {list(CASE_MAP.keys())}")
        sys.exit(1)
    case_dir = ROOT / CASE_MAP[case_id]["dir"]
    if not case_dir.exists():
        print(f"{RED}[ERROR]{RESET} Directory not found: {case_dir}")
        sys.exit(1)
    return case_dir


def _check_command(cmd: str) -> bool:
    """Check if a shell command is available."""
    return subprocess.run(["which", cmd], capture_output=True).returncode == 0


# ── Subcommands ────────────────────────────────────────────────────────────────
def cmd_status(args):
    """Show the status of all case studies."""
    _header()
    print(f"{BOLD}Case Studies Status:{RESET}\n")
    print(f"  {'#':<4} {'Name':<35} {'Notebooks':<12} {'Reports'}")
    print(f"  {'─'*4} {'─'*35} {'─'*12} {'─'*15}")

    STATUS_ICONS = {True: f"{GREEN}✓{RESET}", False: f"{YELLOW}○{RESET}"}

    for cid, meta in CASE_MAP.items():
        case_dir = ROOT / meta["dir"]
        nb_dir = case_dir / "notebooks"
        rp_dir = case_dir / "reports"

        notebooks = list(nb_dir.glob("*.ipynb")) if nb_dir.exists() else []
        reports   = list(rp_dir.glob("*.html")) + list(rp_dir.glob("*.pdf")) if rp_dir.exists() else []

        nb_icon = STATUS_ICONS[len(notebooks) > 0]
        rp_icon = STATUS_ICONS[len(reports) > 0]

        print(f"  {cid:<4} {meta['name']:<35} {nb_icon} {len(notebooks)} notebooks   {rp_icon} {len(reports)} reports")

    print()


def cmd_run(args):
    """Execute a notebook for a given case study."""
    _header()

    if not _check_command("jupyter"):
        print(f"{RED}[ERROR]{RESET} jupyter not found. Install with: pip install jupyterlab")
        sys.exit(1)

    case_dir = _case_exists(args.case)

    if args.notebook not in NOTEBOOK_MAP:
        print(f"{RED}[ERROR]{RESET} Notebook '{args.notebook}' not found. Valid: {list(NOTEBOOK_MAP.keys())}")
        sys.exit(1)

    nb_name = NOTEBOOK_MAP[args.notebook]
    nb_path = case_dir / "notebooks" / nb_name

    if not nb_path.exists():
        print(f"{YELLOW}[WARN]{RESET}  Notebook not found: {nb_path}")
        print(f"        Create the notebook first or check the status with: python cli.py status")
        sys.exit(1)

    print(f"{CYAN}[RUN]{RESET}  Executing: {nb_path.relative_to(ROOT)}")
    print(f"       Case: {CASE_MAP[args.case]['name']}\n")

    cmd = [
        "jupyter", "nbconvert", "--to", "notebook",
        "--execute", "--inplace",
        "--ExecutePreprocessor.timeout=600",
        str(nb_path),
    ]
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"\n{GREEN}[OK]{RESET}   Notebook executed successfully.")
        print(f"       Run 'python cli.py report --case {args.case}' to export HTML.")
    else:
        print(f"\n{RED}[FAIL]{RESET} Execution failed. Check the notebook for errors.")
        sys.exit(1)


def cmd_report(args):
    """Export a case study notebook to HTML report."""
    _header()

    if not _check_command("jupyter"):
        print(f"{RED}[ERROR]{RESET} jupyter not found.")
        sys.exit(1)

    case_dir = _case_exists(args.case)
    nb_dir   = case_dir / "notebooks"
    rp_dir   = case_dir / "reports"
    rp_dir.mkdir(exist_ok=True)

    notebooks = sorted(nb_dir.glob("*.ipynb"))
    if not notebooks:
        print(f"{YELLOW}[WARN]{RESET}  No notebooks found in {nb_dir}")
        sys.exit(1)

    for nb_path in notebooks:
        out_name = nb_path.stem + ".html"
        out_path = rp_dir / out_name
        print(f"{CYAN}[EXPORT]{RESET} {nb_path.name} → reports/{out_name}")

        cmd = [
            "jupyter", "nbconvert", "--to", "html",
            "--no-input",
            f"--output={str(out_path)}",
            str(nb_path),
        ]
        subprocess.run(cmd, check=True)

    print(f"\n{GREEN}[OK]{RESET}   Reports saved to: {rp_dir.relative_to(ROOT)}")
    print(f"       Share the HTML files as LinkedIn articles or GitHub Pages.")


def cmd_metrics(args):
    """Run validation metrics on a saved predictions CSV."""
    _header()

    preds_path = Path(args.predictions)
    if not preds_path.exists():
        print(f"{RED}[ERROR]{RESET} File not found: {preds_path}")
        sys.exit(1)

    try:
        import pandas as pd
        import numpy as np
        sys.path.insert(0, str(ROOT))
        from utils.metrics import ks_statistic, gini_coefficient, psi, cap_ratio
        from sklearn.metrics import roc_auc_score
    except ImportError as e:
        print(f"{RED}[ERROR]{RESET} Missing dependency: {e}")
        sys.exit(1)

    df = pd.read_csv(preds_path)

    required = {args.target_col, args.score_col}
    missing  = required - set(df.columns)
    if missing:
        print(f"{RED}[ERROR]{RESET} Columns not found: {missing}. Available: {df.columns.tolist()}")
        sys.exit(1)

    y_true  = df[args.target_col].values
    y_score = df[args.score_col].values

    ks   = ks_statistic(y_true, y_score)
    gini = gini_coefficient(y_true, y_score)
    auc  = roc_auc_score(y_true, y_score)
    cap  = cap_ratio(y_true, y_score)

    def _badge(val, threshold):
        return f"{GREEN}✓ PASS{RESET}" if val >= threshold else f"{RED}✗ REVIEW{RESET}"

    print(f"{BOLD}Validation Metrics{RESET}\n")
    print(f"  {'Metric':<20} {'Value':>8}   {'Threshold':>10}   Status")
    print(f"  {'─'*20} {'─'*8}   {'─'*10}   {'─'*12}")
    print(f"  {'KS Statistic':<20} {ks:>8.4f}   {'>0.30':>10}   {_badge(ks, 0.30)}")
    print(f"  {'Gini Coefficient':<20} {gini:>8.4f}   {'>0.40':>10}   {_badge(gini, 0.40)}")
    print(f"  {'AUC-ROC':<20} {auc:>8.4f}   {'>0.70':>10}   {_badge(auc, 0.70)}")
    print(f"  {'CAP Ratio':<20} {cap:>8.4f}   {'>0.60':>10}   {_badge(cap, 0.60)}")
    print()


# ── Argument Parser ────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="Credit Risk Lab — Research Portfolio CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py status
  python cli.py run --case 01 --notebook eda
  python cli.py report --case 01
  python cli.py metrics --predictions preds.csv --target-col default --score-col p_default
""",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # status
    sub.add_parser("status", help="Show case study completion status")

    # run
    p_run = sub.add_parser("run", help="Execute a notebook")
    p_run.add_argument("--case",     required=True, choices=CASE_MAP.keys(), help="Case study ID (01–05)")
    p_run.add_argument("--notebook", required=True, choices=NOTEBOOK_MAP.keys(),
                       help="Notebook stage: eda | fe | model")

    # report
    p_rep = sub.add_parser("report", help="Export notebooks to HTML reports")
    p_rep.add_argument("--case", required=True, choices=CASE_MAP.keys(), help="Case study ID (01–05)")

    # metrics
    p_met = sub.add_parser("metrics", help="Compute validation metrics from predictions CSV")
    p_met.add_argument("--predictions",  required=True, help="Path to CSV with predictions")
    p_met.add_argument("--target-col",   default="default", help="Column name for true labels (default: 'default')")
    p_met.add_argument("--score-col",    default="p_default", help="Column name for predicted score (default: 'p_default')")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "status":  cmd_status,
        "run":     cmd_run,
        "report":  cmd_report,
        "metrics": cmd_metrics,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# Download UCI Credit Default dataset
# Run from the 01_credit_scoring/data/ directory

set -e

DATASET_URL="https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
OUTPUT_DIR="$(dirname "$0")"

echo "[INFO] Downloading UCI Credit Default dataset..."
curl -L "$DATASET_URL" -o "${OUTPUT_DIR}/uci_credit.zip"

echo "[INFO] Extracting..."
unzip -o "${OUTPUT_DIR}/uci_credit.zip" -d "${OUTPUT_DIR}/"

echo "[INFO] Done. Files available in ${OUTPUT_DIR}/"

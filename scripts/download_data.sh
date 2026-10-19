#!/usr/bin/env bash
# Download the Lending Club accepted-loans file (Kaggle: wordsforthewise/lending-club).
# The Kaggle bundle is ~1.3 GB and anonymous download works. Only the accepted-loans
# file is extracted; the bundle is deleted afterwards. Re-running is a no-op once the
# file is present and its checksum matches.
set -euo pipefail

RAW_DIR="data/raw"
FILE="accepted_2007_to_2018Q4.csv.gz"
SHA256="55c16f75120f897683f02e7aabcf080d0e4a20c4832feb1d592cfa941bd62a2d"
URL="https://www.kaggle.com/api/v1/datasets/download/wordsforthewise/lending-club"

mkdir -p "$RAW_DIR"

checksum_ok() {
  [ -f "$RAW_DIR/$FILE" ] && echo "$SHA256  $RAW_DIR/$FILE" | shasum -a 256 -c --status
}

if checksum_ok; then
  echo "data: $FILE present, checksum OK"
  exit 0
fi

if [ ! -f "$RAW_DIR/lending-club.zip" ]; then
  echo "data: downloading Kaggle bundle (~1.3 GB)"
  curl -sS -L --fail -o "$RAW_DIR/lending-club.zip" "$URL"
fi
unzip -o -j "$RAW_DIR/lending-club.zip" "$FILE" -d "$RAW_DIR"

if ! checksum_ok; then
  echo "data: checksum mismatch for $FILE" >&2
  exit 1
fi
rm -f "$RAW_DIR/lending-club.zip"
echo "data: $FILE extracted, checksum OK"

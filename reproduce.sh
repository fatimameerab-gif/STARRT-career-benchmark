#!/usr/bin/env bash
# Rebuild the cleaned datasets and the benchmark folds from the raw downloads (see raw/README.md).
# Takes about 5-10 minutes on a laptop, mostly for Djinni.
set -euo pipefail
cd "$(dirname "$0")"
python3 scripts/build_datasets.py .     # raw/ -> processed/
python3 scripts/make_folds.py .         # processed/ -> kaggle/data/
echo "Done. Compare with the released files:  git diff --stat kaggle/data"

#!/usr/bin/env bash
# Run with bash scripts/setup-linux.sh from an unpacked archive or checkout.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUTF8=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if ! command -v uv >/dev/null 2>&1; then
  echo 'uv is required. Install its standalone binary from https://docs.astral.sh/uv/.' >&2
  exit 2
fi
uv python install 3.11
if [[ ! -d .venv ]]; then uv venv --python 3.11 .venv; fi
.venv/bin/python -c 'import sys; assert (3,10) <= sys.version_info[:2] < (3,14), sys.version'
if [[ -f uv.lock ]]; then
  # Exact versions from the committed lock (v0.4 runs); refuses a lock that no longer matches pyproject.
  uv sync --locked --extra data --extra test --extra report
else
  uv pip install --python .venv/bin/python -e '.[data,test,report]'
fi
.venv/bin/python -m costrank.cli doctor
.venv/bin/python -m pytest -q

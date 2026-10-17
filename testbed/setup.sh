#!/usr/bin/env bash
# One-button boot for the ML mastery testbed (macOS / Linux / Git Bash).
#
# Resolves an interpreter, installs what is needed, records what this
# machine can verify, runs the contract tests, and executes the baseline
# experiment.
#
# Two modes:
#   default            creates .venv and installs the PINNED versions in
#                      requirements.txt -- reproducible and isolated
#   --conda-env NAME   uses an existing environment and installs only what
#   --python PATH      is ABSENT, leaving installed versions alone. Pins
#                      are NOT enforced there, so results are reproducible
#                      only on that machine -- note it in your report.
#
# Usage:
#   ./setup.sh
#   ./setup.sh --conda-env qwen35
#   ./setup.sh --python /usr/bin/python3.12
#   ./setup.sh --minimal      core packages only; deep-learning arms stay
#                             UNVERIFIED, by design
#   ./setup.sh --skip-tests

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MINIMAL=0
SKIP_TESTS=0
CONDA_ENV=""
EXISTING_PYTHON=""
while [ $# -gt 0 ]; do
  case "$1" in
    --minimal) MINIMAL=1 ;;
    --skip-tests) SKIP_TESTS=1 ;;
    --conda-env) CONDA_ENV="${2:?--conda-env needs a name}"; shift ;;
    --python) EXISTING_PYTHON="${2:?--python needs a path}"; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

echo
echo "=== ML mastery testbed: boot ==="

# 1. Interpreter ------------------------------------------------------
USE_EXISTING=0
PYTHON=""
if [ -n "$CONDA_ENV" ]; then
  CONDA_BASE="$(conda info --base 2>/dev/null)" || {
    echo "conda not found; pass --python <path> instead." >&2
    exit 1
  }
  ENV_DIR="$CONDA_BASE/envs/$CONDA_ENV"
  if [ -x "$ENV_DIR/bin/python" ]; then
    PYTHON="$ENV_DIR/bin/python"
  elif [ -x "$ENV_DIR/python.exe" ]; then
    PYTHON="$ENV_DIR/python.exe"
  else
    echo "conda environment '$CONDA_ENV' not found" >&2
    exit 1
  fi
  USE_EXISTING=1
elif [ -n "$EXISTING_PYTHON" ]; then
  PYTHON="$EXISTING_PYTHON"
  USE_EXISTING=1
else
  PYTHON="${PYTHON:-python3}"
  command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python
fi

VERSION="$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
echo "interpreter : $PYTHON (Python $VERSION)"
"$PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' || {
  echo "Python 3.11+ required (configs are read with tomllib)." >&2
  exit 1
}

# 2. Environment and dependencies -------------------------------------
if [ "$USE_EXISTING" -eq 1 ]; then
  VENV_PY="$PYTHON"
  echo "mode        : existing environment (pins NOT enforced)"
  MISSING="$("$VENV_PY" -m mlkit.verify --missing)"
  if [ -n "$MISSING" ]; then
    echo "installing  : $MISSING"
    # shellcheck disable=SC2086 -- word splitting is the point here
    "$VENV_PY" -m pip install --upgrade-strategy only-if-needed $MISSING
  else
    echo "installing  : nothing -- every package is already present"
  fi
else
  if [ ! -d .venv ]; then
    echo "creating    : .venv"
    "$PYTHON" -m venv .venv
  else
    echo "venv        : reusing existing .venv"
  fi
  VENV_PY=".venv/bin/python"
  [ -x "$VENV_PY" ] || VENV_PY=".venv/Scripts/python.exe"
  echo "installing  : pinned dependencies (this can take a few minutes)"
  "$VENV_PY" -m pip install --upgrade pip --quiet
  if [ "$MINIMAL" -eq 1 ]; then
    "$VENV_PY" -m pip install --quiet \
      numpy==2.1.3 scipy==1.14.1 pandas==2.2.3 matplotlib==3.9.2 \
      pytest==8.3.3 ruff==0.7.4
  else
    "$VENV_PY" -m pip install --quiet -r requirements.txt || {
      echo "dependency install failed; re-run with --minimal to continue" >&2
      echo "with the NumPy arms only (those arms still verify)." >&2
    }
  fi
fi

# 3. Environment record -----------------------------------------------
echo
"$VENV_PY" -m mlkit.verify --env

# 4. Contract tests ----------------------------------------------------
if [ "$SKIP_TESTS" -eq 0 ]; then
  echo
  echo "running     : contract tests (metrics, splits, leakage)"
  if ! "$VENV_PY" -m pytest tests -q; then
    echo "CONTRACT TESTS FAILED -- stop here and fix before Stage 0." >&2
    exit 1
  fi
fi

# 5. Baseline experiment -----------------------------------------------
echo
echo "running     : EXP-01 baseline"
set +e
"$VENV_PY" -m experiments.exp01_baseline --config configs/exp01.toml
EXP_CODE=$?
set -e

echo
echo "=== boot complete ==="
if [ "$USE_EXISTING" -eq 1 ]; then
  echo "run experiments with:"
  echo "  $VENV_PY -m experiments.exp01_baseline --config configs/exp01.toml"
else
  echo "activate the environment with: source .venv/bin/activate"
fi
if [ "$EXP_CODE" -eq 0 ]; then
  echo "EXP-01 is VERIFIED. Stage 0 is open."
else
  echo "EXP-01 is not fully verified. Read the gate table and the arm notes"
  echo "above, then log the finding in VERIFICATION.md."
fi

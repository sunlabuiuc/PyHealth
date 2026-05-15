#!/usr/bin/env bash
# run_tests.sh — run the in-scope pytest suite with the right interpreter
# and PYTHONPATH. Excludes three pre-existing tests at the tests/ root
# (test_text_embedding.py, test_tuple_time_text_processor.py,
# test_tuple_time_text_tokenizer.py) that import pyhealth.processors,
# which is not carried on this PR branch.
#
# Usage:
#   scripts/run_tests.sh                    # all four in-scope dirs
#   scripts/run_tests.sh tasks              # tests/tasks only
#   scripts/run_tests.sh datasets
#   scripts/run_tests.sh models
#   scripts/run_tests.sh scripts
#   scripts/run_tests.sh tests/models/test_retina_unet.py -k forward  # custom
#
# Extra pytest args (-v, -k, --lf, etc.) after the first argument are
# forwarded to pytest.
#
# Env overrides:
#   PYTHON      — python interpreter (default: ptorch2 env)
#   PYTEST_ARGS — additional default pytest args (e.g. "-q")

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="python"
DEFAULT_ARGS="${PYTEST_ARGS:--q}"

SUBSET="${1:-all}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "$SUBSET" in
  all)
    TARGETS=(tests/tasks tests/datasets tests/models tests/scripts)
    ;;
  tasks|datasets|models|scripts)
    TARGETS=("tests/$SUBSET")
    ;;
  *)
    # Treat as a custom path or pytest node id, forwarded verbatim.
    TARGETS=("$SUBSET")
    ;;
esac

echo "[tests] python=$PYTHON"
echo "[tests] targets=${TARGETS[*]}"
echo "[tests] extra args=$* default=$DEFAULT_ARGS"
echo

PYTHONPATH="${PYTHONPATH:-.}" "$PYTHON" -m pytest $DEFAULT_ARGS "${TARGETS[@]}" "$@"

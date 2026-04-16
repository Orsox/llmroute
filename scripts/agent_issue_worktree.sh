#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <claim|list|commit|complete> [args...]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PROJECT_ROOT}/.venv/Scripts/python.exe"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "python executable not found in ${PROJECT_ROOT}/.venv" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m llmrouter.issue_agent "$@"

#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  echo 'Usage: bash scripts/install.sh [transformers|vllm|client] [venv-directory]'
  echo 'Set PYTHON to a Python 3.11-3.13 executable; Python 3.12 is recommended.'
  exit 0
fi
backend=${1:-transformers}
case "$backend" in
  transformers|vllm|client) ;;
  *) echo "Unknown backend: $backend" >&2; exit 2 ;;
esac
if [[ $# -gt 2 ]]; then echo 'Too many arguments; see --help' >&2; exit 2; fi
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
env_dir=${2:-"$repo_dir/.venv-$backend"}
python_bin=${PYTHON:-python3}
"$python_bin" -c 'import sys; assert (3, 11) <= sys.version_info[:2] < (3, 14), "Use Python 3.11-3.13"'
"$python_bin" -m venv "$env_dir"
"$env_dir/bin/python" -m pip install --upgrade pip
package="$repo_dir[$backend]"
if [[ $backend == client ]]; then package=$repo_dir; fi
"$env_dir/bin/python" -m pip install -e "$package"
"$env_dir/bin/python" -m pip check
printf 'Activate with: source %q\n' "$env_dir/bin/activate"

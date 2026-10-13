#!/usr/bin/env bash
# Local compile check with Tectonic (XeTeX-based; fetches packages on demand).
# Overleaf (pdfLaTeX) remains authoritative for the final page count.
#   tools/build.sh            -> build/main.pdf
#   tools/build.sh --final    -> also fails if \draftfalse would fail (tbd/unverified/placeholders)
set -euo pipefail
cd "$(dirname "$0")/.."
command -v tectonic >/dev/null || { echo "tectonic not found: brew install tectonic" >&2; exit 1; }
mkdir -p build
if [[ "${1:-}" == "--final" ]]; then
  # Build a copy with \draftfalse switched on, without touching main.tex.
  sed 's/^% \\draftfalse/\\draftfalse/' main.tex > build/_final_main.tex
  cp build/_final_main.tex ./_final_main.tex
  trap 'rm -f ./_final_main.tex' EXIT
  tectonic -X compile _final_main.tex --outdir build --keep-logs
  mv build/_final_main.pdf build/main_final.pdf
  echo "OK: build/main_final.pdf"
else
  tectonic -X compile main.tex --outdir build --keep-logs
  echo "OK: build/main.pdf"
fi
grep -E "LaTeX Warning: (Reference|Citation).*undefined|There were undefined" build/*.log || true

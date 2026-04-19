#!/usr/bin/env bash
#
# export_showcase_figures.sh
#
# Re-export the four top-level README visuals into docs/figures/.
#
# This script performs NO experiment and NO analysis. It is a pure copy step
# from already-produced figure sources into the GitHub-visible docs/figures/
# directory used by the root README. Run it after figure sources are updated
# to keep README visuals in sync.
#
# Sources:
#   docs/plots/                                       -- DiT-track figures (tracked)
#   deliverables/d1_emu3_understanding_freeze/figures -- D-1 anchor figures (local)
#
# Destination:
#   docs/figures/   (tracked; referenced from README.md)

set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "${here}/../.." && pwd)"

src_dit="${repo}/docs/plots"
src_d1="${repo}/deliverables/d1_emu3_understanding_freeze/figures"
dst="${repo}/docs/figures"

mkdir -p "${dst}"

copy_if_present () {
  local src="$1" dst="$2"
  if [ ! -f "${src}" ]; then
    echo "WARN: source missing, skipping: ${src}" >&2
    return 0
  fi
  cp -f "${src}" "${dst}"
  echo "OK: ${src} -> ${dst}"
}

copy_if_present "${src_dit}/temporal_plateau_comparison.png" "${dst}/01_dit_temporal_plateau.png"
copy_if_present "${src_dit}/cross_model_speedup.png"         "${dst}/02_dit_cross_model_speedup.png"
copy_if_present "${src_d1}/fig1_prefill_vs_tokens.png"       "${dst}/03_emu3_prefill_scaling.png"
copy_if_present "${src_d1}/fig6_prefill_gen_ratio.png"       "${dst}/04_emu3_prefill_gen_ratio.png"

# Janus-Pro SM-2 matched-k inversion figure. This one is rendered fresh from
# the accepted V4 auxiliary pooled means (no new measurement); the renderer
# lives alongside this script so rerun is self-contained. Requires matplotlib
# + numpy in the invoking Python.
python_bin="${PYTHON_BIN:-python3}"
if ! command -v "${python_bin}" >/dev/null 2>&1; then
  echo "WARN: python_bin '${python_bin}' not found; skipping SM-2 figure" >&2
else
  if "${python_bin}" -c "import matplotlib, numpy" >/dev/null 2>&1; then
    "${python_bin}" "${here}/sm2_matched_k_inversion_figure.py"
  else
    echo "WARN: matplotlib/numpy not importable under ${python_bin}; skipping SM-2 figure" >&2
  fi
fi

echo
echo "docs/figures/ is now:"
ls -la "${dst}"

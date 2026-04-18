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

echo
echo "docs/figures/ is now:"
ls -la "${dst}"

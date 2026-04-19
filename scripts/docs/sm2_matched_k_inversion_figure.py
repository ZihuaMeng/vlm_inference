"""Render the SM-2 matched-k cross-family dominance-inversion figure.

This is a docs / asset-export helper. It performs NO experiment, NO analysis,
and NO measurement. It consumes already-accepted SM-2 V4 auxiliary pooled means
(from `deliverables/sm2_janus_pro_round2/v4_auxiliary_matched_k.md`) and emits
a single presentation-safe figure at `docs/figures/05_sm2_matched_k_inversion.png`.

The numeric rows are pinned to the V4 auxiliary table; re-running this script
produces the same figure as long as that accepted table does not change.

Usage (from repo root, with any Python env that has matplotlib + numpy):
    python scripts/docs/sm2_matched_k_inversion_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Source: deliverables/sm2_janus_pro_round2/v4_auxiliary_matched_k.md (APPENDIX).
# Each row is a per-family pooled mean at matched k; `inversion` = yes on all 4.
K_LEVELS = [0, 1, 2, 3]
JANUS_RATIO = [0.0715, 0.1370, 0.1396, 0.2069]
EMU3_RATIO = [5.9411, 9.5846, 11.6988, 14.5251]
JANUS_PTOK = [635, 1233, 1845, 2443]
EMU3_PTOK = [4479, 8691, 13595, 18492]


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    out_dir = repo / "docs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "05_sm2_matched_k_inversion.png"

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=150)
    xs = np.array(K_LEVELS, dtype=float)

    ax.plot(
        xs,
        EMU3_RATIO,
        marker="o",
        linewidth=2.2,
        color="#1f77b4",
        label="Emu3-Chat (A-2, VQ ≈ 4000 tok/image) — prefill-dominant",
    )
    ax.plot(
        xs,
        JANUS_RATIO,
        marker="s",
        linewidth=2.2,
        color="#d62728",
        label="Janus-Pro (SM-2, SigLIP-L ≈ 576 tok/image) — generation-dominant",
    )
    ax.axhline(
        1.0,
        color="grey",
        linestyle="--",
        linewidth=1.0,
        label="prefill/gen = 1 (crossover line)",
    )

    ax.set_yscale("log")
    ax.set_xticks(K_LEVELS)
    ax.set_xlabel("ICL shots (k)")
    ax.set_ylabel("prefill_s / generation_s  (log scale)")
    ax.set_title(
        "SM-2 matched-k dominance inversion (Janus-Pro vs. Emu3-Chat)",
        pad=14,
    )
    fig.text(
        0.5,
        0.955,
        "bounded: GQA-testdev single-image  •  k ∈ {0,1,2,3}  •  bf16 B200  •  greedy decoding",
        ha="center",
        va="top",
        fontsize=8.5,
        color="#555555",
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="center right", fontsize=9)

    for k, jr, er, jp, ep in zip(K_LEVELS, JANUS_RATIO, EMU3_RATIO, JANUS_PTOK, EMU3_PTOK):
        ax.annotate(
            f"{er:.2f}\n(ptok {ep})",
            xy=(k, er),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#1f77b4",
        )
        ax.annotate(
            f"{jr:.3f}\n(ptok {jp})",
            xy=(k, jr),
            xytext=(0, -22),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#d62728",
        )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    fig.savefig(out_path, bbox_inches="tight")
    print(f"OK: wrote {out_path}")


if __name__ == "__main__":
    main()

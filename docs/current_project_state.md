# Current Project State

**Snapshot date**: 2026-04-18. This file is the GitHub-visible entry point for the project's current branch state. Detailed planning, raw evidence, and recon notes live in the local working tree (not part of git history by convention) — this document summarizes them conservatively.

## Tracks and status

### DiT Track — Temporal step-caching (Qwen-Image-Edit, FLUX Kontext-dev)

Completed. M0–M9 (Qwen) and F0–F4 (FLUX) milestones done; combined temporal + CFG-skip speedups and per-model trade-offs are captured in [`results.md`](results.md) and summarized in the root [`README.md`](../README.md).

### AR Track — AR-1 through AR-5 + AR-UND (Lumina-mGPT-2.0-Omni)

Completed. Characterization (AR-1), KV compression (AR-2), sparse prefill (AR-3), prefix caching (AR-4), RoPE-pathology decomposition (AR-5), and the understanding-task generalization probe (AR-UND) are all resolved on Lumina-mGPT-2.0. See the root [`README.md`](../README.md) for the per-experiment summaries.

### AR-6 / SAICL — Structured prefill acceleration prototype

**Frozen** in the current pass. AR-6 combines SAICL-style structured demo attention with reversed-pyramid layer-adaptive retention and head-selective retention. Phase 3 rerun on B200 remains gated behind the Emu3 generation-capability decision and is not re-opened in this pass.

### Phase A / D-1 — Emu3-Chat understanding regime

**Frozen** as the current clean control baseline. D-1 anchor validation returned `OVERALL_PASS` across the primary per-query gates and the per-cell secondary gates; the claim ledger is closed. The D-1 deliverable tree (report, anchor raw/summary, figures) lives under the local `deliverables/d1_emu3_understanding_freeze/` path.

### D-2a — Emu3 generation-capability / variant audit + feasibility envelope

**Entered PI branch re-judgment after a window-clean negative checkpoint.** No active execution in this repo pass.

Current evidence (all configuration-bound to k=0, q=1, current ICL prompt shape, current colorization scaffold):

| Cell | Variant | Scaffold | `image_area` | Window regime | Gate A | Gate B | Gate C |
|---|---|---|---:|---|---|---|---|
| Stage 1 (depth) | chat | depth | 518400 | inside (131072 ctx) | PASS | FAIL (unusable) | FAIL 8.60× |
| Stage 3 (colorization) | chat | colorization | 518400 | inside (131072 ctx) | PASS | FAIL (unusable) | FAIL 8.49× |
| Stage 2 (colorization) | gen | colorization | 518400 | 35% over (9216 ctx) | PASS | FAIL (unusable) | FAIL 7.23× |
| Stage 2A (colorization) | gen | colorization | 262144 | inside (9216 ctx, headroom 811) | PASS | FAIL (unusable) | FAIL 4.00× |

**Configuration-bound conclusion (preferred wording)**: on the tested zero-shot prompt shape and current colorization scaffold, the within-window re-probe remained Gate-B fail; this removes the Stage-2 context-overflow confound for this configuration.

**What is not claimed**: no Emu3-Gen-wide inability; no Emu3-family-wide inability; no claim about untested k, untested scaffolds, untested prompt shapes, or untested decode paths. Gate C is substrate-bound at the current decode path (DynamicCache grow-cat + SDPA on q_len=1) and a `StaticCache` + `torch.compile` path remains outside D-2a scope.

### SM-1 — Janus-Pro bounded second-model pilot (Round 1)

**Completed 2026-04-18 as a bounded discrimination pilot.** Janus-Pro 7B (`deepseek-ai/Janus-Pro-7B`, pinned revision `5c3eb3fb…`), fresh isolated venv, bf16 on B200, mandatory CodeX audit cleared (two passes). Packet scope: bottleneck-shape characterization + capability sanity on one model. Not a scaling study, not a method experiment.

The primary scientific output is the **dominance-structure contrast against the Emu3-Chat D-1 baseline**: at matched understanding k ∈ {0, 1}, Janus-Pro is generation-dominant (prefill/gen ≈ 0.12 / 0.14) while Emu3-Chat at the same k is prefill-dominant (prefill/gen ≈ 5.94 / 9.58). The contrast is attributable to image-tokenization budget (Janus-Pro's SigLIP-L uses ~576 LM positions per image vs Emu3's ~4,000 VQ positions), which shifts where the wall-clock lives without changing per-family internal ordering.

R3.1 (trivial T2I) passed Gate B (`usable`); R3.2 (k=0 colorization) returned Gate B `clearly_unusable` **strictly configuration-bound** to the tested zero-shot prompt shape and the current colorization scaffold. This is not a Janus-Pro-wide or family-wide inability claim.

SM-1 evidence lives under the local `deliverables/sm1_janus_pro_pilot/` tree.

### SM-2 — Janus-Pro Round 2: low-token understanding-side bottleneck-shape verification

**Completed 2026-04-18.** Bounded Round-2 packet with two PI overrides (V1 continuity as ±20% mean band on 1 warmup + 2 timed; V4 token-matched cross-family comparison as primary verdict basis, matched-k as appendix only). CodeX audit cleared 14/14 items. V1 continuity absorbed a cross-host/device change (`c1003a-s5.ufhpc` GPU1 → `c1006a-s5.ufhpc` GPU0) within band.

On a **36-measurement token-matched sample at 4 k-levels** (k ∈ {0, 1, 2, 3}, 3 queries × 3 timed runs per cell), 36 / 36 pairings show `dominance_inversion = yes` against the nearest-prompt-token Emu3 A-2 point — i.e. the SM-1 bottleneck-shape contrast is direction-stable on Janus at every rung of the tested k-ladder. The matched-k appendix extends SM-1's k=0 / k=1 inversion to k=2 and k=3 (Emu3 ratio 11.70, 14.53; Janus 0.140, 0.207).

**Owned limitation**: all Janus-Pro prompt tokens at k=0..3 (~635–2443) fall below Emu3 A-2's smallest prompt (~4242 tok @ k=0), so every token-matched pair is anchored at Emu3 k=0. The science is carried by direction-invariance on both sides of the pairing, not by raw-number equivalence. This is the precise gap SM-3 was designed to close.

**Scope bounds**: verification is bounded to GQA-testdev-balanced single-image queries at k ∈ {0,1,2,3}, prompt_tokens 629–2470, batch 1, bf16, B200, greedy decoding with `max_new_tokens=32`. Different datasets, batch sizes, decoding regimes, or token budgets may alter the numeric ratio; no family-wide extrapolation is made. SM-1 R3.2 configuration-bound `clearly_unusable` is not revisited.

SM-2 evidence lives under the local `deliverables/sm2_janus_pro_round2/` tree.

### SM-3 — Janus-Pro Round 3: token-overlap gap-closure attempt (paused at continuity gate, unresolved)

**Status: paused at the continuity gate. Unresolved. Not a branch closure. Not a final scientific negative.**

SM-3 was the bounded follow-on packet intended to close SM-2 V4's `LARGE_GAP` limitation by walking Janus-Pro up to k ∈ {4, 5, 6, 7} so that at least one cell produces a token-overlap match with Emu3 A-2. The CodeX audit cleared the code (26/26 items), and a PI-authorized narrow remediation rerun additionally cleared the code delta (GO-WITH-FLAG, one cosmetic minor).

The packet did not proceed past the per-query continuity gate (G1) against the SM-2 V2.k0 reference for the same query (qid 201640614). No downstream SM-3 cells (G2 k-ladder, G3 token-overlap analysis) were executed; no SM-3 scientific claim is on record. SM-3 is **not** treated as scientifically resolved in this pass, and no further engineer-side continuity patching is being performed.

The SM-1 and SM-2 results above are unaffected by the SM-3 pause.

SM-3 evidence lives under the local `deliverables/sm3_janus_pro_round3/` tree.

## Repo hygiene

A bounded Phase R0 refresh (2026-04-18) consolidated the working tree: DiT-track logs, the AR-2 run log, and pre-Stage-1 D-2a smoke artifacts were moved to a local `archive/` subtree, transient caches and empty stubs were removed, and `.gitignore` was tightened. No scientific claim was changed. See [`storage_and_archive_notes.md`](storage_and_archive_notes.md) for what that means for repo readers.

## Locally-held material (untracked by convention)

The following working-tree paths are maintained locally and are **not** part of git history:

- `task_plan.md`, `progress.md`, `findings.md` — planning / control-plane state (gitignored)
- `deliverables/d1_emu3_understanding_freeze/`, `deliverables/d2a_generation_capability/` — raw evidence, stage checkpoints, figures, logs (untracked)
- `recon/` — recon notes and codex audits (untracked)
- `archive/` — historical logs and superseded inventories (gitignored)

Promotion of any of these trees into tracked history is a separate PI-adjudicated decision and is not performed in this pass.

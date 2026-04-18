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

### Second-model preparation

**Upcoming, not yet active.** Second-model reconnaissance is the next preparation path once PI branch re-judgment on D-2a resolves. No second-model execution has started in this pass.

## Repo hygiene

A bounded Phase R0 refresh (2026-04-18) consolidated the working tree: DiT-track logs, the AR-2 run log, and pre-Stage-1 D-2a smoke artifacts were moved to a local `archive/` subtree, transient caches and empty stubs were removed, and `.gitignore` was tightened. No scientific claim was changed. See [`storage_and_archive_notes.md`](storage_and_archive_notes.md) for what that means for repo readers.

## Locally-held material (untracked by convention)

The following working-tree paths are maintained locally and are **not** part of git history:

- `task_plan.md`, `progress.md`, `findings.md` — planning / control-plane state (gitignored)
- `deliverables/d1_emu3_understanding_freeze/`, `deliverables/d2a_generation_capability/` — raw evidence, stage checkpoints, figures, logs (untracked)
- `recon/` — recon notes and codex audits (untracked)
- `archive/` — historical logs and superseded inventories (gitignored)

Promotion of any of these trees into tracked history is a separate PI-adjudicated decision and is not performed in this pass.

# Storage and Archive Notes

**Scope**: brief, GitHub-visible explanation of what the Phase R0 refresh did and did not do. For the full per-path manifest see the local working-tree file `deliverables/repo_refresh_cleanup/cleanup_manifest.md` (untracked).

## What Phase R0 did (2026-04-18)

- Moved stale logs and pre-evidence smoke artifacts into a local `archive/` subtree (gitignored).
- Removed transient Python bytecode (`vlm_icl/__pycache__/`) and a handful of empty directory stubs.
- Tightened `.gitignore` to keep `archive/`, the Phase-R0 cleanup manifest directory, `.codex` sentinel, and editor temps out of tracked history.
- Added a small tracked doc surface (`docs/current_project_state.md`, this file) so that remote readers can orient without depending on untracked working-tree paths.

## What Phase R0 did NOT do

- **No scientific claim changed.** All tracked result tables in `results.md` and the root `README.md` are unchanged; the only substantive addition is a conservative, configuration-bound summary of the D-2a branch state.
- **No blue-drive action.** The real storage pressure (~253 GB on `/blue/yuanyuan.lei/zihua.meng/`, ~89–120 GB actionable) is **not** touched by this pass. It is captured separately for PI adjudication in a local working-tree file (`deliverables/repo_refresh_cleanup/blue_prune_execution_shortlist.md`).
- **No promotion of working-tree trees into git history.** `deliverables/`, `recon/`, and `archive/` remain outside tracked history by existing policy.
- **No commit or push** from this pass yet.

## Where to find what

| Topic | Tracked (on GitHub) | Local working tree (not on GitHub) |
|---|---|---|
| Two-track overview | `README.md` | — |
| Current branch state | `docs/current_project_state.md` | `task_plan.md`, `progress.md`, `findings.md` |
| DiT/AR experiment tables | `docs/results.md`, `docs/milestone_summary.md` | — |
| D-1 report | — | `deliverables/d1_emu3_understanding_freeze/d1_report.md` |
| D-2a stage checkpoints | — | `deliverables/d2a_generation_capability/stage{2,2a}_checkpoint.md` |
| Recon notes and Codex audits | — | `recon/` |
| Phase R0 full manifest | — | `deliverables/repo_refresh_cleanup/cleanup_manifest.md` |
| Blue-drive prune shortlist | — | `deliverables/repo_refresh_cleanup/blue_prune_execution_shortlist.md` |
| Historical proposals / logs | — | `archive/` |

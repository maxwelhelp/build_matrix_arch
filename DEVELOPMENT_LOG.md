# Development log

This file must be updated after every meaningful run analysis and before creating a new architecture version.

The goal is to keep a clear history of:

```text
version -> what changed -> run/result -> anomalies -> interpretation -> next action
```

## Rules

1. Do not create a new version before logging the previous run analysis.
2. Do not add new primitives until reports show why current primitives/grammar fail.
3. Prefer fixes in this order:

```text
metrics/analytics -> loss/schedule fix -> execution/top-k fix -> prune/export -> new grammar/primitives
```

4. Reports must be published under:

```text
reports/<report_name>/
```

5. Lightweight report files only. No weights/checkpoints.

## Run table

| Version / run | Code mode | Key changes | Data / config | Best val acc | Last val acc | Train acc | Exec speedup | Main anomalies | Loss/pressure diagnosis | What it assembled | Decision / next action |
|---|---|---|---|---:|---:|---:|---:|---|---|---|---|
| v2 smoke | `projected_topk` | Clean StepProgram v2, fp32 frontend, JSONL events, grad analytics path | synthetic, L2 B3 S2 K2, 2 epochs | 10.16% | 10.16% | 10.94% | ~1.8 after warmup | Smoke only; not a real quality test | Loss finite; projected top-k activates after warmup | Not analyzed; synthetic smoke confirms runtime path | PASS smoke. Need real SpeechCommands run and published report. |
| v2 real projected-topk | `projected_topk` | same | SpeechCommands, L3 B4 S3 K2, pending report | pending | pending | pending | pending | User reports plateau, but report not yet pushed | Cannot diagnose without metrics/events | pending | Publish run report, then analyze before new version. |

## Current suspected issues before report

These are hypotheses, not confirmed run conclusions.

| Suspect | Why it may cause plateau | How to verify in reports | Possible fix |
|---|---|---|---|
| Top-k freezes too early | After warmup, uncomputed branches stop receiving op-gradient | `skipped_high_prob_primitives`, low `shadow` recovery, same selected ops across epochs | plateau-reactive exploration: raise temperature, increase top-k, increase shadow probability |
| Cost/write pressure too strong | Cost + write L1 + safe gate may push model to cheap/noop states | high safe_mass, low write_gate, class reads safe slots | reduce `lambda_cost`, `lambda_write_l1`; add safe-aware loss only after validation evidence |
| Class read shortcut | Output may read early/base/safe slots and bypass deeper steps | high class_read on `input/base` or high-safe slots | class-read balance or penalty only if slot utility low |
| Shared top-k per address is too coarse | Same selected ops for all samples may underfit class-specific signals | high train plateau too; skipped high-prob class-relevant primitives | top-k per class/group later, not per sample yet |
| No plateau controller | Current projected_topk has no mechanism to re-open choices when val stalls | metrics show plateau while selected/computed masks stay fixed | add PlateauController schedule, not new primitives |

## Required next analysis once report is available

Read:

```text
reports/<run>/AUTO_SUMMARY.md
reports/<run>/auto_summary.json
reports/<run>/metrics.csv
reports/<run>/latest_events.jsonl
reports/<run>/latest_analysis.json
```

Check:

```text
1. train/val gap: underfit vs overfit vs plateau
2. CE vs regularization terms: which pressure dominates
3. top grad_x_gate addresses: what receives task signal
4. low-utility writers: steps writing but not read/useful
5. class reads safe/noop slots
6. expensive computed low-gradient ops
7. skipped high-prob primitives due to top-k
8. primitive top_margin: undecided competition vs frozen choice
9. exec_speedup vs quality tradeoff
10. whether selected/computed masks are stuck after plateau
```

## Planned next mechanism if plateau is confirmed

Do not grow architecture first. Add a plateau-reactive exploration controller:

```text
if validation plateaus for N epochs:
  increase primitive temperature temporarily
  increase top-k from 3 to 4 for M epochs
  increase shadow_prob from 0.10 to 0.30
  reduce cost/write pressure temporarily
  log whether skipped high-prob branches enter top-k and improve loss
```

This is not new architecture capacity. It is a search/recovery mechanism for the current grammar.

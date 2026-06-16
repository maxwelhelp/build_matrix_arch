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
| v2 real projected-topk | `projected_topk` | projected top-k, grad analytics, JSONL reports | SpeechCommands, L3 B4 S3 K2, 10 epochs | 18.60% @ epoch 1 | 18.60% | 26.27% | 1.62 final | Program collapse: primitive entropy -> ~0, read entropy -> ~0, write_l1 -> ~0, all primitive slots choose `identity=1.0`; step reads collapse to `layer_route=1.0`; write gates ~1e-7; class_read gradients dominate | Cost pressure starts from epoch 1; entropy/write schedules harden too early; class head can get train signal while program writes almost nothing; top-k is not the root cause, collapse begins before useful program forms | Almost no real program: identity-only primitive chain, layer_route-only step read, near-zero writes, class_read over late slots | Do not add primitives. First fix loss/schedule: disable early cost/write pressure, keep entropy exploratory, add warmup activity floor or min-write target, delay top-k, add plateau re-exploration only after program has nonzero writes. |
| v2.1 stable no-collapse | `all` first, no plateau controller | Added local patch command: delayed regularization, anti-collapse activity losses, entropy floors, temporary differentiable write floor; no new primitives | SpeechCommands, L3 B4 S3 K2, pending run | pending | pending | pending | 1.0 expected in all-mode | pending | Intended fix: prevent identity/no-write collapse before measuring top-k again | Should assemble nonzero writes and nontrivial primitive mass before any hardening | Superseded by v2.2 command, which includes this fix plus sequential basis. |
| v2.2 sequential-basis | `all` first, no attention, no plateau controller | v2.1 no-collapse + layer route reduced to sequential basis + step-read phase priors + sequential metrics | SpeechCommands, L3 B4 S3 K2, pending run | pending | pending | pending | 1.0 expected in all-mode | pending | Intended fix: restore specialization by defaulting layer flow to prev_same_block and preventing global/memory layer shortcuts | Expected assembly: L0 input/state, L1/L2 prev-step flow, nonzero writes, nontrivial primitive mass, high sequential_route_mass | Run `commands/run_v2_2_sequential_basis.sh`, publish report, analyze before any projected-topk or attention. |

## Confirmed analysis for v2 real projected-topk

Report path:

```text
reports/step_program_v2_fast_projected_topk_grad/
```

Observed metrics:

```text
best val acc: 18.60% at epoch 1
last val acc: 18.60%
last train acc: 26.27%
final val loss: 6.48
final train CE: 1.3866
final exec_speedup: 1.62
```

Main collapse signals:

```text
primitive_entropy: 2.314 -> 0.000002
transition_entropy: 1.738 -> 0.000004
read_entropy: 1.840 -> 0.000002
write_l1: 0.262 -> 0.00000036
cost_loss: 0.414 -> 0.050
```

Interpretation:

```text
The architecture did not learn a useful step-program.  It learned to minimize cost and writes, collapse primitive choice to identity, collapse step reads to layer_route, and let output/class read do most of the remaining work.
```

Specific assembly found in `latest_analysis.json` / `latest_events.jsonl`:

```text
L*.B*.S*.P* mostly identity=1.0
L*.B*.S*.read mostly layer_route=1.0
write_gate around 1e-7
safe_mass reported ~0.333 per utility row because identity dominates one of the safe primitive buckets
expensive_mass ~0
class_read gradients are the largest reported gradients
```

Top-k diagnosis:

```text
Projected top-k is not the first failure.  The model already collapses to identity / no-write.  Top-k then freezes and speeds up that collapsed program.  Fix the learning pressures before changing architecture capacity.
```

## Current suspected issues after report

| Suspect | Status | Why it matters | Verification | Fix |
|---|---|---|---|---|
| Early cost pressure | confirmed likely | cost_loss drops to minimum 0.05 while program becomes identity-only | metrics cost_loss 0.414 -> 0.050 | delay cost penalty until useful writes/nontrivial ops exist; set `lambda_cost=0` for first diagnostic run |
| Write suppression | confirmed | write_l1 collapses to ~3e-7; no step writes useful state | utility rows show write_gate ~1e-7 | disable write_l1 early; add temporary write activity floor/target |
| Entropy hardening / score saturation | confirmed | primitive/read/transition entropies go nearly zero | entropy metrics collapse by epoch 3-5 | keep positive exploration longer; avoid minimizing entropy until CE improves |
| Class-read shortcut | likely | top gradients are class_read; program writes almost nothing | AUTO_SUMMARY top gradients are class_read | add class-read diagnostics; later penalize class-read on no-write slots, but not before program has useful writes |
| Top-k freezes alternatives | secondary | skipped high-prob none, because probabilities already collapsed to identity | skipped high-prob section empty | delay top-k or use all-mode until nontrivial program forms |
| Layer/step routing too free | confirmed design issue | layer route could read input/global/memory shortcuts, while step read collapsed to layer_route | latest report showed step reads mostly layer_route, not previous-step flow | v2.2 sequential-basis: remove global/memory from layer route, add route and step-read priors |
| No plateau controller | true but secondary | plateau controller cannot help if initial program already collapsed | plateau after identity collapse | add only after schedule + sequential basis produce nonzero writes |

## Required next experiment before new architecture

Run a diagnostic with no-collapse schedule and sequential basis, not a bigger architecture:

```text
primitive_exec_mode=all
no attention
no plateau controller
delayed cost/write/entropy regularization
anti-collapse activity losses
strong layer sequential prior
step-read phase priors
```

Success criteria:

```text
write_l1 stays meaningfully above zero
primitive_entropy does not instantly collapse to zero
nontrivial_mass > 0 in utility rows
sequential_route_mass high but not absolute 1.0 forever
step_prev_mass increases for S1/S2
step_layer_route_mass no longer dominates every step
class_read gradients are not the only strong gradients
train CE improves without val loss exploding immediately
```

## Planned next mechanism if plateau remains after stable sequential basis

Do not grow architecture first. Add a plateau-reactive exploration controller only after the base program can keep nonzero writes and nontrivial primitive mass:

```text
if validation plateaus for N epochs:
  increase primitive temperature temporarily
  increase top-k from 3 to 4 for M epochs
  increase shadow_prob from 0.10 to 0.30
  reduce cost/write pressure temporarily
  log whether skipped high-prob branches enter top-k and improve loss
```

This is not new architecture capacity. It is a search/recovery mechanism for the current grammar.

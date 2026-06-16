# build_matrix_arch

Clean repository for the differentiable matrix-program architecture builder.

The current active experiment is a clean StepProgram core for SpeechCommands.  It is not a continuation of the old huge v13 script.  The goal is to test the heart of the idea:

```text
Input/Evidence -> Layer -> Block -> Step -> Primitive slots -> Output read -> Loss
```

The model logs what every important address is doing:

```text
L{layer}.B{block}.S{step}.P{primitive_slot}
L{layer}.B{block}.S{step}.P{k}->P{k+1}
L{layer}.B{block}.S{step}.read
L{layer}.B{block}.route
class.<name>.read.<slot>
```

## Main architecture document

Read and update:

```text
ARCHITECTURE.md
```

This file is the source of truth: grammar, logging contract, projected top-k design, metrics, what is active, and what is rejected/inactive.

## Development log

Read and update after every meaningful run analysis, before writing a new version:

```text
DEVELOPMENT_LOG.md
```

It contains a Markdown table with version, changes, metrics, anomalies, diagnosis, and next action.

## Main experiment

```bash
python experiments/step_program/run_step_program_v2_projected_topk.py --help
```

Smoke test:

```bash
bash commands/run_v2_smoke.sh
```

Stable no-collapse diagnostic run.  Use this before adding new architecture logic:

```bash
bash commands/run_v2_stable_no_collapse.sh
```

This command patches v2 locally with delayed regularization and anti-collapse losses, compiles it, and runs the stable diagnostic.

Fast projected-topk run with gradient analytics.  If this repository is cloned next to the old `architecture_builder` repo, use:

```bash
bash commands/run_v2_fast_projected_topk_grad_local.sh
```

Or explicitly set the data path:

```bash
DATA_ROOT=/path/to/speechcommands/root bash commands/run_v2_fast_projected_topk_grad_local.sh
```

Full all-primitives baseline:

```bash
bash commands/run_v2_all_baseline_local.sh
```

## Output format

Every run writes:

```text
runs/<run>/metrics.csv
runs/<run>/analysis_epoch_XXX.json
runs/<run>/events_epoch_XXX.jsonl
runs/<run>/best.pt
runs/<run>/last.pt
```

`analysis_epoch_XXX.json` is human readable.  `events_epoch_XXX.jsonl` is flat and queryable: one JSON object per address/event, designed for automatic analysis.

Example query:

```bash
python tools/query_events.py ./runs/step_program_v2_fast_projected_topk_grad/events_epoch_001.jsonl --top 20 --sort grad_x_gate
```

## Auto-analysis

Create a compact summary for the latest run:

```bash
python tools/analyze_run.py ./runs/step_program_v2_stable_no_collapse
```

This writes:

```text
AUTO_SUMMARY.md
auto_summary.json
```

Publish lightweight report files to GitHub, without checkpoints:

```bash
bash commands/publish_run_report.sh ./runs/step_program_v2_stable_no_collapse step_program_v2_stable_no_collapse "Add stable no-collapse report"
```

Published reports go to:

```text
reports/<report_name>/
```

The publish command copies only lightweight analysis files: metrics, latest analysis/events, auto summary, final report if present.

## Current rule

Do not add new primitives until the current reports prove what is failing.  First optimize execution, log gradients, analyze competition/utility, then decide what to add or remove.

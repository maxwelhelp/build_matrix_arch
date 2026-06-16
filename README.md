# build_matrix_arch

Clean repository for the differentiable matrix-program architecture builder.

## Active direction: v3 clean sequential

The current active experiment is:

```text
experiments/step_program/run_step_program_v3_clean_sequential.py
```

Core idea:

```text
Input/Evidence
  -> L0 -> L1 -> L2 -> L3
  -> inside each layer: S0 -> S1 -> S2
  -> inside each step: K MatrixFamilyUnit substeps
  -> final-layer output head
  -> loss
```

v3 intentionally removes the main v2 shortcuts:

```text
no layer-route soup
no free step-read router
no identity primitive candidate
no output class-read shortcut
no projected top-k yet
no plateau controller yet
```

The learnable choice is inside each matrix-family unit:

```text
F(x, ctx) = sum_i softmax(gate)_i * Family_i(x, ctx)
h <- Norm(h + write_gate * F(h, ctx))
```

Families:

```text
small_refine
diag_delta
low_rank
butterfly
blockdiag
compare
normalize
```

Identity is not a primitive. Identity exists only as the residual skip around the learned update.

## Archived v2 line

The v2/v2.1/v2.2 experiments are now diagnostic/archive only. See:

```text
archive/v2/README.md
```

They remain in the repository so old reports can be reproduced, but they are no longer the main development path.

## Main architecture document

Read and update:

```text
ARCHITECTURE.md
```

## Development log

Read and update after every meaningful run analysis, before writing a new version:

```text
DEVELOPMENT_LOG.md
```

Short rule: publish report -> analyze -> discuss with Maxim -> then update `DEVELOPMENT_LOG.md` briefly.

## Commands

Smoke test:

```bash
bash commands/run_v3_clean_seq_smoke.sh
```

SpeechCommands run:

```bash
bash commands/run_v3_clean_seq_speechcommands.sh
```

If the dataset is somewhere else:

```bash
DATA_ROOT=/path/to/speechcommands/root bash commands/run_v3_clean_seq_speechcommands.sh
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

`analysis_epoch_XXX.json` is human readable. `events_epoch_XXX.jsonl` is flat and queryable: one JSON object per address/event, designed for automatic analysis.

Example query:

```bash
python tools/query_events.py ./runs/step_program_v3_clean_seq_speechcommands/events_epoch_001.jsonl --top 20 --sort grad_x_gate
```

## Auto-analysis

Create a compact summary:

```bash
python tools/analyze_run.py ./runs/step_program_v3_clean_seq_speechcommands
```

Publish lightweight report files to GitHub, without checkpoints:

```bash
bash commands/publish_run_report.sh ./runs/step_program_v3_clean_seq_speechcommands step_program_v3_clean_seq_speechcommands "Add v3 clean sequential report"
```

Published reports go to:

```text
reports/<report_name>/
```

## Current rule

Do not add attention, projected top-k, growth, or plateau controller until v3 proves whether the clean sequential matrix-family program learns.

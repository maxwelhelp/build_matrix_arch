# build_matrix_arch

Clean repository for the differentiable matrix-program architecture builder.

## Active next experiment: v3.1 no-router matrix chain

The current next experiment is:

```text
experiments/step_program/run_step_program_v3_1_no_router_matrix_chain.py
```

Core idea:

```text
Input/Evidence
  -> L0 -> L1 -> L2 -> L3
  -> inside each layer: S0 -> S1 -> S2
  -> inside each step: fixed matrix-operation chain
  -> final-layer output head
  -> loss
```

v3.1 removes the remaining router-like softmax family choice from v3.

There is no operation selection.  Every operation runs in fixed order and receives gradient:

```text
h <- Norm(h + gain_small     * small_refine(h, ctx))
h <- Norm(h + gain_diag      * diag_delta(h))
h <- Norm(h + gain_low_rank  * low_rank(h))
h <- Norm(h + gain_butterfly * butterfly(h))
h <- Norm(h + gain_blockdiag * blockdiag(h))
h <- Norm(h + gain_compare   * compare(h, ctx))
h <- Norm(h + gain_normalize * normalize_delta(h))
```

The gains are independent sigmoid gates, not a softmax.  Multiple operations can be active together.  Specialization is measured through `gain`, `update_norm`, and `grad_x_gain` per address.

## Previous active line: v3 clean sequential

The previous v3 file remains available:

```text
experiments/step_program/run_step_program_v3_clean_sequential.py
```

It removed the v2 layer/step/class-read shortcuts, but still had a softmax family gate. The first v3 report showed balanced data but no learning: val stayed at 10%, train stayed near 10%, and the confusion matrix predicted one class for all examples. Therefore v3.1 removes the family softmax and tests a fixed chain.

## Archived v2 line

The v2/v2.1/v2.2 experiments are diagnostic/archive only. See:

```text
archive/v2/README.md
```

They remain in the repository so old reports can be reproduced, but they are no longer the main development path.

## Development log

Read and update after every meaningful run analysis, before writing a new version:

```text
DEVELOPMENT_LOG.md
```

Short rule: publish report -> analyze -> discuss with Maxim -> then update `DEVELOPMENT_LOG.md` briefly.

## Commands

v3.1 smoke test:

```bash
bash commands/run_v3_1_no_router_chain_smoke.sh
```

v3.1 SpeechCommands run:

```bash
bash commands/run_v3_1_no_router_chain_speechcommands.sh
```

If the dataset is somewhere else:

```bash
DATA_ROOT=/path/to/speechcommands/root bash commands/run_v3_1_no_router_chain_speechcommands.sh
```

Older v3 softmax-family commands:

```bash
bash commands/run_v3_clean_seq_smoke.sh
bash commands/run_v3_clean_seq_speechcommands.sh
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
python tools/query_events.py ./runs/step_program_v3_1_no_router_chain_speechcommands/events_epoch_001.jsonl --top 20 --sort grad_x_gain
```

## Auto-analysis

Create a compact summary:

```bash
python tools/analyze_run.py ./runs/step_program_v3_1_no_router_chain_speechcommands
```

Publish lightweight report files to GitHub, without checkpoints:

```bash
bash commands/publish_run_report.sh ./runs/step_program_v3_1_no_router_chain_speechcommands step_program_v3_1_no_router_chain_speechcommands "Add v3.1 no-router chain report"
```

Published reports go to:

```text
reports/<report_name>/
```

## Current rule

Do not add attention, projected top-k, growth, or plateau controller until v3.1 proves whether the fixed sequential matrix-chain learns.

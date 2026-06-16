# build_matrix_arch

Clean repository for the differentiable matrix-program architecture builder.

## Active next experiment: v3.1 no-router matrix chain with fixed input/head

The current next experiment is:

```text
experiments/step_program/run_step_program_v3_1_no_router_matrix_chain.py
```

Before running, commands apply:

```text
tools/patch_v3_block_input_concat_head.py
```

This fixes the shared v3/v3.1 failure mode:

```text
old: all blocks start from mean(evidence) + block_emb
new: every block gets its own learned evidence projection

old: final head receives mean(final_blocks)
new: final head receives concat(final_blocks)

old: input context starts very weak
new: S0 gets strong input context, S1 medium, S2 weaker
```

Core idea remains natural and sequential:

```text
Input/Evidence
  -> block-specific input states
  -> L0 -> L1 -> L2 -> L3
  -> inside each layer: S0 -> S1 -> S2
  -> inside each step: fixed matrix-operation chain
  -> concat final blocks
  -> final-layer output head
  -> loss
```

v3.1 has no operation selection. Every operation runs in fixed order and receives gradient:

```text
h <- Norm(h + gain_small     * small_refine(h, ctx))
h <- Norm(h + gain_diag      * diag_delta(h))
h <- Norm(h + gain_low_rank  * low_rank(h))
h <- Norm(h + gain_butterfly * butterfly(h))
h <- Norm(h + gain_blockdiag * blockdiag(h))
h <- Norm(h + gain_compare   * compare(h, ctx))
h <- Norm(h + gain_normalize * normalize_delta(h))
```

The gains are independent sigmoid gates, not a softmax. Multiple operations can be active together. Specialization is measured through `gain`, `update_norm`, and `grad_x_gain` per address.

## Previous reports

The first v3 and v3.1 reports both stayed at 10% despite balanced data. v3.1 showed update norms growing, so operations were running, but the classifier still received no useful class signal. The likely issue was shared block initialization plus final block averaging.

## Archived v2 line

The v2/v2.1/v2.2 experiments are diagnostic/archive only. See:

```text
archive/v2/README.md
```

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

Older v3 softmax-family commands, also patched with fixed block input/head:

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
python tools/query_events.py ./runs/step_program_v3_1_no_router_chain_fixed_input/events_epoch_001.jsonl --top 20 --sort grad_x_gain
```

## Auto-analysis

Create a compact summary:

```bash
python tools/analyze_run.py ./runs/step_program_v3_1_no_router_chain_fixed_input
```

Publish lightweight report files to GitHub, without checkpoints:

```bash
bash commands/publish_run_report.sh ./runs/step_program_v3_1_no_router_chain_fixed_input step_program_v3_1_no_router_chain_fixed_input "Add v3.1 fixed-input no-router chain report"
```

Published reports go to:

```text
reports/<report_name>/
```

## Current rule

Do not add attention, projected top-k, growth, or plateau controller until v3.1 fixed-input proves whether the fixed sequential matrix-chain learns.

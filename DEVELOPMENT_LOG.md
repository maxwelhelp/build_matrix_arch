# Development log

Short version history only. Detailed analysis belongs in `reports/<run>/`.

## Rule

Update this file only after:

```text
1. report is published
2. report is analyzed
3. conclusion is discussed with Maxim
4. next action is agreed
```

Do not use this file as a scratchpad. Keep one short row per tested version/run.

## Version table

| Version / run | Report | Result | Key diagnosis | Decision / next action |
|---|---|---|---|---|
| v2 smoke | local smoke | PASS runtime; projected_topk activates after warmup | synthetic smoke only, not a quality test | real SpeechCommands run required |
| v2 projected-topk | `reports/step_program_v2_fast_projected_topk_grad/` | FAIL: best val 18.60%, last val 18.60%, train 26.27%, speedup 1.62 | collapse to identity/no-write; step reads mostly layer_route; class_read carries gradient; top-k not root cause | do not add primitives; fix schedule/write pressure and routing basis |
| v2.1 stable no-collapse | pending | planned diagnostic | delayed regularization + anti-collapse losses + write floor; no new primitives | superseded by v2.2 command before full run |
| v2.2 sequential-basis | pending | planned diagnostic | v2.1 + sequential layer route + step-read phase priors; no attention, no plateau controller | run `commands/run_v2_2_sequential_basis.sh`, publish report, then discuss before new code |

## Release tags

Use annotated git tags only for discussed/tested versions:

```bash
bash commands/tag_version.sh v0.2.2 "v0.2.2 sequential-basis diagnostic"
```

Rollback:

```bash
git fetch --tags
git checkout v0.2.2
```

See:

```text
docs/RELEASE_WORKFLOW.md
```

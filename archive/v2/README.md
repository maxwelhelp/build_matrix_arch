# Archive: v2 diagnostic line

Status: archived / diagnostic only.

The v2 line remains in the repository for comparison and report reproducibility, but it is no longer the main architecture direction.

## Why archived

v2 grew through patches:

```text
v2 projected_topk
v2.1 stable no-collapse
v2.2 sequential-basis routing
```

This helped diagnose failures, but the file accumulated too much mixed logic:

```text
layer route choices
step read choices
projected top-k
anti-collapse losses
entropy floors
write floors
patch scripts
```

The key findings were useful:

```text
1. Free layer/step routing can destroy sequential specialization.
2. Output class-read over all slots can bypass the step-program.
3. Identity as a primitive can dominate and kill real transformation.
4. Write suppression and cost pressure can collapse the program.
5. Sequential layer/step skeleton is necessary, but not sufficient if primitives still pick identity.
```

## New active direction

Use v3 clean sequential:

```text
experiments/step_program/run_step_program_v3_clean_sequential.py
commands/run_v3_clean_seq_smoke.sh
commands/run_v3_clean_seq_speechcommands.sh
```

v3 removes the main v2 shortcuts:

```text
no layer-route soup
no free step-read router
no identity primitive candidate
no output class-read shortcut
strict layer sequence
strict step sequence
matrix-family choice inside each step
```

Old v2 files are kept for reproducing old reports only.

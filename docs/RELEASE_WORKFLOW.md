# Release and development workflow

This repository should be easy to roll back, compare, and reason about.

## Main rule

`DEVELOPMENT_LOG.md` is not a scratchpad.

It is updated only after:

```text
1. a run report is published under reports/<run>/
2. the report is analyzed
3. the result is discussed with Maxim
4. the next decision is agreed
```

Detailed analysis belongs in:

```text
reports/<run>/AUTO_SUMMARY.md
reports/<run>/auto_summary.json
reports/<run>/latest_analysis.json
reports/<run>/latest_events.jsonl
```

`DEVELOPMENT_LOG.md` must stay short.

## Versioning

Use tags for tested versions:

```text
v0.1.0   initial clean StepProgram
v0.2.0   projected-topk diagnostic
v0.2.1   stable no-collapse schedule
v0.2.2   sequential-basis routing
```

Patch scripts/commands can exist before a version tag, but a version tag should be created only after a report is analyzed and the result is accepted.

## Recommended flow

### 1. Implement / patch

Small, focused changes only:

```text
one architecture idea or one training/schedule fix per version
```

### 2. Run

Run the command from `commands/`.

### 3. Publish report

```bash
bash commands/publish_run_report.sh ./runs/<run_dir> <report_name> "Add <report_name> report"
```

This publishes lightweight report files only. No checkpoints.

### 4. Analyze and discuss

Use:

```text
AUTO_SUMMARY.md
metrics.csv
latest_events.jsonl
latest_analysis.json
```

Then discuss the conclusion with Maxim.

### 5. Update short devlog

Only after the discussion, add one row to `DEVELOPMENT_LOG.md`:

```text
version | report | result | decision | next
```

### 6. Tag tested version

Use:

```bash
bash commands/tag_version.sh v0.2.2 "v0.2.2 sequential-basis diagnostic"
```

## Rollback

To inspect an old version without changing branches:

```bash
git fetch --tags
git checkout v0.2.2
```

To return to main:

```bash
git checkout main
git pull --rebase origin main
```

To make a branch from an old version:

```bash
git checkout -b experiment/from-v0.2.2 v0.2.2
```

## Branch policy

`main` should stay runnable.

Use branches only when a change is large or risky:

```text
experiment/<short-name>
fix/<short-name>
```

For the current small research loop, direct commits to main are acceptable if each commit is focused and reports are published.

## What not to do

Do not:

```text
- add long raw analysis into DEVELOPMENT_LOG.md
- tag untested code as a version
- add new primitives before report analysis proves the need
- push checkpoints/weights/data
- create a plateau controller before the base program is stable
```

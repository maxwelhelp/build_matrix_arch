# Build Matrix Architecture — source of truth

This file is the living design contract for the repository.

## Core goal

We do not want to hand-pick a fixed architecture.  We want to define a differentiable space of small matrix programs, train it end-to-end, then sharpen, prune, export, and reuse the useful parts.

## Active minimal core

```text
Input / Evidence
  -> Layer
    -> Block
      -> Step
        -> Primitive slots
        -> Primitive-to-primitive transitions
      -> Step-to-step reads
    -> Layer/block routes
  -> Output/class read
  -> Loss
```

Gradients must flow through:

```text
class_read
layer/block route gates
step read gates
primitive transition gates
primitive gates
primitive parameters
input/memory/global paths
```

## Logging contract

Every report must use stable address strings.

### Primitive inside step

```text
L{l}.B{b}.S{s}.P{k}.{primitive}
```

Required fields:

```text
prob
selected/computed
cost
entropy
safe_mass/nontrivial_mass
grad_x_gate when enabled
```

### Primitive transition

```text
L{l}.B{b}.S{s}.P{k}->P{k+1}.{transition}
```

Required fields:

```text
prob
grad_x_gate
entropy
```

### Step read

```text
L{l}.B{b}.S{s}.read.{source}
```

Sources:

```text
state
input
prev_step
all_prev_steps
layer_route
global
memory
task
```

Hard mask only impossible reads, for example S0 reading previous steps in sequential mode.

### Layer/block route

```text
L{l}.B{b}.route.{source}
```

Sources:

```text
input
prev_same_block
prev_layer_mean
global_mean
memory_mean
```

### Output read

```text
class.{class_name}.read.{slot_address}
```

This tells which step slot directly influences each class.

### Utility

```text
L{l}.B{b}.S{s}.utility
```

Required fields:

```text
write_gate
class_read_mass
safe_mass
nontrivial_mass
expensive_mass
memory_write_gate
global_write_gate
memory_primitive_mass
global_primitive_mass
utility_proxy
suspicious flags
```

## ProjectedTopK execution

The main speed idea is not to remove the program logic.  It is to stop fully computing every primitive candidate forever.

### Bad version

```text
top-1 from the first batch
unselected branches receive no signal
router mistakes become permanent
```

This kills intelligence.

### Correct version

```text
warmup:       compute all primitives fully, soft gates, gradient to all
projected:    cheap sketch scores all primitives
top-k:        full compute only selected heavy primitives
shadow:       sometimes compute one extra unselected primitive
analysis:     log selected/computed/skipped and grad_x_gate
late/export:  keep compact useful program
```

### Selection granularity

First implementation uses selection per address shared across batch:

```text
L.B.S.P selects top-k primitives
```

This avoids per-sample scatter chaos on Tesla P40 and makes reports easy to parse.

### Safe primitives

Safe primitives are always computed because they are cheap and protect the state:

```text
noop
identity
keep_state
small_refine
normalize
```

Heavy primitives can be projected/top-k:

```text
mlp
matrix_mlp
compare
suppress
```

Memory/global reads are treated as cheap context outputs after their context has already been computed.

## Metrics that matter

### Gate metrics

```text
prob / gate_mass
entropy
selected/computed/skipped
```

### Gradient metrics

```text
grad_x_gate = mean(abs(d loss / d gate) * gate)
```

Interpretation:

```text
high gate + high grad = active useful/contested element
high gate + low grad  = used but not learning, possible dead shortcut
low gate + high grad  = suppressed element wants to enter, possible replacement candidate
low gate + low grad   = prune candidate
```

### Competition metrics

For each address, report top options and margin:

```text
top1
top2
margin = top1_prob - top2_prob
entropy
```

Low margin/high entropy means the address is undecided or competing.

### Utility/suspicious metrics

Suspicious but not necessarily forbidden:

```text
class reads safe/noop slot
expensive primitive mass with low utility
memory write without memory primitive mass
global write without global primitive mass
high read/write but low gradient
```

## Constraint policy

Hard-mask only impossible choices:

```text
S0 reads prev_step/all_prev_steps
future step read in strict sequential mode
shape-invalid route
```

Soft-penalize or only log suspicious choices.  Do not ban unusual combinations too early; unusual programs may be useful.

## Current implementation

```text
experiments/step_program/run_step_program_v2_projected_topk.py
```

Active modes:

```text
--primitive-exec-mode all
--primitive-exec-mode projected_topk
```

The first task is analysis, not adding more primitives.

## Inactive / rejected

### Huge v13 patching as main direction

Status: inactive as main path.

Reason: too much logic in one file makes it hard to know what works.  v13 can remain a baseline, but the new repo should stay clean.

### Permanent soft primitive soup

Status: rejected as final architecture.

Reason: soft mixtures are useful for warmup, but if they never harden, they overfit and do not become readable programs.

### Dynamic growth before utility tracking

Status: rejected for now.

Reason: without step-level utility and gradients, growth will add capacity in the wrong place.

## Roadmap

Immediate:

```text
1. run all-primitives baseline
2. run projected_topk baseline
3. compare speed/accuracy/gap
4. inspect events JSONL for gradients, competition, suspicious slots
5. only then decide what to prune/add
```

Next:

```text
1. improve projected_topk if quality falls
2. add shadow utility comparison
3. add automatic report summarizer
4. add top-k export/prune
```

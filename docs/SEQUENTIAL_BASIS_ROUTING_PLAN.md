# Sequential basis and differentiable routing plan

This document fixes the main logical issue found after the v2 projected-topk collapse: choices inside steps are not enough if layer/step routing destroys sequential specialization.

## Current problem

The current layer route is differentiable, but too free too early.

It chooses a softmax mixture over sources such as:

```text
input
prev_same_block
prev_layer_mean
global_mean
memory_mean
```

This is trainable, but it can bypass the intended program chain.  If every layer can cheaply read the same global/input/mean context, then layers may stop specializing.

Collapse pattern observed:

```text
step read -> layer_route
primitive -> identity
write_gate -> almost zero
class_read -> tries to classify from weak slots
```

So the issue is not only loss pressure.  It is also missing sequential prior.

## Correct principle

Routing must be differentiable, but not structureless.

The model should start from a strong sequential basis:

```text
input -> L0 -> L1 -> L2 -> ... -> output
```

Then it can learn deviations through soft gates.

This means:

```text
start mostly sequential
allow input skips
allow block/layer summaries as weak context
allow memory/global through steps, not as easy layer-level shortcuts
open freer routing only after stable nonzero writes exist
```

## Layer route design

### v2 problem route

```text
route = softmax(router(state, prev_layer, global, memory))
sources = [input, prev_same_block, prev_layer_mean, global_mean, memory_mean]
```

This can let a layer ignore its predecessor and repeatedly read shared context.

### v2.2 sequential route

Use a smaller basis first:

```text
prev_same_block      main sequential path
input_skip           weak raw input skip
prev_layer_mean      weak cross-block context
```

Remove from layer route for now:

```text
global_mean
memory_mean
```

Reason: memory/global should be read inside steps as explicit operations, not as a cheap layer-bypass.

### Initial priors

Initialize route logits so the default is sequential:

```text
prev_same_block: +2.0
input_skip:      -0.5
prev_layer_mean: -1.0
```

This gives a strong backbone but still allows gradients to move probability mass.

## Step read design

Step routing also needs a phase prior.

Allowed sources remain:

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

But the initial priors should depend on step index.

### Step 0

```text
state/input high
prev_step/all_prev_steps masked
layer_route moderate
global/memory low
```

Meaning: first step extracts or prepares.

### Step 1

```text
prev_step high
state/input moderate
layer_route moderate
global/memory low-to-moderate
```

Meaning: second step transforms/compares previous result.

### Step 2+

```text
prev_step high
all_prev_steps moderate
global/memory allowed
task allowed
```

Meaning: later steps repair, aggregate, or use registers.

## Between primitive slots inside a step

Primitive slots should also have a default chain:

```text
P0 prepares
P1 transforms / compares
P2 writes/refines when present
```

Do not make transitions hard.  Use soft basis with priors:

```text
P0->P1: residual / norm_residual favored
P1->P2: residual / compare_mix favored
keep/replace allowed but not dominant initially
```

## What differentiable routing can and cannot do

It can:

```text
move probability mass through gradients
learn soft mixtures
specialize addresses L.B.S.P
prefer different routes per layer/block/step
```

It cannot magically invent a new source unless the architecture gives it a free learned route/path.

Therefore a future optional path can be:

```text
route_output = basis_route_mix + free_gate * learned_route(context)
```

But free_gate should start near zero and only open after the sequential basis is stable.

## Matrix attention option

A matrix-attention route is possible, but it should not replace the sequential backbone first.

Better design:

```text
sequential_backbone = mostly prev_same_block
attn_context = masked matrix attention over allowed previous slots
route_output = backbone_gate * sequential_backbone + context_gate * attn_context
```

Mask rules:

```text
layer l can read only input and layers <= l-1
step s can read only current state and steps <= s-1
no future layer/step reads in sequential_exact mode
```

This keeps differentiability without cheating.

## Metrics required

Add these metrics before judging the next run:

```text
sequential_route_mass      probability on prev_same_block
input_skip_mass            probability on input_skip
prev_layer_mean_mass       probability on prev_layer_mean
memory_global_layer_mass   should be zero in v2.2 layer route
step_prev_mass             prev_step + all_prev_steps
step_input_mass            state + input
step_register_mass         global + memory
average_write_gate
nontrivial_mass
identity_mass
class_read_on_low_write_slots
```

Success signs:

```text
layer route starts high on prev_same_block, then selectively deviates
S0 reads input/state
S1/S2 read previous steps
write gates stay nonzero
primitive choices do not collapse to identity everywhere
class_read uses slots that actually wrote useful state
```

## Implementation order

Do not add new primitives and do not add a plateau controller yet.

1. Keep the v2.1 no-collapse schedule.
2. Add v2.2 sequential route basis.
3. Add step-read phase priors.
4. Add route/step metrics listed above.
5. Run all-mode diagnostic first.
6. Only after stable specialization appears, re-enable projected_topk.

## Key rule

The architecture should learn choices, but choices need a useful grammar.

Without sequential grammar:

```text
many choices -> same shortcut -> no specialization
```

With sequential grammar:

```text
strong default program chain -> differentiable deviations -> specialization
```

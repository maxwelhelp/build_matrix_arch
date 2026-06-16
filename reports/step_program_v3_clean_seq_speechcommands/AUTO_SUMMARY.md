# Auto summary: `step_program_v3_clean_seq_speechcommands`

## Quality

- epochs: `10`
- best val acc: `10.00%` at epoch `1`
- last val acc: `10.00%`
- last train acc: `9.97%`
- exec stats: `{}`

## Top gradient addresses

| type | address | grad_x_gate | prob | cost | computed | selected |
|---|---|---|---|---|---|---|
| layer | L0.input_skip_gate |  |  |  |  |  |
| layer | L0.mean_skip_gate |  |  |  |  |  |
| utility | L0.B0.S0 |  |  |  |  |  |
| family | L0.B0.S0.K0.family.small_refine | 0 | 0.0821334 |  |  |  |
| family | L0.B0.S0.K0.family.diag_delta | 0 | 0.309874 |  |  |  |
| family | L0.B0.S0.K0.family.low_rank | 0 | 0.0401927 |  |  |  |
| family | L0.B0.S0.K0.family.butterfly | 0 | 0.18119 |  |  |  |
| family | L0.B0.S0.K0.family.blockdiag | 0 | 0.250454 |  |  |  |
| family | L0.B0.S0.K0.family.compare | 0 | 0.0681367 |  |  |  |
| family | L0.B0.S0.K0.family.normalize | 0 | 0.0680204 |  |  |  |
| write_gate | L0.B0.S0.K0.write_gate | 0 |  |  |  |  |
| family | L0.B0.S0.K1.family.small_refine | 0 | 0.0025438 |  |  |  |
| family | L0.B0.S0.K1.family.diag_delta | 0 | 0.797679 |  |  |  |
| family | L0.B0.S0.K1.family.low_rank | 0 | 0.00732487 |  |  |  |
| family | L0.B0.S0.K1.family.butterfly | 0 | 0.0989676 |  |  |  |

## Top utility slots

| address | utility_proxy | write_gate | class_read_mass | safe_mass | expensive_mass |
|---|---|---|---|---|---|
| L0.B0.S0 |  |  |  |  |  |
| L0.B0.S1 |  |  |  |  |  |
| L0.B0.S2 |  |  |  |  |  |
| L0.B1.S0 |  |  |  |  |  |
| L0.B1.S1 |  |  |  |  |  |
| L0.B1.S2 |  |  |  |  |  |
| L0.B2.S0 |  |  |  |  |  |
| L0.B2.S1 |  |  |  |  |  |
| L0.B2.S2 |  |  |  |  |  |
| L0.B3.S0 |  |  |  |  |  |
| L0.B3.S1 |  |  |  |  |  |
| L0.B3.S2 |  |  |  |  |  |
| L1.B0.S0 |  |  |  |  |  |
| L1.B0.S1 |  |  |  |  |  |
| L1.B0.S2 |  |  |  |  |  |

## Low utility writers

| address | write_gate | utility_proxy | class_read_mass | safe_mass | expensive_mass |
|---|---|---|---|---|---|
| L0.B0.S0 |  |  |  |  |  |
| L0.B0.S1 |  |  |  |  |  |
| L0.B0.S2 |  |  |  |  |  |
| L0.B1.S0 |  |  |  |  |  |
| L0.B1.S1 |  |  |  |  |  |
| L0.B1.S2 |  |  |  |  |  |
| L0.B2.S0 |  |  |  |  |  |
| L0.B2.S1 |  |  |  |  |  |
| L0.B2.S2 |  |  |  |  |  |
| L0.B3.S0 |  |  |  |  |  |
| L0.B3.S1 |  |  |  |  |  |
| L0.B3.S2 |  |  |  |  |  |
| L1.B0.S0 |  |  |  |  |  |
| L1.B0.S1 |  |  |  |  |  |
| L1.B0.S2 |  |  |  |  |  |

## Competing primitives: low top margin

_none_

## Expensive computed with low gradient

_none_

## Skipped but high probability

_none_

## Suspicious

_none_

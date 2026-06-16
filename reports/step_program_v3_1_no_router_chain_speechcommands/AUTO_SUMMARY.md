# Auto summary: `step_program_v3_1_no_router_chain_speechcommands`

## Quality

- epochs: `10`
- best val acc: `10.00%` at epoch `1`
- last val acc: `10.00%`
- last train acc: `10.03%`
- exec stats: `{}`

## Top gradient addresses

| type | address | grad_x_gate | prob | cost | computed | selected |
|---|---|---|---|---|---|---|
| layer | L0.input_skip_gate |  |  |  |  |  |
| layer | L0.mean_skip_gate |  |  |  |  |  |
| utility | L0.B0.S0 |  |  |  |  |  |
| op | L0.B0.S0.K0.op.small_refine |  | 0.0350578 |  |  |  |
| op | L0.B0.S0.K0.op.diag_delta |  | 0.0505803 |  |  |  |
| op | L0.B0.S0.K0.op.low_rank |  | 0.0211483 |  |  |  |
| op | L0.B0.S0.K0.op.butterfly |  | 0.0261767 |  |  |  |
| op | L0.B0.S0.K0.op.blockdiag |  | 0.0434054 |  |  |  |
| op | L0.B0.S0.K0.op.compare |  | 0.0112919 |  |  |  |
| op | L0.B0.S0.K0.op.normalize |  | 0.0312567 |  |  |  |
| utility | L0.B0.S1 |  |  |  |  |  |
| op | L0.B0.S1.K0.op.small_refine |  | 0.044878 |  |  |  |
| op | L0.B0.S1.K0.op.diag_delta |  | 0.0518295 |  |  |  |
| op | L0.B0.S1.K0.op.low_rank |  | 0.044607 |  |  |  |
| op | L0.B0.S1.K0.op.butterfly |  | 0.054333 |  |  |  |

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

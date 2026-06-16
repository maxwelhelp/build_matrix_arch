# Auto summary: `step_program_v2_fast_projected_topk_grad`

## Quality

- epochs: `10`
- best val acc: `18.60%` at epoch `1`
- last val acc: `18.60%`
- last train acc: `26.27%`
- exec stats: `{'computed_ops': 488.0, 'all_ops': 792.0, 'estimated_speedup': 1.6229508196721312}`

## Top gradient addresses

| type | address | grad_x_gate | prob | cost | computed | selected |
|---|---|---|---|---|---|---|
| class_read | class.left.read.L2.B0.S2 | 4.28777e-05 | 0.027411 |  |  |  |
| class_read | class.left.read.L2.B1.S2 | 4.24666e-05 | 0.0274796 |  |  |  |
| class_read | class.left.read.L2.B0.S1 | 4.21534e-05 | 0.0273604 |  |  |  |
| class_read | class.left.read.L2.B2.S2 | 4.21481e-05 | 0.027439 |  |  |  |
| class_read | class.left.read.L2.B3.S2 | 4.18989e-05 | 0.0274323 |  |  |  |
| class_read | class.left.read.L2.B1.S1 | 4.16126e-05 | 0.0274386 |  |  |  |
| class_read | class.left.read.L2.B0.S0 | 4.12772e-05 | 0.0273045 |  |  |  |
| class_read | class.left.read.L2.B2.S1 | 4.12768e-05 | 0.0273934 |  |  |  |
| class_read | class.left.read.L2.B3.S1 | 4.10162e-05 | 0.0273846 |  |  |  |
| class_read | class.down.read.L2.B0.S2 | 4.07313e-05 | 0.0273467 |  |  |  |
| class_read | class.left.read.L2.B1.S0 | 4.06739e-05 | 0.0273926 |  |  |  |
| class_read | class.left.read.L2.B2.S0 | 4.02667e-05 | 0.027342 |  |  |  |
| class_read | class.left.read.L2.B3.S0 | 4.00233e-05 | 0.0273299 |  |  |  |
| class_read | class.down.read.L2.B0.S1 | 3.99443e-05 | 0.0273306 |  |  |  |
| class_read | class.down.read.L2.B1.S2 | 3.98578e-05 | 0.0272798 |  |  |  |

## Top utility slots

| address | utility_proxy | write_gate | class_read_mass | safe_mass | expensive_mass |
|---|---|---|---|---|---|
| L0.B2.S0 | 4.03392e-09 | 1.60933e-07 | 0.0250659 | 0.333333 | 0 |
| L0.B2.S1 | 3.66318e-09 | 1.43051e-07 | 0.0256075 | 0.333333 | 0 |
| L0.B1.S0 | 3.61647e-09 | 1.43051e-07 | 0.025281 | 0.333333 | 0 |
| L0.B3.S0 | 3.59624e-09 | 1.43051e-07 | 0.0251395 | 0.333333 | 0 |
| L0.B2.S2 | 3.26531e-09 | 1.2517e-07 | 0.0260871 | 0.333333 | 0 |
| L0.B1.S1 | 3.2287e-09 | 1.2517e-07 | 0.0257946 | 0.333333 | 0 |
| L0.B3.S1 | 3.21304e-09 | 1.2517e-07 | 0.0256695 | 0.333333 | 0 |
| L0.B0.S0 | 3.18233e-09 | 1.2517e-07 | 0.0254241 | 0.333333 | 0 |
| L0.B0.S2 | 2.82872e-09 | 1.07288e-07 | 0.0263656 | 0.333333 | 0 |
| L0.B1.S2 | 2.81754e-09 | 1.07288e-07 | 0.0262614 | 0.333333 | 0 |
| L0.B3.S2 | 2.80501e-09 | 1.07288e-07 | 0.0261446 | 0.333333 | 0 |
| L0.B0.S1 | 2.78031e-09 | 1.07288e-07 | 0.0259144 | 0.333333 | 0 |
| L1.B2.S0 | 2.39983e-09 | 8.9407e-08 | 0.0268416 | 0.333333 | 0 |
| L1.B1.S2 | 1.97837e-09 | 7.15256e-08 | 0.0276596 | 0.333333 | 0 |
| L1.B3.S2 | 1.97102e-09 | 7.15256e-08 | 0.0275569 | 0.333333 | 0 |

## Low utility writers

| address | write_gate | utility_proxy | class_read_mass | safe_mass | expensive_mass |
|---|---|---|---|---|---|
| L0.B2.S0 | 1.60933e-07 | 4.03392e-09 | 0.0250659 | 0.333333 | 0 |
| L0.B1.S0 | 1.43051e-07 | 3.61647e-09 | 0.025281 | 0.333333 | 0 |
| L0.B2.S1 | 1.43051e-07 | 3.66318e-09 | 0.0256075 | 0.333333 | 0 |
| L0.B3.S0 | 1.43051e-07 | 3.59624e-09 | 0.0251395 | 0.333333 | 0 |
| L0.B0.S0 | 1.2517e-07 | 3.18233e-09 | 0.0254241 | 0.333333 | 0 |
| L0.B1.S1 | 1.2517e-07 | 3.2287e-09 | 0.0257946 | 0.333333 | 0 |
| L0.B2.S2 | 1.2517e-07 | 3.26531e-09 | 0.0260871 | 0.333333 | 0 |
| L0.B3.S1 | 1.2517e-07 | 3.21304e-09 | 0.0256695 | 0.333333 | 0 |
| L0.B0.S1 | 1.07288e-07 | 2.78031e-09 | 0.0259144 | 0.333333 | 0 |
| L0.B0.S2 | 1.07288e-07 | 2.82872e-09 | 0.0263656 | 0.333333 | 0 |
| L0.B1.S2 | 1.07288e-07 | 2.81754e-09 | 0.0262614 | 0.333333 | 0 |
| L0.B3.S2 | 1.07288e-07 | 2.80501e-09 | 0.0261446 | 0.333333 | 0 |
| L1.B2.S0 | 8.9407e-08 | 2.39983e-09 | 0.0268416 | 0.333333 | 0 |
| L1.B0.S0 | 7.15256e-08 | 1.94814e-09 | 0.027237 | 0.333333 | 0 |
| L1.B0.S1 | 7.15256e-08 | 1.96864e-09 | 0.0275236 | 0.333333 | 0 |

## Competing primitives: low top margin

_none_

## Expensive computed with low gradient

| address | cost | prob | grad_x_gate | computed | selected |
|---|---|---|---|---|---|
| L0.B0.S0.P0.compare | 0.75 | 0 |  | True | True |
| L0.B0.S0.P1.compare | 0.75 | 0 |  | True | True |
| L0.B0.S1.P0.compare | 0.75 | 0 |  | True | True |
| L0.B0.S1.P1.compare | 0.75 | 0 |  | True | True |
| L0.B0.S2.P0.compare | 0.75 | 0 |  | True | True |
| L0.B0.S2.P1.compare | 0.75 | 0 |  | True | True |
| L0.B1.S0.P0.compare | 0.75 | 0 |  | True | True |
| L0.B1.S0.P1.compare | 0.75 | 0 |  | True | True |
| L0.B1.S1.P0.compare | 0.75 | 0 |  | True | True |
| L0.B1.S1.P1.compare | 0.75 | 0 |  | True | True |
| L0.B1.S2.P0.compare | 0.75 | 0 |  | True | True |
| L0.B1.S2.P1.compare | 0.75 | 0 |  | True | True |
| L0.B2.S0.P0.compare | 0.75 | 0 |  | True | True |
| L0.B2.S0.P1.compare | 0.75 | 0 |  | True | True |
| L0.B2.S1.P0.compare | 0.75 | 0 |  | True | True |

## Skipped but high probability

_none_

## Suspicious

_none_

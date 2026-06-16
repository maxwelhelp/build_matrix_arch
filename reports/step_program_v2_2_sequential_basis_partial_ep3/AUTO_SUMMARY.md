# Auto summary: `step_program_v2_2_sequential_basis`

## Quality

- epochs: `6`
- best val acc: `18.60%` at epoch `1`
- last val acc: `0.00%`
- last train acc: `25.75%`
- exec stats: `{'computed_ops': 792.0, 'all_ops': 792.0, 'estimated_speedup': 1.0}`

## Top gradient addresses

| type | address | grad_x_gate | prob | cost | computed | selected |
|---|---|---|---|---|---|---|
| class_read | class.down.read.L1.B3.S0 | 9.3533e-05 | 0.0270869 |  |  |  |
| class_read | class.down.read.L0.B3.S1 | 9.32347e-05 | 0.0266466 |  |  |  |
| class_read | class.down.read.L0.B3.S0 | 9.28122e-05 | 0.0264396 |  |  |  |
| class_read | class.down.read.L0.B3.S2 | 9.27179e-05 | 0.0268212 |  |  |  |
| class_read | class.down.read.L1.B2.S0 | 9.25626e-05 | 0.0270225 |  |  |  |
| class_read | class.down.read.L0.B2.S1 | 9.20271e-05 | 0.0265812 |  |  |  |
| class_read | class.down.read.L0.B2.S2 | 9.19064e-05 | 0.0267705 |  |  |  |
| class_read | class.down.read.L1.B0.S0 | 9.16763e-05 | 0.0271001 |  |  |  |
| class_read | class.down.read.L1.B3.S1 | 9.13671e-05 | 0.0271904 |  |  |  |
| class_read | class.down.read.L0.B0.S2 | 9.12598e-05 | 0.0268335 |  |  |  |
| class_read | class.down.read.L0.B0.S1 | 9.11917e-05 | 0.0266576 |  |  |  |
| class_read | class.down.read.L0.B2.S0 | 9.11157e-05 | 0.0263575 |  |  |  |
| class_read | class.left.read.L0.B3.S1 | 9.07384e-05 | 0.0265229 |  |  |  |
| class_read | class.down.read.L1.B2.S1 | 9.07373e-05 | 0.0271415 |  |  |  |
| class_read | class.left.read.L0.B3.S0 | 9.06803e-05 | 0.0263048 |  |  |  |

## Top utility slots

| address | utility_proxy | write_gate | class_read_mass | safe_mass | expensive_mass |
|---|---|---|---|---|---|
| L2.B0.S2 | 0.0102003 | 0.372186 | 0.0274065 | 0.294385 | 0.00248194 |
| L2.B3.S2 | 0.0101763 | 0.372602 | 0.0273115 | 0.293983 | 0.00251493 |
| L2.B0.S1 | 0.0101762 | 0.369733 | 0.0275231 | 0.296599 | 0.00222069 |
| L2.B3.S1 | 0.0101556 | 0.370164 | 0.0274354 | 0.296197 | 0.00224516 |
| L2.B1.S2 | 0.0101467 | 0.373927 | 0.0271355 | 0.29351 | 0.00266969 |
| L2.B0.S0 | 0.0101463 | 0.367228 | 0.0276295 | 0.298815 | 0.00196624 |
| L2.B2.S2 | 0.0101415 | 0.372112 | 0.0272539 | 0.294778 | 0.00245452 |
| L2.B3.S0 | 0.0101214 | 0.367346 | 0.0275527 | 0.298572 | 0.00196826 |
| L2.B1.S1 | 0.0101151 | 0.37168 | 0.0272144 | 0.29572 | 0.00239822 |
| L2.B2.S1 | 0.0101084 | 0.369653 | 0.0273457 | 0.297072 | 0.00218746 |
| L2.B2.S0 | 0.0100585 | 0.36663 | 0.0274351 | 0.299367 | 0.00192115 |
| L2.B1.S0 | 0.0100562 | 0.36853 | 0.0272873 | 0.298176 | 0.00212726 |
| L1.B0.S2 | 0.0099297 | 0.360787 | 0.0275224 | 0.303433 | 0.00161874 |
| L1.B3.S2 | 0.00991508 | 0.361276 | 0.0274446 | 0.303109 | 0.00162703 |
| L1.B1.S2 | 0.00987067 | 0.362107 | 0.027259 | 0.30279 | 0.00172552 |

## Low utility writers

_none_

## Competing primitives: low top margin

_none_

## Expensive computed with low gradient

| address | cost | prob | grad_x_gate | computed | selected |
|---|---|---|---|---|---|
| L0.B0.S0.P0.matrix_mlp | 1.35 | 0.000400066 | 0 | True | True |
| L0.B0.S0.P1.matrix_mlp | 1.35 | 0.000664711 | 0 | True | True |
| L0.B0.S1.P0.matrix_mlp | 1.35 | 2.00868e-05 | 0 | True | True |
| L0.B0.S1.P1.matrix_mlp | 1.35 | 5.08428e-05 | 0 | True | True |
| L0.B0.S2.P0.matrix_mlp | 1.35 | 2.46763e-05 |  | True | True |
| L0.B0.S2.P1.matrix_mlp | 1.35 | 6.31809e-05 |  | True | True |
| L0.B1.S0.P0.matrix_mlp | 1.35 | 0.000446796 |  | True | True |
| L0.B1.S0.P1.matrix_mlp | 1.35 | 0.000730515 |  | True | True |
| L0.B1.S1.P0.matrix_mlp | 1.35 | 2.17557e-05 |  | True | True |
| L0.B1.S1.P1.matrix_mlp | 1.35 | 5.51343e-05 |  | True | True |
| L0.B1.S2.P0.matrix_mlp | 1.35 | 2.68817e-05 |  | True | True |
| L0.B1.S2.P1.matrix_mlp | 1.35 | 6.84857e-05 |  | True | True |
| L0.B2.S0.P0.matrix_mlp | 1.35 | 0.000386477 |  | True | True |
| L0.B2.S0.P1.matrix_mlp | 1.35 | 0.000641346 |  | True | True |
| L0.B2.S1.P0.matrix_mlp | 1.35 | 1.93715e-05 |  | True | True |

## Skipped but high probability

_none_

## Suspicious

_none_

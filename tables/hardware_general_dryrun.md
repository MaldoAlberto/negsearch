**DRY RUN: noise-model prediction, NOT hardware data.**

# Hardware report: FakeFez(noise model)  (exact cover (6 sets, 4 elements))

shots per circuit 2000, seeds 2, calibration {'median_cz_error': 0.0039033828523107883, 'n_cz_edges': 352, 'taken': '2026-10-06 12:23:22'}

reference: uniform P(feasible) = 0.0469, ideal P(optimum): A_q 0.25, after one iteration 1.00

## mode: plain

| circuit | CZ (per seed) | P(feasible) [95% CI] | per seed | P(optimum) [95% CI] | per seed |
|---|---|---|---|---|---|
| uniform | [0, 0] | 0.045 [0.039, 0.052] | [0.048, 0.042] | 0.015 [0.012, 0.019] | [0.015, 0.015] |
| Aq | [24, 24] | 0.912 [0.903, 0.921] | [0.887, 0.938] | 0.239 [0.226, 0.252] | [0.23, 0.248] |
| Aq+iter | [333, 328] | 0.466 [0.450, 0.481] | [0.456, 0.475] | 0.321 [0.306, 0.335] | [0.307, 0.335] |
| noise ctrl | [332, 328] | 0.631 [0.616, 0.646] | [0.626, 0.637] | 0.167 [0.156, 0.179] | [0.16, 0.174] |

* H1 (A_q feasible share > measured uniform share): p = 0  -> SUPPORTED
* H2 (one iteration raises P(optimum) over A_q alone): p = 2.2e-16  -> SUPPORTED
* H3 (iteration beats the depth-matched noise control): p = 0  -> SUPPORTED

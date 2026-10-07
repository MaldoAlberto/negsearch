# QOBLIB portfolio instances used in `experiments/portfolio_*.py`

Data from the Quantum Optimization Benchmarking Library (QOBLIB), class `06-portfolio`
(https://github.com/ZIB-AOPT/QOBLIB), licensed CC BY 4.0. Koch et al., *Quantum Optimization Benchmark
Library*, Nature Computational Science 6, 653–671 (2026), doi:10.1038/s43588-026-00991-1.

| directory | source | change |
|---|---|---|
| `po_a003_t02_orig` | `instances/po_a003_t02_orig` | none (re-written in the same format) |
| `po_a004_t04_orig-T2` | `instances/po_a004_t04_orig` | first 2 of 4 periods kept |
| `po_a005_t04_orig-T2` | `instances/po_a005_t04_orig` | first 2 of 4 periods kept |

Model parameters (reference model `bqp_u3_c10.zpl`, checked with the official checker
`06-portfolio/check`): `ub = 1`, `cash = 300000` (C = 3), `cs1 = 2` (net position in [0, 3]),
budget B = 3 (`cs2 = 2`) for a003 and B = 4 (`cs2 = 3`) for a004/a005, λ = 1e-5 / 2e-5 / 4e-5
(as in the ISQR submission to QOBLIB for these instances). Other parameters are the defaults of
`parameter_u3_c10.zpl`. Example check:

    check_portfolio po_a003_t02_orig sol.sol --ub 1 --cash 300000 --cs1 2 --cs2 2

Scaling study (`experiments/portfolio_scaling_mps.py`): full copies of `po_a004_t04_orig`, `po_a005_t04_orig`,
`po_a010_t10_orig` and `po_a010_t15_orig` (unchanged data; for a010_t10 the first 4 or 7 periods are also used).
Parameters there: `ub = 1`, C = 3 (`cs1 = 2`), B = 4 (`cs2 = 3`), λ = 2e-5 / 4e-5 for a004 / a005 and λ = 1e-5
(a QOBLIB `l1e-5` variant) for a010.

"""Cost-informed preparation with the same gates: the symmetric block preparation (Dicke on longs, Dicke on compressed shorts, controlled-swap decompression)
with the rotation angles of the Dicke blocks replaced by fitted ones, so that the state has support F and biased positive amplitudes.
Uses the exact 4-CX doubly controlled RY (the 3-CX form is exact only for the unbiased angles)."""
import numpy as np
from scipy.optimize import least_squares
from qiskit.quantum_info import Statevector
from . import dicke_opt as D


def prep_with(m, l, s, angles=None):
    """returns (circuit, default angles).  angles=None records the unbiased ones."""
    orig_cry, orig_ccry, orig_giv, orig_rel = D.cry, D.ccry, D.GIV['on'], D.REL['on']
    rec = []; it = iter(angles) if angles is not None else None
    def cry(qc, th, c, t):
        a = next(it) if it is not None else th; rec.append(th); orig_cry(qc, a, c, t)
    def ccry(qc, th, a_, b_, t):
        a = next(it) if it is not None else th; rec.append(th); orig_ccry(qc, a, a_, b_, t)
    D.cry, D.ccry, D.MODE['ccry'], D.GIV['on'], D.REL['on'] = cry, ccry, ccry, False, False
    try:
        qc = D.symmetric_block_prep_opt(m, l, s, cheap=False)
    finally:
        D.cry, D.ccry, D.GIV['on'], D.REL['on'] = orig_cry, orig_ccry, orig_giv, orig_rel
        D.MODE['ccry'] = orig_ccry
    return qc, np.array(rec)


def mean_field_g(hl, Jl, X):
    """linear field of the block QUBO with the other variables replaced by their marginals on the feasible set X (rows)"""
    mf = X.mean(0); g = np.array(hl, float).copy()
    for (i, j), w in Jl.items(): g[i] += w * mf[j]; g[j] += w * mf[i]
    return g


def tilt_weights(X, g, tau):
    w = np.exp(-tau * (X @ g) / max(np.abs(g).max(), 1e-9)); return w / w.sum()


def fit_angles(m, l, s, pidx, ptar, restarts=6, seed=0):
    """least-squares fit of the Dicke angles so that |amplitude| on the feasible strings (little-endian index list pidx) matches sqrt(ptar)"""
    amp = np.sqrt(ptar); th0 = prep_with(m, l, s)[1]; rng = np.random.default_rng(seed); best = None
    resid = lambda a: np.abs(Statevector(prep_with(m, l, s, a)[0]).data[pidx]) - amp
    for r in range(restarts):
        o = least_squares(resid, th0 + (0 if r == 0 else rng.normal(0, 0.3, len(th0))), max_nfev=120)
        if best is None or o.cost < best.cost: best = o
        if best.cost < 1e-10: break
    return best.x, float(best.cost)

"""Resources of the four methods on the residual quantum core (after the exact classical reduction), for the portfolio families of core_map.py.
Same core, same dense objective (an UPPER bound: f(f-1)/2 ZZ terms among the f free qubits, each 2 CZ), so the objective layer is common to all methods and the
discriminating quantities are (a) overhead beyond the objective, (b) qubits, (c) what the method guarantees.
  ours-QAOA     p=1 layer: symmetric preparation of every non-trivial block + objective + pair-exchange mixers (exact, 100 % feasible by construction)
  ours-Grover   per iteration: A^dag, reflection about |0>, A (the objective-threshold oracle is common and excluded); iterations k = (pi/4)/sqrt(a)
  Fourier-LCU   single-basis-circuit variant: constraint layer = single-qubit Rz (no 2-qubit gates), objective + coherent exclusivity pairs (already in the dense objective)
  unbalanced    penalty terms land on pairs that the dense objective already couples -> no extra 2-qubit gates
CZ are counted after transpilation to {cz, rz, sx, x} WITHOUT routing (routing on a heavy-hex device multiplies them).  Feasible fraction of the baselines is NOT extrapolated:
only the uniform reference |F|/2^f is given.  Writes results/four_methods.json."""
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT)
import sys; sys.path.insert(0, ROOT)
import json, math, warnings; warnings.filterwarnings('ignore')
import numpy as np
from qiskit import QuantumCircuit, transpile
from negsearch.symlower import symmetric_block_prep
from negsearch.hw_large import pair_swap_mixer
from experiments.core_map import FAMILIES, period_options, nF, block_free

_cache = {}


def cz(qc): return transpile(qc, basis_gates=['cz', 'rz', 'sx', 'x'], optimization_level=1).count_ops().get('cz', 0)


def prep_cz(m, l, s):
    k = ('p', m, l, s)
    if k not in _cache: _cache[k] = cz(symmetric_block_prep(m, l, s))
    return _cache[k]


def mixer_cz():
    if 'm4' not in _cache:
        q = QuantumCircuit(4); pair_swap_mixer(q, 0, 1, 2, 3, 0.7); _cache['m4'] = cz(q)
        q = QuantumCircuit(2); q.rxx(0.7, 0, 1); q.ryy(0.7, 0, 1); q.rzz(0.7, 0, 1); _cache['m2'] = cz(q)
    return _cache['m4'], _cache['m2']


def sample(fam, m, T, nsec=2, ntup=30, seed=0):
    p = FAMILIES[fam]; rng = np.random.default_rng(seed); opts = period_options(m, nsec, p)
    w = np.array([o[2] for o in opts], float); w /= w.sum(); Q = int(round(0.6 * T * p['Bmax'])); out = []
    for _ in range(ntup):
        rem = Q; pick = []
        for t in range(T):
            ok = [i for i, o in enumerate(opts) if o[1] <= rem]
            if not ok: break
            o = opts[ok[rng.choice(len(ok), p=w[ok] / w[ok].sum())]]; pick.append(o); rem -= o[1]
        if len(pick) == T: out.append([c for o in pick for c in o[0]])
    return out


def evaluate(fam, m, T, a_levels=(1e-6, 1e-9)):
    m4, m2 = mixer_cz(); rows = []
    for blocks in sample(fam, m, T):
        f = sum(block_free(m, l, s) for l, s in blocks); logF = sum(math.log2(nF(m, l, s)) for l, s in blocks)
        prep = sum(prep_cz(m, l, s) for l, s in blocks if block_free(m, l, s))
        mix = sum((m - 1) * (m2 if (l == 0 or s == 0) else m4) for l, s in blocks if block_free(m, l, s))
        obj = f * (f - 1)            # f(f-1)/2 ZZ terms x 2 CZ
        refl = 12 * (f - 1)          # reflection about |0>: 2(f-1) Toffolis x 6 CZ
        rows.append((f, logF, prep, mix, obj, refl))
    r = np.array(rows).mean(0); f, logF, prep, mix, obj, refl = r
    it = 2 * prep + refl
    res = dict(family=fam, m=m, T=T, n=2 * m * 2 * T, free=f, log2F=logF, objective_cz=obj,
               qaoa_ours_cz=prep + mix + obj, qaoa_overhead_cz=prep + mix, qaoa_overhead_ratio=(prep + mix) / obj if obj else None,
               lcu_cz=obj, unbalanced_cz=obj, grover_iter_cz=it, uniform_feasible_log2=logF - f,
               grover_std_over_ours_iter_log2=(f - logF) / 2)
    for a in a_levels:
        k = (math.pi / 4) / math.sqrt(a); res[f'grover_total_cz_a{a:g}'] = k * it; res[f'grover_iters_a{a:g}'] = k
    return res


if __name__ == '__main__':
    out = []
    print(f"{'family':7s} {'n':>4s} {'free':>6s} {'log2F':>6s} | {'objective':>10s} {'ours QAOA':>10s} {'overhead':>9s} {'ovh/obj':>8s} | LCU,unbal | Grover iter CZ  (a=1e-6 total)  | std/ours iters 2^")
    for fam in FAMILIES:
        for m, T in [(6, 2), (8, 4), (12, 4), (16, 8)]:
            r = evaluate(fam, m, T); out.append(r)
            print(f"{fam:7s} {r['n']:4d} {r['free']:6.0f} {r['log2F']:6.1f} | {r['objective_cz']:10.0f} {r['qaoa_ours_cz']:10.0f} {r['qaoa_overhead_cz']:9.0f} {r['qaoa_overhead_ratio']:8.2f} | "
                  f"{r['lcu_cz']:9.0f} | {r['grover_iter_cz']:12.0f} ({r['grover_total_cz_a1e-06']:.1e}) | {r['grover_std_over_ours_iter_log2']:.1f}", flush=True)
    json.dump(out, open('results/four_methods.json', 'w'), indent=1)

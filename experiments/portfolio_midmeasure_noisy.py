"""ibm_fez noise model for 'sector + Dicke' on po_a003_t02_orig: each shot draws the count sector (L_t, S_t)
classically from the A_q weights (equivalently a mid-circuit-measured coin), the chip prepares Dicke states and
runs the XY hybrid layers.  Shots are split multinomially over the sector circuits.
Usage: python experiments/portfolio_midmeasure_noisy.py [shots]"""
import os as _os, sys as _sys
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..'))
_os.chdir(ROOT); _sys.path.insert(0, ROOT)
import json, time, numpy as np
from collections import Counter
from qiskit import transpile
from qiskit_aer import AerSimulator
from negsearch.portfolio_qaoa import Sim
from experiments.portfolio_qaoa import load
from experiments.portfolio_midmeasure import sector_circuit, sectors, BK
from experiments.portfolio_hardware_opt import stats

sig = lambda z: 1 / (1 + np.exp(-z))
if __name__ == '__main__':
    SHOTS = int(_sys.argv[1]) if len(_sys.argv) > 1 else 4000
    key = 'a003_t02'; P = load(key, order='shorts_first'); P.xy_ring = False; S = Sim(P); sc = 1 / np.abs(S.f).max()
    R = json.load(open('results/portfolio_midmeasure.json'))[key]
    L, Sc = P.counts(S.bits[:, :P.nq]); out = {}
    sim = AerSimulator.from_backend(BK); rng = np.random.default_rng(1)
    for p in (1, 2):
        m = R[f'p{p} | sector + Dicke']; x = np.array(m['x']); a = P.prep_state(sig(x[-1]))
        w = {}
        for i in np.nonzero(S.feas)[0]:
            c = tuple((int(L[i, t]), int(Sc[i, t])) for t in range(P.T)); w[c] = w.get(c, 0) + abs(a[i]) ** 2
        secs = list(w); shots = rng.multinomial(SHOTS, [w[s] for s in secs])
        total = Counter(); cz_w = 0.0; t0 = time.time()
        for s, n in zip(secs, shots):
            if n == 0: continue
            qc = sector_circuit(P, s, x[:p], x[p:2 * p], P.h * sc, 'dicke'); qc.measure_all()
            t = transpile(qc, BK, optimization_level=3, seed_transpiler=1)
            cz_w += n / SHOTS * t.count_ops().get('cz', 0)
            total.update(sim.run(t, shots=int(n), seed_simulator=int(rng.integers(1 << 30))).result().get_counts())
        out[f'sector + Dicke p={p}'] = dict(mean_cz_per_shot=cz_w, ideal={k: m[k] for k in ('p_feas', 'quality', 'p_opt')},
                                            noisy=stats(P, S, dict(total)), seconds=time.time() - t0)
        print(json.dumps(out[f'sector + Dicke p={p}']), flush=True)
    json.dump(out, open('results/portfolio_midmeasure_noisy.json', 'w'), indent=1)

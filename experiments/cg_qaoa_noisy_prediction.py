"""Noisy (FakeFez) prediction for the hardware-feasible circuits on karate-sub18: A_q sampling and penalty QAOA p=1."""
import os as _os, sys as _sys
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT); _sys.path.insert(0, ROOT)
import json, sys
from negsearch.qaoa_mis import read_gph, MIS
from negsearch.qaoa_circuits import prep_circuit, penalty_qaoa
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeFez
which, shots, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ORDER = sys.argv[4] if len(sys.argv) > 4 else 'lowdeg'
def degeneracy_order(N, n):
    rem = set(range(n)); seq = []
    while rem:
        v = min(rem, key=lambda u: (len(N[u] & rem), u)); seq.append(v); rem.remove(v)
    return seq[::-1]
n, E = read_gph('instances/qoblib/karate-sub18.gph'); g = MIS(n, E)
order = degeneracy_order(g.N, n) if ORDER == 'degeneracy' else sorted(range(n), key=lambda v: (len(g.N[v]), v))
R = json.load(open('results/cg_qaoa_sub.json'))['karate-sub18']['p1']
lam = R['penalty']['lam']; x = R['params'][f'pen{lam}']; q = 0.95
qc = penalty_qaoa(n, E, lam, [x[0]], [x[1]]) if which == 'penalty' else prep_circuit(n, g.N, order, q)
if which != 'penalty': qc.measure_all()
fake = FakeFez(); t = transpile(qc, fake, optimization_level=3, seed_transpiler=1)
cnt = AerSimulator.from_backend(fake).run(t, shots=shots, seed_simulator=seed).result().get_counts()
fn = 'results/cg_qaoa_noisy_karate.json'
d = json.load(open(fn)) if _os.path.exists(fn) else {}
which = which + ('_degeneracy' if ORDER == 'degeneracy' else '')
d.setdefault(which, dict(q=q if which != 'penalty' else None, lam=lam if which == 'penalty' else None, params=x if which == 'penalty' else None,
                         cz=t.count_ops().get('cz', 0), counts={}))
for k, v in cnt.items(): d[which]['counts'][k] = d[which]['counts'].get(k, 0) + v
json.dump(d, open(fn, 'w'), indent=1); print(which, sum(d[which]['counts'].values()), 'shots saved')

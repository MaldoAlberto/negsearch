"""ibm_fez noise model on po_a003_t02_orig for the cheapest hybrid (shorts-first lean A_q, XY on a path, linear phase
only), with the same transpilation (best of 5 seeds) as experiments/portfolio_hardware_opt.py.
Usage: python experiments/portfolio_noisy_lean.py [shots]"""
import os as _os, sys as _sys
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..'))
_os.chdir(ROOT); _sys.path.insert(0, ROOT)
import json, time, numpy as np
from qiskit_aer import AerSimulator
from negsearch.portfolio_circuits import qaoa_circuit, prep_circuit_lean
from negsearch.portfolio_qaoa import Sim
from experiments.portfolio_qaoa import load
from experiments.portfolio_hardware_opt import transp, stats, BK

sig = lambda z: 1 / (1 + np.exp(-z))
if __name__ == '__main__':
    SHOTS = int(_sys.argv[1]) if len(_sys.argv) > 1 else 4000
    key = 'a003_t02'; P = load(key, order='shorts_first'); P.xy_ring = False; S = Sim(P); sc = 1 / np.abs(S.f).max()
    pr = json.load(open('results/portfolio_prune.json'))[key]['path_J1.01']
    out = {}
    for p in (1, 2):
        x = np.array(pr[f'p{p}']['x'])
        qc = qaoa_circuit(P, 'cg_xy', (x[:p], x[p:2 * p]), q=sig(x[-1]), h=P.h * sc, J={}, counter='lean')
        regs = prep_circuit_lean(P, 0.5).qregs; wl, ws = regs[1].size, regs[2].size   # counter widths (period 0)
        qc.measure_all(); t = transp(qc); t0 = time.time()
        cnt = AerSimulator.from_backend(BK).run(t, shots=SHOTS, seed_simulator=1).result().get_counts()
        out[f'hybrid p={p} (lean, path XY, linear phase)'] = dict(
            qubits=qc.num_qubits, cz=int(t.count_ops().get('cz', 0)), depth=int(t.depth()),
            ideal={k: v for k, v in pr[f'p{p}'].items() if k in ('p_feas', 'quality', 'p_opt')},
            noisy=stats(P, S, cnt, wl, ws), seconds=time.time() - t0)
        print(json.dumps(out), flush=True)
        json.dump(out, open('results/portfolio_noisy_lean.json', 'w'), indent=1)

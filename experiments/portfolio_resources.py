"""Hardware cost of one QAOA layer (and of the preparation A_q) for each encoding, transpiled for ibm_fez
(FakeFez, optimization_level=3, best of 3 seeds).  Usage: python experiments/portfolio_resources.py"""
import os as _os, sys as _sys
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..'))
_os.chdir(ROOT); _sys.path.insert(0, ROOT)
import json, numpy as np
from qiskit import transpile, QuantumCircuit
from qiskit_ibm_runtime.fake_provider import FakeFez
from negsearch.portfolio_circuits import prep_circuit, qaoa_circuit
from experiments.portfolio_qaoa import INST, load

BK = FakeFez()


def cost(qc, seeds=(1, 2, 3)):
    best = None
    for s in seeds:
        t = transpile(qc, BK, optimization_level=3, seed_transpiler=s)
        two = sum(v for k, v in t.count_ops().items() if k in ('cz', 'ecr', 'cx'))
        d2 = t.depth(lambda ins: ins.operation.num_qubits == 2)
        r = dict(qubits=qc.num_qubits, twoq=int(two), depth=int(t.depth()), twoq_depth=int(d2))
        if best is None or r['twoq'] < best['twoq']: best = r
    return best


if __name__ == '__main__':
    out = {}
    for key in INST:
        P = load(key); a = 2.0; D = float(np.abs(P.h).max())
        hu, Ju, cu = P.unbalanced_poly(a * D, a * D)
        hs, Js, cs_, ntot = P.slack_qubo(a * D)
        g, b = [0.3], [0.4]
        R = dict(qubits_x=P.nq, qubits_slack_qubo=ntot)
        R['A_q (preparation only)'] = cost(prep_circuit(P, 0.4))
        R['penalty_slack'] = cost(qaoa_circuit(P, 'penalty_slack', (g, b), h=hs, J=Js, nq_total=ntot))
        R['penalty_unbalanced'] = cost(qaoa_circuit(P, 'penalty_unbal', (g, b), h=hu, J=Ju))
        R['A_init_Xmixer'] = cost(qaoa_circuit(P, 'warm_x', (g, b), q=0.4, h=hu, J=Ju))
        R['A_init_XYmixer'] = cost(qaoa_circuit(P, 'cg_xy', (g, b), q=0.4, h=P.h, J=P.J))
        R['A_init_Grover_mixer'] = cost(qaoa_circuit(P, 'cg_grover', (g, b), q=0.4, h=P.h, J=P.J))
        # per additional layer = cost(p=2) - cost(p=1)
        for kind, nm, hh, JJ, nt in (('penalty_slack', 'penalty_slack', hs, Js, ntot), ('penalty_unbal', 'penalty_unbalanced', hu, Ju, None),
                                     ('warm_x', 'A_init_Xmixer', hu, Ju, None), ('cg_xy', 'A_init_XYmixer', P.h, P.J, None),
                                     ('cg_grover', 'A_init_Grover_mixer', P.h, P.J, None)):
            c2 = cost(qaoa_circuit(P, kind, ([0.3, 0.5], [0.4, 0.2]), q=0.4, h=hh, J=JJ, nq_total=nt))
            R[nm]['p2_twoq'] = c2['twoq']; R[nm]['p2_depth'] = c2['depth']
        out[key] = R
        print(key, json.dumps(R), flush=True)
        json.dump(out, open('results/portfolio_resources.json', 'w'), indent=1)

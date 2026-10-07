"""Compile-only (no execution) depth/CZ/duration/ESP validation on the ibm_fez topology (FakeFez) of
 (a) A_q alone, (b) A_q + one Grover iteration (oracle = marked optimum strings, V-chain MCZ),
 (c) A_q + one Grover-mixer QAOA layer (p=1, cost as Z-polynomial)
over several constraint families.  Writes results/hardware_depth_multi.json and prints a go/no-go table."""
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT)
import sys; sys.path.insert(0, ROOT)
import json, math
import numpy as np
from qiskit import transpile
from qiskit_ibm_runtime.fake_provider import FakeFez
from negsearch.automaton import build_automaton, compile_Aq, mu_q
from negsearch.depth_study import walsh, grover_circuit, qaoa_circuit
from negsearch import families as fam
import warnings; warnings.filterwarnings('ignore')
tgt = None


def legal(t):
    edges = set(map(tuple, bk.coupling_map.get_edges()))
    for inst in t.data:
        nm = inst.operation.name
        if nm in ('barrier', 'measure', 'delay'):
            continue
        qs = tuple(t.find_bit(q).index for q in inst.qubits)
        if not bk.target.instruction_supported(nm, qs):
            return False
    return True


def esp_and_time(t):
    lg = 0.0; used = set()
    for inst in t.data:
        nm = inst.operation.name
        if nm in ('barrier', 'delay'):
            continue
        qs = tuple(t.find_bit(q).index for q in inst.qubits)
        used.update(qs)
        try:
            e = bk.target[nm][qs].error
        except Exception:
            e = None
        if e:
            lg += math.log1p(-min(e, 0.999))
    return math.exp(lg), sorted(used)

bk = FakeFez(); NS = 5
cases = [('exact cover 6/4 (hw)', fam.hardware_instance()), ('cardinality 3 of 6', fam.cardinality_qp(6, 3, 0)),
         ('knapsack n=6', fam.knapsack(6, 11)), ('MIS n=6', fam.mis(6, 0.4, 22)), ('set packing n=6', fam.set_packing(6, 5, 33)),
         ('3-colouring 3 vert', fam.colouring(3, 55, 0.7)), ('assignment 3x3', fam.assignment_qap(3, 6))]


def measure(qc):
    R = []
    for s in range(NS):
        t = transpile(qc, bk, optimization_level=3, seed_transpiler=s + 1)
        sch = transpile(t, bk, optimization_level=0, scheduling_method='alap')
        esp, used = esp_and_time(t)
        t2 = float(np.median([bk.qubit_properties(q).t2 for q in used]))
        R.append(dict(cz=t.count_ops().get('cz', 0), depth=t.depth(), qubits=len(used), legal=legal(t),
                      dur_us=sch.duration * bk.dt * 1e6, t2_us=t2 * 1e6, esp=esp))
    med = lambda k: float(np.median([r[k] for r in R]))
    return dict(cz=med('cz'), depth=med('depth'), qubits=med('qubits'), legal=all(r['legal'] for r in R),
                dur_us=med('dur_us'), t2_us=med('t2_us'), esp=med('esp'))


out = []
print(f"{'case':22s} {'circuit':10s} {'q':>3s} {'CZ':>6s} {'depth':>6s} {'dur(us)':>8s} {'dur/T2':>7s} {'ESP':>7s} legal")
for name, P in cases:
    n = P.n; A = build_automaton(P.feas, n); G = np.flatnonzero(P.good(0.0))
    c = (P.cost - P.opt) / (P.worst - P.opt); co = walsh(c, n)
    Aq, _ = compile_Aq(A, 0.5)
    circs = {'A_q': Aq, 'A_q+Grover k=1': grover_circuit(A, list(G), 1), 'A_q+GM-QAOA p=1': qaoa_circuit(A, co, [0.7, 1.3])}
    for cname, qc in circs.items():
        m = measure(qc); m.update(case=name, circuit=cname, n=n, width=A.w, terms=len(co)); out.append(m)
        print(f"{name:22s} {cname:10s} {m['qubits']:3.0f} {m['cz']:6.0f} {m['depth']:6.0f} {m['dur_us']:8.1f} {m['dur_us']/m['t2_us']:7.2f} {m['esp']:7.3f} {m['legal']}", flush=True)
json.dump(out, open('results/hardware_depth_multi.json', 'w'), indent=1)

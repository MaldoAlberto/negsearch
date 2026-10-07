"""ZX-calculus on the hardware-friendly hybrid (portfolio, QOBLIB): does PyZX reduce CZ / depth on ibm_fez further?
Pipelines (each followed by the same FakeFez transpilation, optimisation level 3, best of 5 seeds):
  qiskit    : the circuit as built
  zx_full   : full_reduce + circuit extraction (graph-theoretic simplification, re-synthesised CNOTs)
  zx_tele   : teleport_reduce (phase teleportation / phase folding, keeps the CNOT structure)
Exactness is checked by comparing statevectors (up to global phase).  Usage: python experiments/portfolio_zx.py"""
import os as _os, sys as _sys
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..'))
_os.chdir(ROOT); _sys.path.insert(0, ROOT)
import json, time, numpy as np, pyzx as zx
from qiskit import QuantumCircuit, transpile, qasm2
from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime.fake_provider import FakeFez
from negsearch.portfolio_circuits import qaoa_circuit, prep_circuit_lean
from experiments.portfolio_qaoa import INST, load

BK = FakeFez(); BASIS = ['cx', 'rz', 'h', 'x']
sig = lambda z: 1 / (1 + np.exp(-z))


def hw(qc):
    best = None
    for s in (1, 2, 3, 4, 5):
        t = transpile(qc, BK, optimization_level=3, seed_transpiler=s)
        if best is None or t.count_ops().get('cz', 0) < best.count_ops().get('cz', 0): best = t
    return dict(cz=int(best.count_ops().get('cz', 0)), depth=int(best.depth()),
                cz_depth=int(best.depth(lambda i: i.operation.num_qubits == 2)))


def to_zx(qc):
    b = transpile(qc, basis_gates=BASIS, optimization_level=1)
    return zx.Circuit.from_qasm(qasm2.dumps(b))


def from_zx(c, n):
    q = QuantumCircuit.from_qasm_str(c.to_basic_gates().to_qasm())
    out = QuantumCircuit(n); out.compose(q, range(q.num_qubits), inplace=True); return out


def zx_full(qc):
    g = to_zx(qc).to_graph(); zx.full_reduce(g)
    c = zx.extract_circuit(g.copy()); c = zx.basic_optimization(c.to_basic_gates())
    return from_zx(c, qc.num_qubits)


def zx_tele(qc):
    g = to_zx(qc).to_graph(); zx.teleport_reduce(g)
    c = zx.Circuit.from_graph(g); c = zx.basic_optimization(c.to_basic_gates())
    return from_zx(c, qc.num_qubits)


def same(a, b):
    if a.num_qubits > 24: return None
    va, vb = Statevector(a).data, Statevector(b).data
    return bool(abs(abs(np.vdot(va, vb)) - 1) < 1e-6)


if __name__ == '__main__':
    out = {}
    pr = json.load(open('results/portfolio_prune.json'))
    for key in INST:
        P = load(key, order='shorts_first'); P.xy_ring = False
        x = pr[key]['path_J1.01']['p1']['x']; q = sig(x[-1]); sc = 1e-3
        comps = {
            'A_q (lean)': prep_circuit_lean(P, q),
            'hybrid p=1, linear phase': qaoa_circuit(P, 'cg_xy', ([x[0]], [x[1]]), q=q, h=P.h * sc, J={}, counter='lean'),
            'hybrid p=2, linear phase': qaoa_circuit(P, 'cg_xy', ([x[0], .7 * x[0]], [x[1], .8 * x[1]]), q=q, h=P.h * sc, J={}, counter='lean'),
            'hybrid p=1, full phase (all RZZ)': qaoa_circuit(P, 'cg_xy', ([x[0]], [x[1]]), q=q, h=P.h * sc,
                                                             J={k: v * sc for k, v in P.J.items() if v}, counter='lean'),
        }
        rec = {}
        for name, qc in comps.items():
            r = {'qiskit': hw(qc)}
            for tag, f in (('zx_full', zx_full), ('zx_tele', zx_tele)):
                t0 = time.time()
                try:
                    z = f(qc); r[tag] = dict(hw(z), exact=same(qc, z), seconds=round(time.time() - t0, 1))
                except Exception as e:
                    r[tag] = dict(error=repr(e)[:200])
            rec[name] = r
            print(key, name, json.dumps(r), flush=True)
        out[key] = rec
        json.dump(out, open('results/portfolio_zx.json', 'w'), indent=1)

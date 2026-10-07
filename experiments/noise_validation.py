"""Validate the global-depolarising / ESP model against a gate-level noisy simulation (Aer, noise model of FakeFez: gate errors + thermal relaxation + readout) on the
n=64 case of case_four.py (core of 16 free qubits).  For each circuit: ideal feasible fraction (statevector), model prediction  ESP*ideal + (1-ESP)|F|/2^f,
and the feasible fraction / quality measured in the noisy simulation (shots), plus quality after post-selecting on feasibility.  Writes results/noise_validation.json."""
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT)
import sys; sys.path.insert(0, ROOT)
import json, math, os, warnings; warnings.filterwarnings('ignore')
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
import experiments.case_four as C
import experiments.prune_circuit as P
from negsearch.hw_large import ising_from_qubo
from negsearch.depth_study import phase_poly
from negsearch.fez_check import bk, esp_and_time

SHOTS = int(os.environ.get('NV_SHOTS', 600))
cp = C.core_problem(); free, col, Fm, bl = cp['free'], cp['col'], cp['F'], cp['bl']; f = len(free); nF = len(Fm); n = C.n
codeF = set(int(c) for c in (Fm * (1 << np.arange(f))).sum(1)); cF = C.cost_of(Fm, cp['h1'], cp['J1'], col); cmin, cmax = cF.min(), cF.max()
cost_code = {int(c): float(v) for c, v in zip((Fm * (1 << np.arange(f))).sum(1), cF)}
ang = json.load(open('results/case_four.json'))['ours_qaoa']['angles']
groups = P.groups_of(bl, cp['forced'])


def with_measure(qc):
    q2 = QuantumCircuit(n, f); q2.compose(qc, inplace=True)
    for v in free: q2.measure(v, col[v])
    return q2


def ideal_probs(qc):
    gc, used = C.compact(qc); loc = {v: k for k, v in enumerate(used)}; sv = Statevector(gc).probabilities(); pr = {}
    for s_i, pv in enumerate(sv):
        if pv < 1e-14: continue
        code = 0
        for v in free:
            if (s_i >> loc[v]) & 1: code |= 1 << col[v]
        pr[code] = pr.get(code, 0) + pv
    return pr


def stats(dist):
    tot = sum(dist.values()); fe = sum(v for c, v in dist.items() if c in codeF) / tot
    q = sum(v * (1 - (cost_code[c] - cmin) / (cmax - cmin)) for c, v in dist.items() if c in codeF) / max(1e-12, sum(v for c, v in dist.items() if c in codeF))
    return fe, q


def baseline(rz):
    h, J = cp['h1'].copy(), dict(cp['J1'])
    qc = QuantumCircuit(n)
    for v in free: qc.h(v)
    phase_poly(qc, ising_from_qubo(h, J), 0.3)
    if rz:
        for vs, tgt in groups:
            for v in vs: qc.rz(0.4, v)
    for v in free: qc.rx(0.8, v)
    return qc


circuits = {'ours_qaoa': P.circuit(C.A, C.T, C.S, bl, cp['h1'], cp['J1'], [ang[0]], [ang[1]], True), 'lcu_like': baseline(True)}
sim = AerSimulator.from_backend(bk); out = {}
for name, qc in circuits.items():
    ip = ideal_probs(qc); ife, iq = stats(ip)
    qm = with_measure(qc); best = None
    for s in range(4):
        t = transpile(qm, bk, optimization_level=3, seed_transpiler=s + 1); e = esp_and_time(t)[0]
        if best is None or e > best[0]: best = (e, t)
    esp, t = best; used = len({t.find_bit(q).index for i in t.data for q in i.qubits})
    res = sim.run(t, shots=SHOTS).result().get_counts(); dist = {int(k.replace(' ', ''), 2): v for k, v in res.items()}
    nfe, nq = stats(dist)
    model = esp * ife + (1 - esp) * nF / 2 ** f
    out[name] = dict(esp=esp, cz=t.count_ops().get('cz', 0), depth=t.depth(), qubits_used=used, ideal_feasible=ife, ideal_quality=iq, model_feasible=model, sim_feasible=nfe,
                     sim_feasible_se=math.sqrt(nfe * (1 - nfe) / SHOTS), sim_quality_postselected=nq, shots=SHOTS)
    print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[name].items()}, flush=True)
json.dump(out, open('results/noise_validation.json', 'w'), indent=1)

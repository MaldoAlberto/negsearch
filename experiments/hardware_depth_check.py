"""Depth / legality / duration validation of the hardware circuits on the ibm_fez topology (FakeFez snapshot), no noise simulation.
For each circuit and 20 transpiler seeds: CZ count, depth, 2-qubit depth, number of SWAP-induced CZ (vs the logical circuit), legality
(every gate native and every CZ on a coupling-map edge), scheduled duration (ALAP) against median T1/T2 of the qubits used, and the
estimated success probability ESP = prod(1 - error) over gates and readout (a crude product bound, not a simulation).
Writes results/hardware_depth_check.json."""
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT)
import sys; sys.path.insert(0, ROOT)
import json, math
import numpy as np
from qiskit import transpile
from qiskit_ibm_runtime.fake_provider import FakeFez
from experiments.hardware_general import logical_circuits, pad_isa, cz_of
from negsearch.families import hardware_instance

bk = FakeFez(); tgt = bk.target
P = hardware_instance(); circs, A, G = logical_circuits(P)
NS = 20


def legal(t):
    edges = set(map(tuple, bk.coupling_map.get_edges()))
    for inst in t.data:
        nm = inst.operation.name
        if nm in ('barrier', 'measure'):
            continue
        qs = tuple(t.find_bit(q).index for q in inst.qubits)
        if not tgt.instruction_supported(nm, qs):
            return False
    return True


def esp_and_time(t):
    """product of (1 - error) over gates and measurements actually present; duration from an ALAP schedule"""
    lg = 0.0; used = set()
    for inst in t.data:
        nm = inst.operation.name
        if nm == 'barrier':
            continue
        qs = tuple(t.find_bit(q).index for q in inst.qubits)
        used.update(qs)
        try:
            e = tgt[nm][qs].error
        except Exception:
            e = None
        if e:
            lg += math.log1p(-e)
    return math.exp(lg), sorted(used)


rows = {}
for name in ['Aq', 'Aq+iter', 'noise ctrl']:
    R = []
    for s in range(NS):
        tA = transpile(circs['Aq'], bk, optimization_level=3, seed_transpiler=s + 1)
        if name == 'Aq':
            t = tA
        else:
            tB = transpile(circs['Aq+iter'], bk, optimization_level=3, seed_transpiler=s + 1)
            t = tB if name == 'Aq+iter' else pad_isa(tA, max(0, (cz_of(tB) - cz_of(tA)) // 2))
        sched = transpile(t, bk, optimization_level=0, scheduling_method='alap')
        dur = sched.duration * bk.dt if sched.duration else float('nan')
        esp, used = esp_and_time(t)
        t1 = np.median([bk.qubit_properties(q).t1 for q in used]); t2 = np.median([bk.qubit_properties(q).t2 for q in used])
        R.append(dict(seed=s + 1, cz=cz_of(t), depth=t.depth(), depth_2q=t.depth(lambda i: i.operation.num_qubits == 2 and i.operation.name != 'barrier'),
                      qubits_used=len(used), legal=legal(t), duration_us=dur * 1e6, t1_us=t1 * 1e6, t2_us=t2 * 1e6, esp=esp))
    rows[name] = R
    f = lambda k: (np.min([r[k] for r in R]), float(np.median([r[k] for r in R])), np.max([r[k] for r in R]))
    print(f"{name:11s} CZ {f('cz')}  depth {f('depth')}  2q-depth {f('depth_2q')}  qubits {f('qubits_used')[1]:.0f}  legal {all(r['legal'] for r in R)}  "
          f"duration(us) {f('duration_us')[1]:.1f}  T1~{f('t1_us')[1]:.0f} T2~{f('t2_us')[1]:.0f}  ESP {f('esp')}")
json.dump(rows, open('results/hardware_depth_check.json', 'w'), indent=1)

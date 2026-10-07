"""Compile-only checks on the ibm_fez topology (FakeFez): CZ, depth, ALAP duration, ESP, legality."""
import math
import numpy as np
from qiskit import transpile
from qiskit_ibm_runtime.fake_provider import FakeFez
bk = FakeFez(); NS = 5
import warnings; warnings.filterwarnings('ignore')


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




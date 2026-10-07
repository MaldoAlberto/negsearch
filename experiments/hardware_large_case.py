"""Executable hardware package for the multi-constraint portfolio case (A assets, T periods, sectors, global trading budget), conditioned on a
classically drawn interface.  Default instance: A=6, T=2, sectors [3,3] -> n=24 decision variables, 2620 feasible strings, 21 constraints.

  python experiments/hardware_large_case.py --dry-run              # build, validate, compile on the ibm_fez topology (FakeFez), write circuits
  python experiments/hardware_large_case.py --run --apikey apikey_personal.json --backend ibm_fez --shots 4000   # execute (needs your key)

Circuits written to hardware_circuits/ (QPY and OpenQASM 3) with a metadata JSON.  Controls per interface tuple: 'prep' (A only, p=0),
'qaoa1' (p=1), and a uniform-superposition circuit (Hadamards on the 24 variables).  The report gives the fraction of feasible samples,
the mean objective and P(optimum | interface) with Wilson intervals."""
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT)
import sys; sys.path.insert(0, ROOT)
import argparse, itertools, json, math, warnings; warnings.filterwarnings('ignore')
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile, qpy
from qiskit.qasm3 import dumps as qasm3_dumps
from negsearch.hw_large import (interface_tuples, block_strings, block_choice, qubo_cost, ising_from_qubo, build_hw_circuit, x_index)

A, T, SECT, SEED = 6, 2, [3, 3], 5
NV = 2 * A * T


def tuple_support(iface):
    """list of (x bit tuple over all NV variables in index order (t,i,d), probability, per-block amplitudes) for one interface tuple"""
    sec_off = np.cumsum([0] + SECT); blocks = []
    for t in range(T):
        for si, size in enumerate(SECT):
            l, s = iface[t][si]
            B = block_choice(size, l, s)[1]
            blocks.append((t, si, block_strings(B)))
    return blocks


def full_strings(blocks):
    """enumerate product of block strings -> array (m, NV) of bits and array (m, nblocks) of block indices"""
    sec_off = np.cumsum([0] + SECT)
    sizes = [len(b[2]) for b in blocks]
    combos = np.array(list(itertools.product(*[range(s) for s in sizes])))
    X = np.zeros((len(combos), NV), int)
    for bi, (t, si, strs) in enumerate(blocks):
        arr = np.array([s for s, _ in strs])
        for j in range(SECT[si]):
            for d in (0, 1):
                X[:, x_index(A, t, sec_off[si] + j, d)] = arr[combos[:, bi], 2 * j + d]
    return X, combos


def cost_of(X, h, J):
    c = X @ h
    for (i, j), v in J.items():
        c = c + v * X[:, i] * X[:, j]
    return c


def tensor_qaoa(blocks, combos, c, params):
    """exact QAOA on the product space of the blocks: phase on the full cost, then a Grover mixer about each block's own state"""
    p = len(params) // 2
    shape = [len(b[2]) for b in blocks]
    amps = [np.sqrt(np.array([m for _, m in b[2]])) for b in blocks]
    psi = amps[0]
    for a in amps[1:]:
        psi = np.multiply.outer(psi, a)
    psi = psi.astype(complex)
    cT = np.zeros(shape); cT[tuple(combos.T)] = c
    for l in range(p):
        psi = psi * np.exp(-1j * params[l] * cT)
        for ax, a in enumerate(amps):
            ov = np.tensordot(a, psi, axes=([0], [ax]))                   # <u_b|psi> over axis ax
            psi = psi - (1 - np.exp(-1j * params[p + l])) * np.moveaxis(np.multiply.outer(a, ov), 0, ax)
    return np.abs(psi) ** 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true'); ap.add_argument('--run', action='store_true')
    ap.add_argument('--apikey', default='apikey_personal.json'); ap.add_argument('--backend', default='ibm_fez')
    ap.add_argument('--shots', type=int, default=4000); ap.add_argument('--seeds', type=int, default=20)
    ap.add_argument('--top', type=int, default=3, help='number of interface tuples (most probable) plus the one containing the optimum')
    ap.add_argument('--assets', type=int, default=6); ap.add_argument('--periods', type=int, default=2)
    ap.add_argument('--sectors', default='3,3'); ap.add_argument('--tag', default='')
    ap.add_argument('--force', action='store_true', help='submit circuits below --min-esp as well')
    ap.add_argument('--skip-qaoa', action='store_true', help='compile only the preparation circuits (use for n=24: the QAOA layer is not executable)')
    ap.add_argument('--min-esp', type=float, default=0.05, help='with --run, circuits whose estimated success probability is below this are not submitted')
    a = ap.parse_args()
    global A, T, SECT, NV
    A, T, SECT = a.assets, a.periods, [int(x) for x in a.sectors.split(',')]; NV = 2 * A * T
    assert sum(SECT) == A
    from qiskit_ibm_runtime.fake_provider import FakeFez
    bk = FakeFez()
    h, J = qubo_cost(A, T, SEED); coeffs = ising_from_qubo(h, J)
    tuples, total = interface_tuples(A, T, SECT)
    print(f'n={NV} constraints={T*(2+len(SECT)+A)+1} interface tuples={len(tuples)} feasible strings={total}')
    # classical side: exact optimum over F and the tuple that contains it
    allc = []
    for iface, w in tuples:
        bl = tuple_support(iface); X, combos = full_strings(bl); allc.append((cost_of(X, h, J), X, bl, combos))
    opt = min(c.min() for c, *_ in allc); worst = max(c.max() for c, *_ in allc)
    opt_tuple = [i for i, (c, *_) in enumerate(allc) if abs(c.min() - opt) < 1e-9][0]
    chosen = sorted(range(len(tuples)), key=lambda i: -tuples[i][1])[:a.top]
    if opt_tuple not in chosen:
        chosen.append(opt_tuple)
    # angles: maximise the expected quality over ALL interface tuples (exact tensor simulation)
    def expected(params):
        q = 0.0; po = 0.0
        for (iface, w), (c, X, bl, combos) in zip(tuples, allc):
            P = tensor_qaoa(bl, combos, (c - opt) / (worst - opt), params).reshape(-1) if False else None
            cn = (c - opt) / (worst - opt)
            Pt = tensor_qaoa(bl, combos, cn, params)[tuple(combos.T)]
            q += w * (Pt * (1 - cn)).sum(); po += w * Pt[np.abs(c - opt) < 1e-9].sum()
        return q, po
    rng = np.random.default_rng(1); best = None
    for r in range(4):
        x0 = rng.uniform(0, 1.5, 2)
        res = minimize(lambda x: -expected(x)[0], x0, method='Nelder-Mead', options=dict(maxiter=60, xatol=1e-3, fatol=1e-5))
        if best is None or res.fun < best.fun:
            best = res
    params = best.x; q1, po1 = expected(params); q0, po0 = expected(np.zeros(2))
    print(f'exact numerics (all interface tuples, p=1 angles {np.round(params,3)}): quality {q0:.3f} -> {q1:.3f}, P(opt) {po0:.4f} -> {po1:.4f}')
    # circuits
    _os.makedirs('hardware_circuits', exist_ok=True)
    meta = dict(A=A, T=T, sectors=SECT, seed=SEED, n=NV, opt=float(opt), worst=float(worst), params=params.tolist(), circuits=[])
    print(f"{'interface (l,s per period/sector)':40s} {'circuit':8s} {'q':>4s} {'CZ':>5s} {'depth':>6s} {'dur(us)':>8s} {'d/T2':>5s} {'ESP':>6s}")
    for i in chosen:
        iface, w = tuples[i]
        variants = {'prep': build_hw_circuit(A, T, SECT, iface, coeffs, [], True)[0]}
        if not a.skip_qaoa:
            variants['qaoa1'] = build_hw_circuit(A, T, SECT, iface, coeffs, list(params), True)[0]
        for name, qc in variants.items():
            best_t = None
            for s in range(a.seeds):
                t = transpile(qc, bk, optimization_level=3, seed_transpiler=s + 1)
                esp = esp_of(t, bk)
                if best_t is None or esp > best_t[0]:
                    best_t = (esp, t)
            esp, t = best_t
            sch = transpile(t, bk, optimization_level=0, scheduling_method='alap')
            used = sorted({t.find_bit(q).index for inst in t.data for q in inst.qubits if inst.operation.name != 'barrier'})
            t2 = float(np.median([bk.qubit_properties(q).t2 for q in used]))
            dur = sch.duration * bk.dt * 1e6
            legal = all_legal(t, bk)
            print(f"{str(iface):40s} {name:8s} {len(used):4d} {t.count_ops().get('cz',0):5d} {t.depth():6d} {dur:8.1f} {dur/(t2*1e6):5.2f} {esp:6.3f} {'legal' if legal else 'ILLEGAL'}", flush=True)
            fn = f'hardware_circuits/large_case{a.tag}_{i}_{name}'
            with open(fn + '.qpy', 'wb') as f:
                qpy.dump(t, f)
            try:
                open(fn + '.qasm3', 'w').write(qasm3_dumps(t))
            except Exception as e:
                pass
            meta['circuits'].append(dict(index=i, iface=[[list(b) for b in per] for per in iface], weight=w, name=name, qubits=len(used), cz=t.count_ops().get('cz', 0),
                                         depth=t.depth(), duration_us=dur, esp=esp, legal=legal, file=fn + '.qpy'))
    json.dump(meta, open(f'hardware_circuits/large_case{a.tag}_meta.json', 'w'), indent=1)
    if a.run:
        run_on_hardware(a, meta, h, J, opt, worst)


def esp_of(t, bk):
    import math
    lg = 0.0
    for inst in t.data:
        nm = inst.operation.name
        if nm in ('barrier', 'delay'):
            continue
        qs = tuple(t.find_bit(q).index for q in inst.qubits)
        try:
            e = bk.target[nm][qs].error
        except Exception:
            e = None
        if e:
            lg += math.log1p(-min(e, 0.999))
    return math.exp(lg)


def all_legal(t, bk):
    for inst in t.data:
        nm = inst.operation.name
        if nm in ('barrier', 'measure', 'delay'):
            continue
        qs = tuple(t.find_bit(q).index for q in inst.qubits)
        if not bk.target.instruction_supported(nm, qs):
            return False
    return True


def run_on_hardware(a, meta, h, J, opt, worst):
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    key = json.load(open(a.apikey))
    svc = QiskitRuntimeService(channel='ibm_quantum_platform', token=key.get('apikey', key.get('token')) if isinstance(key, dict) else key)
    backend = svc.backend(a.backend)
    circs = []
    meta = dict(meta, circuits=[c for c in meta['circuits'] if c['esp'] >= a.min_esp or a.force])
    for c in meta['circuits']:
        circs.append(qpy.load(open(c['file'], 'rb'))[0])
    # circuits were compiled to the FakeFez snapshot of the same device; re-transpile for the live calibration, keeping the layout seeds
    isa = [transpile(c, backend, optimization_level=3, seed_transpiler=1) for c in circs]
    job = Sampler(mode=backend).run(isa, shots=a.shots)
    res = job.result()
    out = []
    for c, r in zip(meta['circuits'], res):
        counts = r.data.x.get_counts()
        out.append(dict(c, counts=counts))
    json.dump(dict(meta=meta, results=out, backend=a.backend, shots=a.shots), open(f'results/hardware_large_case{a.tag}_raw.json', 'w'))
    print('saved results/hardware_large_case<tag>_raw.json; analyse with experiments/hardware_large_case_report.py')


if __name__ == '__main__':
    main()

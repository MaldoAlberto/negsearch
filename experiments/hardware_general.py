"""Hardware (or dry-run) experiment for the general construction, with the controls that turn predictions into data.

Circuits (all on the 6-variable exact-cover instance of the notebook; measured on the 6 decision qubits):
  uniform      : |+>^6, no constraint information (reference for 'feasible by chance')
  Aq           : negation-forced state A_q (24 CZ)
  Aq+iter      : A_q + one amplification iteration (about 330 CZ); ideal P(optimum) = 1
  noise ctrl   : A_q followed by pairs of identical CZ gates separated by barriers, padded to the CZ count of Aq+iter.
                 Ideally it equals A_q, so any drop relative to A_q is depth/noise, not logic.
Each circuit is transpiled with several seeds (different layouts) and run in the requested error-suppression modes.
Output: results/hardware_general_<tag>.json with raw counts, CZ counts, calibration snapshot and the analytic references.

Usage
  python experiments/hardware_general.py --dry-run                       # FakeFez noise model, no account needed
  python experiments/hardware_general.py --backend ibm_fez --apikey apikey_personal.json --shots 8000 --seeds 3 --modes plain,dd_twirl
Never commit the API key file (.gitignore excludes apikey*.json)."""
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT)
import sys; sys.path.insert(0, ROOT)
import argparse, json, math, time
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from negsearch.automaton import build_automaton, compile_Aq, mu_q, grover_iteration_circuit, from_qiskit
from negsearch.families import hardware_instance


def with_meas(circ, n):
    c = QuantumCircuit(circ.num_qubits, n)
    c.compose(circ, inplace=True)
    for i in range(n):
        c.measure(i, i)
    return c


def logical_circuits(P):
    n = P.n
    A = build_automaton(P.feas, n)
    G = P.good(0.0)
    Aq, _ = compile_Aq(A, 0.5)
    B, _ = grover_iteration_circuit(A, G, vchain=True)
    U = QuantumCircuit(n)
    U.h(range(n))
    return {'uniform': with_meas(U, n), 'Aq': with_meas(Aq, n), 'Aq+iter': with_meas(B, n)}, A, G


def pad_isa(t, extra_pairs):
    """insert `extra_pairs` pairs of identical CZ gates (barrier-separated) before the measurements of an ISA circuit"""
    ops = list(t.data)
    first_meas = next(i for i, inst in enumerate(ops) if inst.operation.name == 'measure')
    # keep a trailing barrier that precedes the measurements together with the measurements
    while first_meas > 0 and ops[first_meas - 1].operation.name == 'barrier':
        first_meas -= 1
    body, tail = ops[:first_meas], ops[first_meas:]
    edges = []
    for inst in body:
        if inst.operation.name == 'cz':
            e = tuple(t.find_bit(q).index for q in inst.qubits)
            if e not in edges:
                edges.append(e)
    new = t.copy_empty_like()
    for inst in body:
        new.append(inst.operation, inst.qubits, inst.clbits)
    active = sorted({t.find_bit(q).index for inst in body for q in inst.qubits})
    for k in range(extra_pairs):                      # serial pairs: a full-width barrier after each CZ keeps the depth of the real circuit
        a, b = edges[k % len(edges)]
        for _ in range(2):
            new.cz(a, b); new.barrier(*[new.qubits[i] for i in active])
    for inst in tail:
        new.append(inst.operation, inst.qubits, inst.clbits)
    return new


def cz_of(t):
    return t.count_ops().get('cz', 0)


def decode(counts, n):
    out = np.zeros(2 ** n)
    for k, v in counts.items():
        bits = [int(ch) for ch in k.replace(' ', '')[::-1]]
        out[int(''.join(map(str, bits)), 2)] += v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', default='ibm_fez'); ap.add_argument('--apikey', default='apikey_personal.json')
    ap.add_argument('--shots', type=int, default=8000); ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--modes', default='plain,dd_twirl'); ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--tag', default=None); ap.add_argument('--estimate', action='store_true')
    a = ap.parse_args()
    P = hardware_instance(); n = P.n
    circs, A, G = logical_circuits(P)
    mu, _ = mu_q(A, 0.5)

    if a.dry_run:
        from qiskit_ibm_runtime.fake_provider import FakeFez
        backend = FakeFez(); noisy = AerSimulator.from_backend(backend); bname = 'FakeFez(noise model)'
        modes = ['plain']
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
        service = QiskitRuntimeService(channel='ibm_cloud', token=json.load(open(a.apikey))['apikey'])
        try:
            backend = service.backend(a.backend)
        except Exception:
            backend = service.least_busy(operational=True, simulator=False, min_num_qubits=20)
        bname = backend.name; modes = a.modes.split(',')

    # transpile with several seeds; build the noise control for each seed
    isa = {}                                           # (seed, name) -> circuit
    for s in range(a.seeds):
        for name, c in circs.items():
            isa[(s, name)] = transpile(c, backend, optimization_level=3, seed_transpiler=s + 1)
        extra = max(0, (cz_of(isa[(s, 'Aq+iter')]) - cz_of(isa[(s, 'Aq')])) // 2)
        isa[(s, 'noise ctrl')] = pad_isa(isa[(s, 'Aq')], extra)
    keys = list(isa)
    cz = {f'{s}|{name}': cz_of(isa[(s, name)]) for s, name in keys}
    print('CZ counts per seed:', {name: [cz[f'{s}|{name}'] for s in range(a.seeds)] for name in ['uniform', 'Aq', 'Aq+iter', 'noise ctrl']})
    if a.estimate:
        print(f'circuits per job: {len(keys)}; jobs: {len(modes)}; shots: {a.shots}; total shots: {len(keys) * len(modes) * a.shots}')
        return

    raw = {}
    for mode in modes:
        t0 = time.time()
        if a.dry_run:
            res = noisy.run([isa[k] for k in keys], shots=a.shots, seed_simulator=1).result()
            counts = [res.get_counts(i) for i in range(len(keys))]; job_id = 'dry-run'
        else:
            sampler = Sampler(mode=backend)
            if mode == 'dd_twirl':
                sampler.options.dynamical_decoupling.enable = True
                sampler.options.twirling.enable_gates = True
            job = sampler.run([isa[k] for k in keys], shots=a.shots)
            job_id = job.job_id(); print(f'[{mode}] job_id {job_id} submitted'); res = job.result()
            counts = [dict(res[i].data.c.get_counts()) for i in range(len(keys))]
        raw[mode] = dict(job_id=job_id, seconds=time.time() - t0,
                         counts={f'{s}|{name}': counts[i] for i, (s, name) in enumerate(keys)})
        print(f'[{mode}] done in {time.time() - t0:.0f}s')

    calib = {}
    try:
        errs = [v.error for v in backend.target['cz'].values() if v is not None and v.error is not None]
        calib = dict(median_cz_error=float(np.median(errs)), n_cz_edges=len(errs), taken=time.strftime('%Y-%m-%d %H:%M:%S'))
    except Exception as e:
        calib = dict(note=f'calibration snapshot unavailable: {e!r}'[:160])
    ref = dict(uniform_p_feasible=float(P.nF / 2 ** n), uniform_p_optimum=float(G.sum() / 2 ** n), ideal_Aq_p_optimum=float(mu[G].sum()),
               ideal_iter_p_optimum=1.0, nF=int(P.nF), n=n)
    tag = a.tag or (('dryrun' if a.dry_run else bname) + time.strftime('_%Y%m%d_%H%M%S'))
    out = dict(backend=bname, shots=a.shots, seeds=a.seeds, modes=modes, cz=cz, calibration=calib, reference=ref,
               feas=P.feas.astype(int).tolist(), good=G.astype(int).tolist(), n=n, raw=raw, instance=P.name)
    fn = f'results/hardware_general_{tag}.json'
    json.dump(out, open(fn, 'w'))
    print('saved', fn)


if __name__ == '__main__':
    main()

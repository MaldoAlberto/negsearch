"""Measurement-based uncomputation of the generic automaton register, with the registers of all blocks sharing one pool of reused qubits.  For the three cores, the PREPARATION of the
generic A_q compiled four ways (ibm_fez, best of 3 seeds): (a) compile_Aq with uncomputation (as in generic_vs_symmetric), (b) one-hot register + relative-phase Toffolis, undone in reverse,
(c) the same with measurement-based uncomputation: CZ of the compute part, and corrections applied unconditionally (worst case; the expected number is the compute part + half of the corrections),
plus the one QAOA layer built on (b) and (c).  Exactness of the measurement-based preparation is checked by sampling all measurement outcomes on small blocks.  Writes results/mbu_compare_<tag>.json."""
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT)
import sys; sys.path.insert(0, ROOT)
import json, os, warnings; warnings.filterwarnings('ignore')
from qiskit import QuantumCircuit, ClassicalRegister
import experiments.case_four as C
from negsearch.hw_large import ising_from_qubo, swap_pairs, pair_swap_mixer
from negsearch.depth_study import phase_poly
from negsearch.symlower import symmetric_block_prep
from negsearch.automaton import compile_Aq
from negsearch.large_case import sector_block_automaton
from negsearch.onehot_rc import compile_Aq_onehot_rc
from negsearch.mbu import compile_Aq_onehot_mbu
from experiments.qaoa_realistic import best as compile_best

cp = C.core_problem(); bl, h1, J1 = cp['bl'], cp['h1'], cp['J1']; n = C.n


def build(kind, layer):
    nontriv = [b for b in bl if len(b['F']) > 1]
    if kind == 'mbu0' or kind == 'mbuU':
        autos = {id(b): sector_block_automaton(b['size'], b['l'], b['s']) for b in nontriv}
        circs = {id(b): compile_Aq_onehot_mbu(autos[id(b)], 0.5, False, corrections=(kind == 'mbuU'))[0] for b in nontriv}
        pool = max(c.num_qubits - 2 * b['size'] for b, c in ((b, circs[id(b)]) for b in nontriv)); ncl = max(circs[id(b)].num_clbits for b in nontriv)
        qc = QuantumCircuit(n + pool); qc.add_register(ClassicalRegister(ncl, 'm'))
        for b in bl:
            if len(b['F']) == 1: qc.compose(symmetric_block_prep(b['size'], b['l'], b['s']), qubits=b['xq'], inplace=True); continue
            c = circs[id(b)]; extra = list(range(n, n + c.num_qubits - 2 * b['size']))
            qc.compose(c, qubits=b['xq'] + extra, clbits=list(range(c.num_clbits)), inplace=True)
    else:
        preps = []; pos = n
        for b in bl:
            if len(b['F']) > 1:
                A = sector_block_automaton(b['size'], b['l'], b['s'])
                circ = (compile_Aq(A, 0.5, uncompute=True)[0] if kind == 'cAq' else compile_Aq_onehot_rc(A, 0.5, uncompute=True)[0])
                extra = list(range(pos, pos + circ.num_qubits - 2 * b['size'])); pos += len(extra); preps.append((b['xq'] + extra, circ))
            else: preps.append((b['xq'], symmetric_block_prep(b['size'], b['l'], b['s'])))
        qc = QuantumCircuit(pos)
        for qs, c in preps: qc.compose(c, qubits=qs, inplace=True)
    if layer:
        phase_poly(qc, ising_from_qubo(h1, J1), 0.37)
        for b in bl:
            if len(b['F']) == 1: continue
            xq = b['xq']; two = b['l'] == 0 or b['s'] == 0; d = 1 if b['l'] == 0 else 0
            for j, k in swap_pairs(b['size'], 'path'):
                if two: qc.rxx(0.6, xq[2 * j + d], xq[2 * k + d]); qc.ryy(0.6, xq[2 * j + d], xq[2 * k + d]); qc.rzz(0.6, xq[2 * j + d], xq[2 * k + d])
                else: pair_swap_mixer(qc, xq[2 * j], xq[2 * j + 1], xq[2 * k], xq[2 * k + 1], 0.6)
    return qc


def compact(qc):
    used = sorted({qc.find_bit(q).index for i in qc.data for q in i.qubits}); mp = {q: k for k, q in enumerate(used)}
    out = QuantumCircuit(len(used), qc.num_clbits)
    for i in qc.data:
        out.append(i.operation, [mp[qc.find_bit(q).index] for q in i.qubits], [out.clbits[qc.find_bit(c).index] for c in i.clbits])
    return out


out = {}
for kind in ('cAq', 'onehot_rc', 'mbu0', 'mbuU'):
    for layer in (False, True):
        if layer and kind in ('cAq', 'mbu0'): continue
        k = f"{kind}_{'layer' if layer else 'prep'}"
        qc_ = compact(build(kind, layer))
        if qc_.num_qubits > 156: out[k] = dict(qubits=qc_.num_qubits, cz=None, note='does not fit in 156 qubits'); print(f'{k:18s} qubits {qc_.num_qubits} (does not fit)', flush=True); continue
        r = compile_best(qc_, seeds=3); out[k] = r
        print(f"{k:18s} qubits {r['qubits']:3d} CZ {r['cz']:6d} depth {r['depth']:6d} ESP {r['esp']:.3g}", flush=True)
out['mbu_expected_prep_cz'] = out['mbu0_prep']['cz'] + 0.5 * (out['mbuU_prep']['cz'] - out['mbu0_prep']['cz'])
print('expected CZ of mbu prep', out['mbu_expected_prep_cz'])
json.dump(out, open(f"results/mbu_compare_{os.environ.get('CF_TAG', 'n64')}.json", 'w'), indent=1)

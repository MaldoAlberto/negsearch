"""Our construction without the Dicke shortcut: one QAOA layer and the preparation alone on the three cores, with the generic constraint-automaton preparation A_q
(register-based, registers uncomputed afterwards, one-hot/binary as compile_Aq) against the symmetry-aware lowering (Dicke + controlled swaps).  Same objective, same mixers.
Compiled to ibm_fez (best of 3 seeds).  Writes results/generic_vs_symmetric_<tag>.json."""
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT)
import sys; sys.path.insert(0, ROOT)
import json, os, warnings; warnings.filterwarnings('ignore')
from qiskit import QuantumCircuit
import experiments.case_four as C
import experiments.prune_circuit as P
from negsearch.hw_large import ising_from_qubo, swap_pairs, pair_swap_mixer
from negsearch.depth_study import phase_poly
from negsearch.symlower import symmetric_block_prep
from negsearch.automaton import compile_Aq
from negsearch.large_case import sector_block_automaton
from experiments.qaoa_realistic import best as compile_best

cp = C.core_problem(); bl, h1, J1 = cp['bl'], cp['h1'], cp['J1']; n = C.n


def build(generic, with_layer):
    preps = []; pos = n
    for b in bl:
        if generic and len(b['F']) > 1:
            circ = compile_Aq(sector_block_automaton(b['size'], b['l'], b['s']), 0.5, uncompute=True)[0]
            extra = list(range(pos, pos + circ.num_qubits - 2 * b['size'])); pos += len(extra); preps.append((b['xq'] + extra, circ))
        else: preps.append((b['xq'], symmetric_block_prep(b['size'], b['l'], b['s'])))
    qc = QuantumCircuit(pos)
    for qs, c in preps: qc.compose(c, qubits=qs, inplace=True)
    if with_layer:
        phase_poly(qc, ising_from_qubo(h1, J1), 0.37)
        for b in bl:
            if len(b['F']) == 1: continue
            xq = b['xq']; two = b['l'] == 0 or b['s'] == 0; d = 1 if b['l'] == 0 else 0
            for j, k in swap_pairs(b['size'], 'path'):
                if two: qc.rxx(0.6, xq[2 * j + d], xq[2 * k + d]); qc.ryy(0.6, xq[2 * j + d], xq[2 * k + d]); qc.rzz(0.6, xq[2 * j + d], xq[2 * k + d])
                else: pair_swap_mixer(qc, xq[2 * j], xq[2 * j + 1], xq[2 * k], xq[2 * k + 1], 0.6)
    return qc


out = {}
for gen in (False, True):
    for layer in (False, True):
        k = ('generic' if gen else 'symmetric') + ('_layer' if layer else '_prep')
        qc, = (build(gen, layer),)
        r = compile_best(C.compact(qc)[0], seeds=3); out[k] = r
        print(f"{k:18s} qubits {r['qubits']:3d} CZ {r['cz']:6d} depth {r['depth']:6d} ESP {r['esp']:.3g}", flush=True)
json.dump(out, open(f"results/generic_vs_symmetric_{os.environ.get('CF_TAG', 'n64')}.json", 'w'), indent=1)

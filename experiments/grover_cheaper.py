"""One Grover iteration (A^dag, reflection about |0>, A) on the n=64 and n=192 cores: ancilla-free reflection (as in case_four / case_large) against the Toffoli
V-chain reflection with clean ancillas, and the variants 'recursion' and 'v-chain-dirty'.  Compiled to ibm_fez, best of 3 seeds.  Writes results/grover_cheaper.json."""
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT)
import sys; sys.path.insert(0, ROOT)
import json, os, warnings; warnings.filterwarnings('ignore')
from qiskit import QuantumCircuit
import experiments.case_four as C
from negsearch.symlower import symmetric_block_prep
from experiments.qaoa_realistic import best as compile_best

cp = C.core_problem(); bl = cp['bl']
prep = QuantumCircuit(C.n)
for b in bl: prep.compose(symmetric_block_prep(b['size'], b['l'], b['s']), qubits=b['xq'], inplace=True)
gc, _ = C.compact(prep); fq = gc.num_qubits
print('core qubits', fq, flush=True)
out = {'f': fq}
for mode in ('noancilla', 'v-chain', 'v-chain-dirty', 'recursion'):
    anc = max(0, fq - 3) if mode == 'v-chain' else (fq - 3 if mode == 'v-chain-dirty' else 1)
    it = QuantumCircuit(fq + (anc if mode != 'noancilla' else 0)); qs = list(range(fq)); an = list(range(fq, it.num_qubits))
    it.compose(gc.inverse(), qubits=qs, inplace=True); it.x(qs); it.h(qs[-1])
    if mode == 'noancilla': it.mcx(qs[:-1], qs[-1])
    elif mode == 'v-chain': it.mcx(qs[:-1], qs[-1], an[:fq - 3], mode='v-chain')
    elif mode == 'v-chain-dirty': it.mcx(qs[:-1], qs[-1], an[:fq - 3], mode='v-chain-dirty')
    else: it.mcx(qs[:-1], qs[-1], an[:1], mode='recursion')
    it.h(qs[-1]); it.x(qs); it.compose(gc, qubits=qs, inplace=True)
    r = compile_best(it, seeds=3); out[mode] = r
    print(f"{mode:14s} qubits {r['qubits']:3d} CZ {r['cz']:6d} depth {r['depth']:6d} ESP {r['esp']:.3g}", flush=True)
json.dump(out, open(f"results/{os.environ.get('CF_TAG', 'grover_cheaper')}.json", 'w'), indent=1)

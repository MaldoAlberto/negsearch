"""E4b': scalable equivalence check of ZX-optimised components: U_orig followed by U_zx^dagger must
return every input exactly. Inputs: random basis states on variables + H on variable 1 (detects phases).
Sampled with MPS; equivalence <=> all shots return the all-zero string after undoing the input prep."""
import os as _os, sys as _sys
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..'))
_os.chdir(ROOT); _sys.path.insert(0, ROOT)
import random, sys, json
sys.path.insert(0, ROOT)
from negsearch.core import *
from negsearch.circuits import component_circuits
from negsearch.zxtools import *
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import os
sim=AerSimulator(method='matrix_product_state'); rng=random.Random(78)
out=json.load(open('results/e4_verify_mps.json')) if os.path.exists('results/e4_verify_mps.json') else []
for fam,n in [('3-SAT',6),('1-in-3-SAT',8),('3-COL',8)]:
    f=sample_satisfiable(fam,n,rng)
    order=list(range(1,n+1)); rng.shuffle(order)
    if any(r['family']==fam for r in out): continue
    for name,qc in component_circuits(f,order,None).items():
        ct=to_clifford_t(qc); c2,_=zx_reduce(ct); q2=zx_to_qiskit(c2)
        bad=0; S=200
        for trial in range(4):
            prep=QuantumCircuit(ct.num_qubits)
            xs=[i for i in range(n) if rng.random()<.5]
            for i in xs: prep.x(i)
            prep.h(0)
            full=prep.compose(ct).compose(q2.inverse()).compose(prep.inverse()); full.measure_all()
            cnt=sim.run(transpile(full,basis_gates=['cx','h','t','tdg','s','sdg','x','z','measure','swap'],optimization_level=0),shots=S,seed_simulator=trial).result().get_counts()
            bad+=sum(c for k,c in cnt.items() if '1' in k)
        out.append(dict(family=fam,n=n,component=name,qubits=ct.num_qubits,T_orig=counts_qiskit(ct)['T'],T_zx=c2.tcount(),
                        CX_orig=counts_qiskit(ct)['CX'],CX_zx=counts_zx(c2)['CX'],shots=4*S,nonzero_outcomes=bad))
        print(out[-1],flush=True); json.dump(out,open('results/e4_verify_mps.json','w'),indent=1)

"""E4b: equivalence of ZX-optimised components with originals (statevector on random inputs)."""
import os as _os, sys as _sys
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..'))
_os.chdir(ROOT); _sys.path.insert(0, ROOT)
import random, sys, json
sys.path.insert(0, ROOT)
from negsearch.core import *
from negsearch.circuits import component_circuits
from negsearch.zxtools import *
rng=random.Random(77); out=[]
for fam,n in [('3-SAT',5),('1-in-3-SAT',6),('3-COL',6),('4-SAT',5)]:
    f=sample_satisfiable(fam,n,rng); order=list(range(1,n+1)); rng.shuffle(order)
    for name,qc in component_circuits(f,order,None).items():
        ct=to_clifford_t(qc)
        if ct.num_qubits>24: print('skip',fam,name,ct.num_qubits); continue
        c2,_=zx_reduce(ct); q2=zx_to_qiskit(c2)
        fid=equivalent_on_inputs(ct,q2,n,rng)
        out.append(dict(family=fam,n=n,component=name,qubits=ct.num_qubits,T_orig=counts_qiskit(ct)['T'],T_zx=c2.tcount(),min_fidelity=fid))
        print(out[-1],flush=True)
json.dump(out,open('results/e4_verify.json','w'),indent=1)

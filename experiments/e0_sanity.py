import os as _os, sys as _sys
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..'))
_os.chdir(ROOT); _sys.path.insert(0, ROOT)
import random, sys, math, time
sys.path.insert(0, ROOT)
from negsearch.core import *
from negsearch.circuits import NegForcedSearch, for_simulation
from qiskit import transpile
from qiskit_aer import AerSimulator
sim=AerSimulator(method='matrix_product_state')
rng=random.Random(3)
for fam,n in [('3-COL',12)]:
    f=sample_satisfiable(fam,n,rng); order=list(range(1,n+1)); rng.shuffle(order)
    qc=NegForcedSearch(f,order).amplified(0); t=time.time(); S=20000
    cnt=sim.run(for_simulation(qc),shots=S,seed_simulator=1).result().get_counts()
    pq=sum(c for s,c in cnt.items() if f.check_bitstring(s))/S; pe=exact_success(f,order)
    print(f"{fam:11s} n={n} M={f.meta['M']} qubits={qc.num_qubits} p_MPS={pq:.4f}±{math.sqrt(pe*(1-pe)/S):.4f} p_exact={pe:.4f} p_grover={f.meta['M']/2**n:.4f} ({time.time()-t:.0f}s)",flush=True)

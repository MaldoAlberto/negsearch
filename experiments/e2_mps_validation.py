"""E2: circuit-level validation with Qiskit Aer MPS.
 (a) A|0> success probability vs exact theory for large n
 (b) full amplitude amplification (A + k iterations of Q) vs theory, and Grover baseline."""
import os as _os, sys as _sys
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..'))
_os.chdir(ROOT); _sys.path.insert(0, ROOT)
import random, sys, math, time, json
sys.path.insert(0, ROOT)
from negsearch.core import *
from negsearch.circuits import NegForcedSearch, for_simulation
from qiskit_aer import AerSimulator
sim=AerSimulator(method='matrix_product_state')
part=sys.argv[1]; fn=f'results/e2{part}.json'
import os
out=json.load(open(fn)) if os.path.exists(fn) else []
done=lambda fam,n: sum(1 for r in out if r['family']==fam and r['n']==n)
rng=random.Random(100 if part=='a' else 200)
def run(qc,S):
    t=time.time(); cnt=sim.run(for_simulation(qc),shots=S,seed_simulator=7).result().get_counts(); return cnt,time.time()-t
if part=='a':
    for n in [12,16,20,24,28,32]:
        for fam in ['3-SAT','4-SAT','1-in-3-SAT','3-COL']:
            if fam=='4-SAT' and n>12: continue
            for inst in range(done(fam,n),2):
                f=sample_satisfiable(fam,n,rng); order=list(range(1,n+1)); rng.shuffle(order)
                pe=exact_success(f,order); S=4000
                qc=NegForcedSearch(f,order).amplified(0); cnt,dt=run(qc,S)
                pm=sum(c for s,c in cnt.items() if f.check_bitstring(s))/S
                out.append(dict(family=fam,n=n,M=f.meta['M'],clauses=f.clauses,order=order,qubits=qc.num_qubits,p_exact=pe,p_mps=pm,shots=S,
                                se=math.sqrt(max(pe*(1-pe),1e-12)/S),p_grover=f.meta['M']/2**n,secs=dt))
                print(out[-1],flush=True); json.dump(out,open(fn,'w'),indent=1)
else:
    for n in [8,10,12,14]:
        for fam in ['3-SAT','4-SAT','1-in-3-SAT','3-COL']:
            if fam=='4-SAT' and n>10: continue
            if fam=='3-COL' and n<10: continue
            f=sample_satisfiable(fam,n,rng); order=list(range(1,n+1)); rng.shuffle(order)
            pe=exact_success(f,order); pg=f.meta['M']/2**n
            for label,p,base in [('negation-forced',pe,False),('Grover',pg,True)]:
                k=amplification_iterations(p); S=1000
                qc=NegForcedSearch(f,order,baseline=base).amplified(k); cnt,dt=run(qc,S)
                pm=sum(c for s,c in cnt.items() if f.check_bitstring(s))/S
                th=amplified_success(p,k)
                out.append(dict(family=fam,n=n,M=f.meta['M'],clauses=f.clauses,order=order,method=label,p0=p,k=k,qubits=qc.num_qubits,
                                success_theory=th,success_mps=pm,se=math.sqrt(max(th*(1-th),1e-12)/S),shots=S,secs=dt,
                                gates=sum(for_simulation(qc).count_ops().values())))
                print(out[-1],flush=True); json.dump(out,open(fn,'w'),indent=1)

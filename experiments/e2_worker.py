"""One E2 run (part a or b) for a given family, n and seed; appends a JSON line."""
import os as _os, sys as _sys
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..'))
_os.chdir(ROOT); _sys.path.insert(0, ROOT)
import random, sys, math, time, json
sys.path.insert(0, ROOT)
from negsearch.core import *
from negsearch.circuits import NegForcedSearch, for_simulation
from qiskit_aer import AerSimulator
part,fam,n,seed=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4])
rng=random.Random(seed); sim=AerSimulator(method='matrix_product_state')
f=sample_satisfiable(fam,n,rng); order=list(range(1,n+1)); rng.shuffle(order)
sols=enumerate_models(f); pe,_=success_by_solutions(f,order,sols); pg=len(sols)/2**n
def run(qc,S):
    t=time.time(); cnt=sim.run(for_simulation(qc),shots=S,seed_simulator=7).result().get_counts(); return cnt,time.time()-t
recs=[]
if part=='a':
    S=4000; qc=NegForcedSearch(f,order).amplified(0); cnt,dt=run(qc,S)
    pm=sum(c for s,c in cnt.items() if f.check_bitstring(s))/S
    recs.append(dict(family=fam,n=n,M=len(sols),seed=seed,qubits=qc.num_qubits,p_exact=pe,p_mps=pm,shots=S,
                     se=math.sqrt(max(pe*(1-pe),1e-12)/S),p_grover=pg,secs=dt))
else:
    for label,p,base in [('negation-forced',pe,False),('Grover',pg,True)]:
        if base and n>8: continue   # uniform-preparation baseline only at n=8 (its MPS cost explodes)
        k=amplification_iterations(p); S=1000
        qc=NegForcedSearch(f,order,baseline=base).amplified(k); cnt,dt=run(qc,S)
        pm=sum(c for s,c in cnt.items() if f.check_bitstring(s))/S; th=amplified_success(p,k)
        rec=dict(family=fam,n=n,M=len(sols),seed=seed,method=label,p0=p,k=k,qubits=qc.num_qubits,
                         success_theory=th,success_mps=pm,se=math.sqrt(max(th*(1-th),1e-12)/S),shots=S,secs=dt,
                         gates=sum(for_simulation(qc).count_ops().values()))
        with open(f'results/e2{part}.jsonl','a') as fh: fh.write(json.dumps(rec)+'\n')
        print(rec,flush=True)
if part=='a':
    with open(f'results/e2{part}.jsonl','a') as fh:
        for r in recs: fh.write(json.dumps(r)+'\n')
    print(recs,flush=True)

"""E4: ZX (PyZX full_reduce + extraction) T/CX reduction per component (A, O, S0).
   E5: condition selection: total T-cost vs K (max forcing conditions per variable per polarity)."""
import os as _os, sys as _sys
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..'))
_os.chdir(ROOT); _sys.path.insert(0, ROOT)
import random, sys, math, time, json
sys.path.insert(0, ROOT)
from negsearch.core import *
from negsearch.circuits import component_circuits
from negsearch.zxtools import *
fam=sys.argv[1]; sizes=[int(x) for x in sys.argv[2].split(',')]
rng=random.Random(500+len(fam)); out=[]; fn=f'results/e45_{fam}.json'
def zx_cost(qc):
    ct=to_clifford_t(qc); c2,dt=zx_reduce(ct); return counts_qiskit(ct), counts_zx(c2), dt
for n in sizes:
    for inst in range(2):
        f=sample_satisfiable(fam,n,rng); sols=enumerate_models(f)
        order=list(range(1,n+1)); rng.shuffle(order)
        comp=component_circuits(f,order,None)
        O0,Oz,dO=zx_cost(comp['O']); S0,Sz,dS=zx_cost(comp['S0'])
        rec=dict(family=fam,n=n,m=f.m,M=len(sols),clauses=f.clauses,order=order,O=dict(orig=O0,zx=Oz,secs=dO),S0=dict(orig=S0,zx=Sz,secs=dS),K={})
        for K in [0,1,2,3,None]:
            if K==0:
                p=len(sols)/2**n; A0=Az=dict(T=0,CX=0,total=n,qubits=n); dA=0
            else:
                p,_=success_by_solutions(f,order,sols,K)
                A0,Az,dA=zx_cost(component_circuits(f,order,K)['A'])
            it=math.pi/(4*math.sqrt(p))
            per0=2*A0['T']+O0['T']+S0['T']; perz=2*Az['T']+Oz['T']+Sz['T']
            rec['K'][str(K)]=dict(p=p,iters=it,A_orig=A0,A_zx=Az,secs=dA,T_total_orig=it*per0,T_total_zx=it*perz,
                                  CX_total_orig=it*(2*A0['CX']+O0['CX']+S0['CX']),CX_total_zx=it*(2*Az['CX']+Oz['CX']+Sz['CX']))
            print(fam,n,inst,'K',K,f"p={p:.2e} T/iter orig={per0} zx={perz} totalT orig={it*per0:.3g} zx={it*perz:.3g}",flush=True)
        out.append(rec); json.dump(out,open(fn,'w'),indent=1)

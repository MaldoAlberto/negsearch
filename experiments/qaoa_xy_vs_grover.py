"""QAOA on the 3-of-6 cardinality instance (Dicke state): XY-ring mixer vs Grover mixer. Angles optimised without noise (Nelder-Mead, 12 restarts
per depth, local optimum only), CZ/depth/ESP compiled on the ibm_fez topology (FakeFez), nothing executed. Writes results/qaoa_xy_vs_grover.json."""
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")); _os.chdir(ROOT)
import json
import sys,math,warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,ROOT)
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.circuit.library import XXPlusYYGate
from qiskit.quantum_info import Statevector
from negsearch import depth_study as ds
from negsearch.automaton import *
from negsearch.dicke import dicke_circuit
from negsearch.fez_check import measure
from negsearch import families as fam

def xy_qaoa(D,co,params,n,ring=True):
    p=len(params)//2; qc=QuantumCircuit(n); qc.compose(D,inplace=True)
    edges=[(i,i+1) for i in range(0,n-1,2)]+[(i,i+1) for i in range(1,n-1,2)]+([(n-1,0)] if ring and n>2 else [])
    for l in range(p):
        ds.phase_poly(qc,co,params[l])
        for a,b in edges: qc.append(XXPlusYYGate(2*params[p+l]),[a,b])
    return qc
def fid_xy(prm,D,co,n,G):
    p=Statevector(xy_qaoa(D,co,prm,n)).probabilities(); return -sum(p[from_idx(i,n)] for i in G)
def from_idx(i,n): # Qiskit little-endian index of MSB-first string i
    return int(format(i,f'0{n}b')[::-1],2)
P=fam.cardinality_qp(6,3,0); n=6; G=list(np.flatnonzero(P.good(0.0))); F=P.feas
c=(P.cost-P.opt)/(P.worst-P.opt); co=ds.walsh(c,n); D=dicke_circuit(6,3)
psi0=F/np.sqrt(F.sum()); rng=np.random.default_rng(0)
print('|F|',F.sum(),'a0=P(opt) of Dicke',G and len(G)/F.sum())
res={}
for p in (1,2,3):
    bx=0;bg=0
    for r in range(12):
        x0=rng.uniform(0,2*math.pi,2*p)
        r1=minimize(fid_xy,x0,args=(D,co,n,G),method='Nelder-Mead',options=dict(maxiter=400,xatol=1e-3,fatol=1e-4)); bx=max(bx,-r1.fun)
        r2=minimize(lambda x:-(np.abs(gm_qaoa_state(psi0,c,x))**2)[G].sum(),x0,method='Nelder-Mead',options=dict(maxiter=400)); bg=max(bg,-r2.fun)
    res[p]=dict(p=p,p_opt_xy=bx,p_opt_grover=bg); print(f"p={p}: best P(opt)  XY-ring mixer {bx:.3f}   Grover mixer {bg:.3f}",flush=True)
# compile costs
for p in (1,2):
    prm=[0.5]*(2*p)
    m=measure(xy_qaoa(D,co,prm,n)); 
    prmg=[0.5]*(2*p); mg=measure(ds.qaoa_from_prep(D,n,co,prmg))
    res[p].update(cz_xy=m['cz'],depth_xy=m['depth'],esp_xy=m['esp'],cz_gm=mg['cz'],depth_gm=mg['depth'],esp_gm=mg['esp'])
    print(f"p={p} compile: XY ring CZ {m['cz']:.0f} depth {m['depth']:.0f} dur/T2 {m['dur_us']/m['t2_us']:.2f} ESP {m['esp']:.3f} | Grover mixer CZ {mg['cz']:.0f} depth {mg['depth']:.0f} dur/T2 {mg['dur_us']/mg['t2_us']:.2f} ESP {mg['esp']:.3f}",flush=True)

json.dump(list(res.values()), open('results/qaoa_xy_vs_grover.json', 'w'), indent=1)

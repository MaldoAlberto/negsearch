"""Search QOBLIB-derived induced subgraphs (n=18) where min-degree greedy (= CG prep with q->1) is NOT optimal."""
import os as _os, sys as _sys
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..'))
_os.chdir(ROOT); _sys.path.insert(0, ROOT)
import numpy as np, json, itertools
from negsearch.qaoa_mis import read_gph, MIS, prep_biased
Q=_os.environ.get('QOBLIB_MIS','qoblib/07-independentset/')+'instances/'
def greedy(n,N,order):
    S=set()
    for v in order:
        if not (N[v]&S): S.add(v)
    return len(S)
cands=[]
rng=np.random.default_rng(11)
for name in ['frb45-21-3','frb50-23-3','brock200-1','keller4','C125-9','johnson8-2-4','hamming6-4','aves-sparrow-social','chesapeake','football','karate','ibm32','MANN-a9']:
    n0,E0=read_gph(Q+name+'.gph'); N0=[set() for _ in range(n0)]
    for a,b in E0: N0[a].add(b); N0[b].add(a)
    found=0
    for t in range(300):
        # connected-ish sample: BFS ball from a random root, truncated to 18
        root=int(rng.integers(n0)); sel=[root]; frontier=list(N0[root]); rng.shuffle(frontier)
        while len(sel)<18 and frontier:
            v=frontier.pop(0)
            if v not in sel: sel.append(v); nb=list(N0[v]-set(sel)); rng.shuffle(nb); frontier+=nb
        if len(sel)<18: continue
        idx={v:i for i,v in enumerate(sel)}; E=[(idx[a],idx[b]) for a,b in E0 if a in idx and b in idx]
        g=MIS(18,E); order=sorted(range(18),key=lambda v:(len(g.N[v]),v))
        gr=greedy(18,g.N,order)
        if gr<g.opt:
            p05=g.metrics(prep_biased(g,order,0.5))['p_opt']
            nopt=int((g.feas&(g.size==g.opt)).sum())
            cands.append(dict(source=name,vertices=[int(v)+1 for v in sel],m=len(E),opt=int(g.opt),greedy=gr,n_optimal=nopt,P_prep_q05=p05))
            found+=1
            if found>=3: break
    print(name, 'found', found, flush=True)
json.dump(cands,open('results/cg_qaoa_subinstance_candidates.json','w'),indent=1)
for c in sorted(cands,key=lambda c:c['P_prep_q05'])[:12]: print(c['source'],'m',c['m'],'opt',c['opt'],'greedy',c['greedy'],'#opt',c['n_optimal'],'P(q=.5)',round(c['P_prep_q05'],5))

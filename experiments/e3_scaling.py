"""E3: scaling of the success probability of negation-forced preparation vs uniform (Grover),
for four problem families; exact via Lemma 3.  Also classical CDCL (MiniSat) effort."""
import os as _os, sys as _sys
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..'))
_os.chdir(ROOT); _sys.path.insert(0, ROOT)
import random, sys, math, time, json, statistics, zlib, os
sys.path.insert(0, ROOT)
from negsearch.core import *
from pysat.solvers import Minisat22
fam=sys.argv[1]; grid=[int(x) for x in sys.argv[2].split(',')]
SEED=zlib.crc32(fam.encode())  # stable across runs (Python's hash() is salted)
rng=random.Random(SEED); out=[]; fn=f'results/e3_{fam}.json'
os.makedirs(f'instances/e3/{fam}',exist_ok=True)
for n in grid:
    for inst in range(10):
        t0=time.time()
        try: f=sample_satisfiable(fam,n,rng,cap=20000)
        except RuntimeError: print('skip',n); continue
        sols=enumerate_models(f,cap=20000)
        ps=[];Dmean=[];orders=[]
        for r in range(10):
            o=list(range(1,n+1)); rng.shuffle(o); orders.append(o)
            p,Ds=success_by_solutions(f,o,sols); ps.append(p); Dmean.append(statistics.mean(Ds))
        name=f'instances/e3/{fam}/n{n:03d}_i{inst}.cnf'
        with open(name,'w') as fh:
            fh.write(f'c family={fam} n={n} instance={inst} seed={SEED} models={len(sols)}\n')
            for o in orders: fh.write('c order '+' '.join(map(str,o))+'\n')
            fh.write(f'p cnf {f.n} {f.m}\n')
            for c in f.clauses: fh.write(' '.join(map(str,c))+' 0\n')
        s=Minisat22(bootstrap_with=[list(c) for c in f.clauses]); t=time.time(); s.solve(); ts=time.time()-t
        st=s.accum_stats(); s.delete()
        out.append(dict(family=fam,n=n,m=f.m,M=len(sols),instance_file=name,seed=SEED,p_grover=len(sols)/2**n,p_orders=ps,
                        p_median=statistics.median(ps),p_max=max(ps),D_mean=statistics.mean(Dmean),
                        minisat_conflicts=st.get('conflicts',0),minisat_decisions=st.get('decisions',0),minisat_secs=ts,secs=time.time()-t0))
        r=out[-1]; print(f"{fam} n={n} M={r['M']} log2 pG={math.log2(r['p_grover']):.1f} log2 pPPZ(med)={math.log2(r['p_median']):.1f} D={r['D_mean']:.1f} conf={r['minisat_conflicts']} ({r['secs']:.0f}s)",flush=True)
        json.dump(out,open(fn,'w'),indent=1)

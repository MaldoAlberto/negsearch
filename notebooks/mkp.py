import os as _os, sys as _sys
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..'))
_os.chdir(ROOT); _sys.path.insert(0, ROOT)
import itertools, sys
sys.path.insert(0, ROOT)
from negsearch.core import CNF, enumerate_models, success_by_solutions, forcing_plan
def mkp_cnf(w, v, caps, V):
    I, K = len(w), len(caps)
    var = lambda i,k: i*K + k + 1
    cls = []
    for i in range(I):                                   # each item in at most one knapsack
        for k1,k2 in itertools.combinations(range(K),2): cls.append((-var(i,k1),-var(i,k2)))
    for k,C in enumerate(caps):                          # minimal covers: forbidden overweight subsets
        for r in range(1,I+1):
            for S in itertools.combinations(range(I),r):
                if sum(w[i] for i in S)>C and all(sum(w[j] for j in S if j!=i)<=C for i in S):
                    cls.append(tuple(-var(i,k) for i in S))
    # value >= V : for every maximal set T of (item,knapsack) slots... use item-level: chosen items set
    # forbidden: chosen-item set inside an insufficient set U (value(U) < V, maximal) -> some item outside U must be packed
    for r in range(I+1):
        for U in itertools.combinations(range(I),r):
            if sum(v[i] for i in U)<V and all(sum(v[i] for i in U)+v[j]>=V for j in range(I) if j not in U):
                out=[j for j in range(I) if j not in U]
                cls.append(tuple(var(j,k) for j in out for k in range(K)))
    return CNF(I*K, cls, 'MKP', dict(w=w,v=v,caps=caps,V=V))
def brute(w,v,caps,V):
    I,K=len(w),len(caps); sols=[]
    for a in itertools.product(range(K+1),repeat=I):   # 0 = not packed, k+1 = knapsack k
        if all(sum(w[i] for i in range(I) if a[i]==k+1)<=caps[k] for k in range(K)) and sum(v[i] for i in range(I) if a[i]>0)>=V:
            sols.append(a)
    return sols
if __name__=='__main__':
    import random
    w=[3,4,2,5]; v=[4,5,3,6]; caps=[6,5]
    for V in [12,13,14,15]:
        f=mkp_cnf(w,v,caps,V); sols=enumerate_models(f); b=brute(w,v,caps,V)
        best=None
        for _ in range(200):
            o=list(range(1,f.n+1)); random.shuffle(o); p,_=success_by_solutions(f,o,sols)
            if best is None or p>best[0]: best=(p,o)
        print(f'V={V} vars={f.n} clauses={f.m} M_cnf={len(sols)} M_brute={len(b)} p_uniform={len(sols)/2**f.n:.4f} p_NF(best order)={best[0]:.4f}')

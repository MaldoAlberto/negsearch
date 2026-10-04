import numpy as np, json
rows=json.load(open('results/e1_classicality.json'))
rng=np.random.default_rng(5)
for r in rows:
    n=int(r['problem'].split('P')[-1]); it=r['iters']; T=2_000_000; ok=0
    for chunk in range(T//200_000):
        x=rng.integers(0,2,(200_000,n),dtype=np.int8)
        for _ in range(it):
            viol=(x[:,:-1]==x[:,1:])          # edge (i,i+1) violated
            x[:,:-1]^=viol.astype(np.int8)      # flip left vertex of every violated edge (simultaneous)
        ok+=int(np.all(x[:,:-1]!=x[:,1:],axis=1).sum())
    r['classical']=ok/T; r['classical_trials']=T; print(r)
json.dump(rows,open('results/e1_classicality.json','w'),indent=1)

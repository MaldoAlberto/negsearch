import json, math, numpy as np
from dynsearch import *
rows=[]
for k in range(4,25,2):
    NM=2**k; s=math.sqrt(NM)
    rows.append(dict(log2_NM=k, grover_knownM=expected_queries_grover_known_M(NM,1)/s,
                     bbht=expected_queries_bbht(NM,1,trials=6000)/s,
                     heralded_c5=expected_queries_herald(NM,1,5.0)/s,
                     classical=NM/s))
    print(rows[-1])
# robustness: fixed N=2^16, sweep M (unknown to algorithm)
N=2**16; rob=[]
for M in [1,2,4,16,64,256,1024,4096]:
    s=math.sqrt(N/M)
    rob.append(dict(M=M, heralded=expected_queries_herald(N,M,5.0)/s, bbht=expected_queries_bbht(N,M,trials=6000)/s,
                    grover_wrongM=None))
    print(rob[-1])
json.dump(dict(scaling=rows,robust=rob),open('scaling.json','w'),indent=1)

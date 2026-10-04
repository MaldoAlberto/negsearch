"""E1: measure-and-flip dynamic circuits (original proposal) equal classical randomized algorithms."""
import random, json, subprocess, sys
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.classical import expr
from qiskit_aer import AerSimulator
def original_dynamic(n, edges, n_iter):
    v=QuantumRegister(n,'v'); e=QuantumRegister(len(edges),'e')
    ce=ClassicalRegister(len(edges),'ce'); cv=ClassicalRegister(n,'cv')
    qc=QuantumCircuit(v,e,ce,cv); qc.h(v)
    for _ in range(n_iter):
        for k,(i,j) in enumerate(edges): qc.cx(v[i],e[k]); qc.cx(v[j],e[k])
        qc.measure(e,ce)
        for k,(i,j) in enumerate(edges):
            with qc.if_test(expr.logic_not(ce[k])): qc.x(v[i])
        qc.reset(e)
    qc.measure(v,cv); return qc
def classical_rule(n, edges, n_iter, trials, rng):
    w=0
    for _ in range(trials):
        x=[rng.randint(0,1) for _ in range(n)]
        for _ in range(n_iter):
            viol=[x[i]==x[j] for i,j in edges]
            for k,(i,j) in enumerate(edges):
                if viol[k]: x[i]^=1
        w+=all(x[i]!=x[j] for i,j in edges)
    return w/trials
rng=random.Random(1); stab=AerSimulator(method='stabilizer'); out=[]
for n in [10,30,60]:
    E=[(i,i+1) for i in range(n-1)]
    for it in [n//4, n//2, n, 2*n]:
        qc=original_dynamic(n,E,it); S=50000
        cnt=stab.run(transpile(qc,stab),shots=S,seed_simulator=1).result().get_counts()
        q=sum(c for b,c in cnt.items() if all(b.split()[0][n-1-i]!=b.split()[0][n-1-j] for i,j in E))/S
        cl=classical_rule(n,E,it,200000 if n<=10 else 5000,rng)
        out.append(dict(problem=f'2-col path P{n}',iters=it,quantum_stabilizer=q,classical=cl,shots=S)); print(out[-1],flush=True)
json.dump(out,open('results/e1_classicality.json','w'),indent=1)

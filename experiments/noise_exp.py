import sys, json, math, time
from qiskit import transpile, QuantumCircuit
from qiskit.circuit import Delay
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error
from qiskit_ibm_runtime.fake_provider import FakeFez
from dynsearch import *
be=FakeFez()
def noise_model(L_ns):
    nm=NoiseModel.from_backend(be)
    if L_ns>0:
        for q in range(be.num_qubits):
            pr=be.qubit_properties(q); t1=pr.t1; t2=min(pr.t2,2*t1)
            nm.add_quantum_error(thermal_relaxation_error(t1,t2,L_ns*1e-9),'delay',[q])
    return nm
def add_latency(qc, mid_clbits, L_ns, active):
    new=qc.copy_empty_like()
    for inst in qc.data:
        op=inst.operation
        if op.name=='if_else':
            blocks=[add_latency(b, mid_clbits, L_ns, [q for q in b.qubits]) for b in op.blocks]
            op=op.replace_blocks(blocks)
            new.append(op, inst.qubits, inst.clbits); continue
        new.append(op, inst.qubits, inst.clbits)
        if L_ns>0 and op.name=='measure' and inst.clbits[0] in mid_clbits:
            for q in active: new.append(Delay(L_ns,'ns'),[q])
    return new
def active_qubits(qc):
    s=set()
    for inst in qc.data:
        s.update(inst.qubits)
    return [q for q in qc.qubits if q in s]
def run(qc, L, shots=1000, seed=5):
    tq=transpile(qc,be,optimization_level=3,seed_transpiler=7)
    mid=set()
    for r in tq.cregs:
        if r.name in ('mb','h'): mid.update(r)
    tq=add_latency(tq, mid, L, active_qubits(tq))
    sim=AerSimulator(method='statevector',noise_model=noise_model(L))
    return split_counts(sim.run(tq,shots=shots,seed_simulator=seed).result().get_counts(),qc)
name=sys.argv[1]; L=float(sys.argv[2])
p={'sudoku':binary_sudoku_2x2('xor'),'3sat':planted_ksat(4,8,3,94)}[name]
N=2**p.n; M=len(p.solutions()); k=optimal_iterations(N,M); res=[]
shots=int(sys.argv[3]) if len(sys.argv)>3 else 1000
for label,qc in [('Grover textbook',textbook_grover(p,k)),('Grover coherent-rccx',grover_circuit(p,k,'coherent')),('Grover MBU-rccx',grover_circuit(p,k,'mbu'))]:
    t=time.time(); r=run(qc,L,shots)
    pv=sum(c for d,c in r if p.check_bitstring(d['out']))/shots
    res.append(dict(inst=name,L=L,method=label,p_valid=pv,cost=(k+1)/max(pv,1e-9),cert=None,p_click=None))
    print(res[-1], f'{time.time()-t:.0f}s', flush=True)
for T in [2,4,6]:
    qc=heralded_circuit(p,T,5.0,'coherent'); t=time.time(); r=run(qc,L,shots)
    cost=0; succ=0; clicks=0; cv=0
    for d,c in r:
        it=click_iteration(d['h']); ok=p.check_bitstring(d['out'])
        if it is not None:
            clicks+=c; cv+=c*ok; cost+=c*(it+1); succ+=c*ok   # click: accept certified answer (still verified classically)
        else:
            cost+=c*(T+1); succ+=c*ok
    res.append(dict(inst=name,L=L,method=f'Heralded T={T}',p_valid=succ/shots,cost=cost/max(succ,1),cert=cv/max(clicks,1),p_click=clicks/shots))
    print(res[-1], f'{time.time()-t:.0f}s', flush=True)
json.dump(res,open(f'noise_{name}_{int(L)}.json','w'),indent=1)

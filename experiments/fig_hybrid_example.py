"""Toy example of the hybrid circuit (1 period, 2 assets from po_a003_t02_orig, B=2, C=1): A_q (lean counters,
shorts-first), cost phase, XY mixer.  Checks the circuit against the exact simulation and draws it."""
import sys, numpy as np
import os; os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); sys.path.insert(0, '.')
from negsearch.portfolio import Instance, Portfolio
from negsearch.portfolio_circuits import prep_circuit_lean, cost_layer, xy_edges
from negsearch.portfolio_qaoa import Sim
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import XXPlusYYGate
inst=Instance('instances/qoblib_portfolio/po_a003_t02_orig',periods=1,assets=2)
P=Portfolio(inst,2,'1e-05',cash=100000,cs1=1,cs2=2,order='shorts_first'); P.xy_ring=False
order,act=P.plan()
names={P.q(0,d,i):('L' if d==0 else 'S')+'_'+inst.symbols[i] for d in (0,1) for i in range(2)}
for k,(d,i) in enumerate(order): print(names[P.q(0,d,i)], act[k])
S=Sim(P); q=0.5
psi=P.prep_state(q)
for idx in np.nonzero(np.abs(psi)>1e-12)[0]:
    b=[(idx>>k)&1 for k in range(P.nq)]
    print({names[k]:b[k] for k in range(P.nq)}, 'amp^2=%.3f'%abs(psi[idx])**2, 'f=',S.f[idx])
print('feasible',S.feas.sum(),'of',2**P.nq)

from qiskit import QuantumRegister, ClassicalRegister
import matplotlib; matplotlib.use('Agg')
A=prep_circuit_lean(P,q)
print(A.qregs, A.count_ops())
# named registers, drawn in processing order
lab=[names[P.q(0,d,i)] for (d,i) in order]
xs={nm:QuantumRegister(1,nm) for nm in lab}
cl=QuantumRegister(A.qregs[1].size,'cont_L'); cs=QuantumRegister(A.qregs[2].size,'cont_S')
cr=ClassicalRegister(P.nq,'m')
qc=QuantumCircuit(*xs.values(),cs,cl,cr)
pos={P.q(0,d,i):xs[names[P.q(0,d,i)]][0] for (d,i) in order}
amap=[pos[k] for k in range(P.nq)]+list(cl)+list(cs)
qc.compose(A,amap,inplace=True)
qc.barrier(label='← 1. A_q')
gamma,beta=0.6,0.5
sc=1/np.abs(S.f).max()
cost_layer(qc,[pos[k] for k in range(P.nq)],P.h*sc,{k:v*sc for k,v in P.J.items() if v},gamma)
qc.barrier(label='← 2. fase')
for L_ in xy_edges(P):
    for a,b in L_: qc.append(XXPlusYYGate(2*beta),[pos[a],pos[b]])
qc.barrier(label='← 3. XY')
for k in range(P.nq): qc.measure(pos[k],cr[k])
# verify against exact simulation (marginal on decision qubits)
nomeas=qc.remove_final_measurements(inplace=False)
sv=Statevector(nomeas)
probs=sv.probabilities_dict(qargs=[nomeas.find_bit(pos[k]).index for k in range(P.nq)])
pr=np.zeros(2**P.nq)
for kstr,v in probs.items(): pr[int(kstr,2)]+=v
ref=np.abs(S.state('xy',[gamma],[beta],S.f*sc,P.prep_state(q)))**2
print('circuit == exact:',np.allclose(pr,ref,atol=1e-9), 'P(feasible)=',pr[S.feas].sum())
fig=qc.draw('mpl',fold=26,style={'name':'clifford','fontsize':11,'subfontsize':8},plot_barriers=True,initial_state=True)
fig.savefig('figures/fig_hybrid_example_circuit.png',dpi=170,bbox_inches='tight')
print(qc.count_ops())

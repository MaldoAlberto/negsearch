import matplotlib; matplotlib.use('Agg')
from qiskit import QuantumCircuit, QuantumRegister
x=QuantumRegister(3,'x'); A1=QuantumRegister(1,'a_c1'); A0=QuantumRegister(1,'a_c0'); F1=QuantumRegister(1,'f_1'); F0=QuantumRegister(1,'f_0'); G=QuantumRegister(1,'g')
qc=QuantumCircuit(x,A1,A0,F1,F0,G)
a=[A1[0],A0[0]]; fl=[F1[0],F0[0],G[0]]
# variable x3 with clauses c1=(x1 v x2 v x3) in C1 and c0=(~x1 v x2 v ~x3) in C0
qc.x([x[0],x[1]]); qc.ccx(x[0],x[1],a[0]); qc.x([x[0],x[1]])        # a_c1 = ~x1 & ~x2
qc.x(x[1]); qc.ccx(x[0],x[1],a[1]); qc.x(x[1])                     # a_c0 = x1 & ~x2
qc.cx(a[0],fl[0]); qc.cx(a[1],fl[1])                               # f1, f0 (single clause each)
qc.barrier()
qc.cx(fl[0],x[2])
qc.x([fl[0],fl[1]]); qc.ccx(fl[0],fl[1],fl[2]); qc.x([fl[0],fl[1]])
qc.ch(fl[2],x[2])
qc.x([fl[0],fl[1]]); qc.ccx(fl[0],fl[1],fl[2]); qc.x([fl[0],fl[1]])
qc.barrier()
qc.cx(a[1],fl[1]); qc.cx(a[0],fl[0])
qc.x(x[1]); qc.ccx(x[0],x[1],a[1]); qc.x(x[1])
qc.x([x[0],x[1]]); qc.ccx(x[0],x[1],a[0]); qc.x([x[0],x[1]])
fig=qc.draw('mpl',style='bw',fold=-1,scale=0.55)
fig.savefig('figures/fig_step.pdf',bbox_inches='tight')

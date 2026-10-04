import numpy as np, random, collections
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import QFT
from qiskit.circuit.classical import expr
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
# ---- user's functions (verbatim logic) ----
def add_value(data_qubits, const):
    qc = QuantumCircuit(data_qubits); a = bin(const)[2:].zfill(data_qubits)[::-1]
    list_a=[0]*data_qubits
    for i in range(data_qubits):
        if a[i]=='1':
            k=0
            for j in range(i,data_qubits):
                list_a[data_qubits-j-1]+=np.pi/float(2**k); k+=1
    for i in range(data_qubits):
        if list_a[i]!=0: qc.p(list_a[i],i)
    return qc.to_gate(label=str(const))
def add(data_qubits, list1):
    qi=QuantumRegister(len(list1),"index"); qd=QuantumRegister(data_qubits,"data")
    qc=QuantumCircuit(qi,qd); qc.h(qd)
    for i,v in enumerate(list1): qc.append(add_value(data_qubits,v).control(1),[qi[i]]+list(qd))
    return qc.compose(QFT(num_qubits=data_qubits,inverse=True),qd)
def add_list_d(data_qubits, list1, iters=3):
    ni=len(list1); qi=QuantumRegister(ni,"index"); qd=QuantumRegister(data_qubits,"data")
    c=ClassicalRegister(ni); qc=QuantumCircuit(qi,qd,c); qc.h(qi)
    for _ in range(iters):
        qc.append(add(data_qubits,list1),qi[:]+qd[:]); qc.measure(qd,c)
        with qc.if_test(expr.logic_not(c[0])): qc.x(0)
        with qc.if_test(expr.logic_not(c[1])): qc.x(1)
        with qc.if_test(expr.logic_not(c[2])): qc.x(3)
        with qc.if_test((c[3],1)): qc.x(7)
        with qc.if_test((c[2],1)): qc.x(6)
        with qc.if_test((c[1],1)): qc.x(5)
        with qc.if_test((c[0],1)): qc.x(4)
    qc.measure(qi,c); return qc
L=[1,2,3,4]; D=len(bin(sum(L))[2:])  # 4
# ---- 1) is the adder a permutation of basis states? ----
A=add(D,L); table={}; perm=True
for idx in range(16):
    sv=Statevector.from_int(idx,2**(4+D)).evolve(A); p=sv.probabilities()
    j=int(np.argmax(p)); perm &= p[j]>1-1e-9; table[idx]=j>>4
print('adder is a permutation on basis states:',perm,' sums:',[table[i] for i in range(16)])
# ---- 2) quantum circuit ----
qc=add_list_d(D,L); sim=AerSimulator()
cnt=sim.run(transpile(qc,sim),shots=20000,seed_simulator=3).result().get_counts()
q={k:v/20000 for k,v in cnt.items()}
# ---- 3) classical emulation of the same rules ----
cl=collections.Counter()
for _ in range(200000):
    idx=[random.randint(0,1) for _ in range(4)]; data=[0]*4
    for _ in range(3):
        s=table[sum(b<<i for i,b in enumerate(idx))]; d=[(s>>k)&1 for k in range(4)]  # measured data
        if not d[0]: idx[0]^=1
        if not d[1]: idx[1]^=1
        if not d[2]: idx[3]^=1
        data=[0]*4   # the c[k]==1 -> X(data k) rules reset data to |0>
    cl[''.join(str(b) for b in idx[::-1])]+=1
c={k:v/200000 for k,v in cl.items()}
keys=sorted(set(q)|set(c),key=lambda k:-q.get(k,0))
print('state  quantum  classical')
for k in keys[:8]: print(k, f'{q.get(k,0):.3f}   {c.get(k,0):.3f}')
tv=0.5*sum(abs(q.get(k,0)-c.get(k,0)) for k in keys); print('total variation distance:',round(tv,4))

"""Optimised Dicke-state preparation (same state as negsearch.dicke.dicke, exactly), with hand-decomposed gates.
cRY(theta)  = RY(t/2) CX RY(-t/2) CX                         (2 CX)
ccRY(theta) = [RY(t/4) CX(b,t) RY(-t/4) CX(a,t)] x2          (4 CX, exact; X RY(phi) X = RY(-phi) makes the angles add only when a=b=1)
The SCS block is CX, cRY/ccRY, CX; neighbouring CX are cancelled by the transpiler."""
import math
from qiskit import QuantumCircuit

def cry(qc, th, c, t):
    qc.ry(th / 2, t); qc.cx(c, t); qc.ry(-th / 2, t); qc.cx(c, t)

def ccry(qc, th, a, b, t):
    for _ in range(2):
        qc.ry(th / 4, t); qc.cx(b, t); qc.ry(-th / 4, t); qc.cx(a, t)

def ccry3(qc, th, a, b, t):
    """3-CX version of ccRY, exact only on the inputs the SCS block produces (relative phases / don't-cares);
    found by least squares on the whole Dicke circuit, then fixed to closed form, verified against the exact state for n <= 14."""
    qc.ry(-th / 4 + math.pi / 2, t); qc.cx(b, t); qc.ry(th / 4, t); qc.cx(a, t); qc.ry(-th / 4, t); qc.cx(b, t); qc.ry(th / 4 - math.pi / 2, t)

MODE = {'ccry': ccry}
import os as _os
REL = {'on': _os.environ.get('NS_REL_PHASE') == '1'}

def givens(qc, th, a, b):
    """CX(b,a) cRY(th; a->b) CX(b,a) as a two-qubit unitary, synthesised with 2 CX (it is a rotation in span{|01>,|10>})."""
    from qiskit.quantum_info import Operator
    from qiskit.synthesis import TwoQubitBasisDecomposer
    from qiskit.circuit.library import CXGate
    sub = QuantumCircuit(2); sub.cx(1, 0); sub.cry(th, 0, 1); sub.cx(1, 0)
    qc.compose(TwoQubitBasisDecomposer(CXGate())(Operator(sub)), [a, b], inplace=True)

GIV = {'on': True}

def _scs(qc, q, n, k):
    if GIV['on']: givens(qc, 2 * math.acos(math.sqrt(1 / n)), q[n - 1], q[n - 2])
    else:
        qc.cx(q[n - 2], q[n - 1]); cry(qc, 2 * math.acos(math.sqrt(1 / n)), q[n - 1], q[n - 2]); qc.cx(q[n - 2], q[n - 1])
    for l in range(2, k + 1):
        qc.cx(q[n - l - 1], q[n - 1]); MODE['ccry'](qc, 2 * math.acos(math.sqrt(l / n)), q[n - 1], q[n - l], q[n - l - 1]); qc.cx(q[n - l - 1], q[n - 1])

def dicke_opt(qc, q, k):
    n = len(q)
    if k == 0: return
    if k == n:
        for x in q: qc.x(x)
        return
    for x in q[n - k:]: qc.x(x)
    for m in range(n, k, -1): _scs(qc, q, m, k)
    for m in range(k, 1, -1): _scs(qc, q, m, m - 1)

def dicke_opt_circuit(n, k, cheap=True):
    MODE['ccry'] = ccry3 if cheap else ccry
    qc = QuantumCircuit(n); dicke_opt(qc, list(range(n)), k); return qc


def symmetric_block_prep_opt(m, l, s, cheap=True):
    """Same state as negsearch.symlower.symmetric_block_prep (uniform over the C(m,l) C(m-l,s) arrangements), with
    (i) the optimised Dicke blocks, (ii) the decompression chain pruned to the range that can hold a nonzero qubit
    (compressed register holds at most m-l-1+min(i,l) ... ), (iii) the role with the cheaper chain chosen as the controls."""
    from math import comb
    def chain_len(a, b):    # a = controls (count), b = compressed species count: cswaps needed
        return sum(max(0, min(m - 2, m - a - 1 + min(i, a)) - i + 1) for i in range(m))
    swap_roles = chain_len(s, l) < chain_len(l, s)
    ctrl, comp = (1, 0) if swap_roles else (0, 1)          # qubit parity inside a unit: 0 = long, 1 = short
    a, b = (s, l) if swap_roles else (l, s)
    qc = QuantumCircuit(2 * m)
    C = [2 * j + ctrl for j in range(m)]; P = [2 * j + comp for j in range(m)]
    MODE['ccry'] = ccry3 if cheap else ccry
    if a: dicke_opt(qc, C, a)
    if b: dicke_opt(qc, P[:m - a], b)
    if a and b:
        for i in range(m):
            for k in range(min(m - 2, m - a - 1 + min(i, a)), i - 1, -1):
                if REL['on']:       # relative-phase Toffoli inside the swap: same support and magnitudes, phases in quarter turns
                    from qiskit.circuit.library import RCCXGate
                    qc.cx(P[k + 1], P[k]); qc.append(RCCXGate(), [C[i], P[k], P[k + 1]]); qc.cx(P[k + 1], P[k])
                else:
                    qc.cswap(C[i], P[k], P[k + 1])
    return qc

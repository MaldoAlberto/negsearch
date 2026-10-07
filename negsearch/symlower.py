"""Symmetry-aware preparation of A_q for a block of m interchangeable units, each unit = (long_j, short_j) with long/short exclusive,
conditioned on l longs and s shorts.  Feasible set F = {arrangements}, |F| = C(m,l) C(m-l,s); uniform amplitudes (any support-F state is valid).
Construction, no auxiliary register and no uncomputation:
  1. Dicke(m, l) on the long qubits                      -> uniform set of long positions
  2. Dicke(m-l, s) on the first m-l short qubits         -> uniform choice of s shorts among the m-l free slots (compressed)
  3. decompression: for i = 0..m-1, controlled on long_i, shift the short qubits i..m-1 one place right (chain of controlled swaps),
     which inserts a 0 at every long position and spreads the compressed bits over the free positions in order.
Layout [long_0, short_0, long_1, short_1, ...] (same as the rest of the code)."""
from qiskit import QuantumCircuit
from .dicke import dicke


def symmetric_block_prep(m, l, s):
    import os
    if os.environ.get('NS_OPT_PREP') == '1':      # optimised Dicke blocks + pruned decompression (negsearch.dicke_opt), same state
        from .dicke_opt import symmetric_block_prep_opt
        return symmetric_block_prep_opt(m, l, s)
    qc = QuantumCircuit(2 * m)
    L = [2 * j for j in range(m)]; S = [2 * j + 1 for j in range(m)]
    if l: dicke(qc, L, l)
    if s: dicke(qc, S[:m - l], s)
    if l and s:
        for i in range(m):
            for k in range(m - 2, i - 1, -1):
                qc.cswap(L[i], S[k], S[k + 1])
    return qc

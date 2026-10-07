"""Resynthesis of the controlled move used in the pruned decompression chain, exploiting that the chain never feeds it control=1 with target=1.
Target (qubits c,a,b), exact including phases, on the 6 reachable basis inputs: c=0 -> identity (any a,b); c=1,b=0 -> |c=1, a'=0, b'=a>.
Template: layers of generic single-qubit unitaries and CX gates (all CX placements), optimised by least squares.  Prints the shortest template that reaches residual < 1e-12.
Counts only."""
import sys, itertools, math, numpy as np
from scipy.optimize import least_squares
from multiprocessing import Pool
def ry(t): c, s = math.cos(t / 2), math.sin(t / 2); return np.array([[c, -s], [s, c]], complex)
def rz(t): return np.array([[np.exp(-1j * t / 2), 0], [0, np.exp(1j * t / 2)]])
def u3(a, b, c): return rz(b) @ ry(a) @ rz(c)
def kron(*ms):
    o = np.eye(1)
    for m in ms: o = np.kron(o, m)
    return o
def cx(c, t):
    M = np.zeros((8, 8))
    for i in range(8):
        b = [(i >> (2 - q)) & 1 for q in range(3)]
        if b[c]: b[t] ^= 1
        M[(b[0] << 2) | (b[1] << 1) | b[2], i] = 1
    return M
reach = [(c, a, b) for c in (0, 1) for a in (0, 1) for b in (0, 1) if not (c == 1 and b == 1)]
def idx(s): return (s[0] << 2) | (s[1] << 1) | s[2]
cols = [idx(s) for s in reach]
T = np.zeros((8, len(reach)), complex)
for j, (c, a, b) in enumerate(reach): T[idx((c, a, b) if c == 0 else (1, 0, a)), j] = 1
def build(p, pat):
    M = np.eye(8, dtype=complex)
    for L in range(len(pat) + 1):
        M = kron(*[u3(*p[9 * L + 3 * q: 9 * L + 3 * q + 3]) for q in range(3)]) @ M
        if L < len(pat): M = cx(*pat[L]) @ M
    return M
def run(args):
    pat, seed = args; rng = np.random.default_rng(seed); n = 9 * (len(pat) + 1); best = 1e9
    for _ in range(8):
        r = least_squares(lambda p: (lambda D: np.concatenate([D.real.ravel(), D.imag.ravel()]))(build(p, pat)[:, cols] - T), rng.uniform(-math.pi, math.pi, n), max_nfev=250)
        best = min(best, r.cost)
        if best < 1e-14: break
    return pat, best
if __name__ == '__main__':
    pairs = [(a, b) for a in range(3) for b in range(3) if a != b]
    for ncx in range(int(sys.argv[1]), int(sys.argv[2]) + 1):
        pats = [p for p in itertools.product(pairs, repeat=ncx) if all(p[i] != p[i + 1] for i in range(ncx - 1))]
        with Pool(2) as P: out = P.map(run, [(p, i) for i, p in enumerate(pats)], chunksize=2)
        out.sort(key=lambda r: r[1]); print(ncx, len(pats), 'best residual', out[0][1], out[0][0], flush=True)
        if out[0][1] < 1e-12: break

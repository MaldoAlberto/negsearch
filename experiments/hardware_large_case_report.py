"""Analysis of a hardware run of experiments/hardware_large_case.py (results/hardware_large_case<tag>_raw.json).
For each circuit: fraction of samples that satisfy ALL constraints (checked on the 24 or 8 decision bits), Wilson 95% interval, the uniform-state
share of feasible strings (what an unconstrained start would give), mean objective of the feasible samples, best objective found and, for QAOA
circuits, comparison with the same interface's preparation circuit.  Usage: python experiments/hardware_large_case_report.py results/hardware_large_case_n24_raw.json"""
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT)
import sys; sys.path.insert(0, ROOT)
import json, math
import numpy as np
from negsearch.hw_large import qubo_cost


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - r, c + r)


def feasible(bits, A, T, sectors, cap=1, Dmax=3, Bmax=3):
    Q = int(round(0.6 * T * Bmax)); tot = 0; off = np.cumsum([0] + list(sectors))
    for t in range(T):
        lon = [bits[((t * A + i) * 2)] for i in range(A)]; sho = [bits[((t * A + i) * 2) + 1] for i in range(A)]
        L, S = sum(lon), sum(sho); tot += L + S
        if not (0 <= L - S <= Dmax and L + S <= Bmax) or any(a and b for a, b in zip(lon, sho)):
            return False
        for si in range(len(sectors)):
            if sum(lon[off[si]:off[si + 1]]) > cap:
                return False
    return tot <= Q


def main(path):
    d = json.load(open(path)); m = d['meta']; A, T, S = m['A'], m['T'], m['sectors']; n = 2 * A * T
    h, J = qubo_cost(A, T, m['seed'])
    print(f"backend {d['backend']}, shots {d['shots']}, n={n}")
    prep = {}
    for r in d['results']:
        counts = r['counts']; tot = sum(counts.values()); k = 0; costs = []
        for s, v in counts.items():
            bits = [int(c) for c in s.replace(' ', '')[::-1]][:n]
            if feasible(bits, A, T, S):
                k += v; x = np.array(bits); c = float(x @ h + sum(w * x[i] * x[j] for (i, j), w in J.items())); costs += [c] * v
        lo, hi = wilson(k, tot)
        mean = np.mean(costs) if costs else float('nan'); best = min(costs) if costs else float('nan')
        print(f"interface {r['iface']} {r['name']:6s} feasible {k/tot:.3f} [{lo:.3f},{hi:.3f}]  mean cost {mean:.3f}  best {best:.3f}  (optimum {m['opt']:.3f})")
        if r['name'] == 'prep':
            prep[str(r['iface'])] = (k / tot, mean)
        elif str(r['iface']) in prep:
            print(f"    QAOA minus preparation: feasible {k/tot - prep[str(r['iface'])][0]:+.3f}, mean cost {mean - prep[str(r['iface'])][1]:+.3f}")


if __name__ == '__main__':
    main(sys.argv[1])

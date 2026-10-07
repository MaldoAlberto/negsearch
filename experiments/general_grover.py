"""Generality test 1 (Grover / amplitude amplification): the SAME pipeline on seven constraint families.
For each instance: standard Grover on |+>^n (oracle must mark feasible AND optimal)  vs  amplitude amplification
on the negation-forced state A_q|0> (oracle only needs the objective threshold).  Exact numerics.
Writes results/general_grover.json."""
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT)
import sys; sys.path.insert(0, ROOT)
import json, math
import numpy as np
from negsearch.automaton import build_automaton, mu_q, best_k, grover_curve
from negsearch import families as fam

QS = np.round(np.arange(0.05, 0.96, 0.05), 2)
RANKS = [0.0, 0.1]          # 0: optimal solutions;  0.1: best 10% of the feasible cost range


def instances(seeds=range(5)):
    out = []
    for s in seeds:
        out += [fam.cardinality_qp(10, 4, s), fam.knapsack(10, 10 + s), fam.mis(10, 0.35, 20 + s),
                fam.set_packing(10, 7, 30 + s), fam.exact_cover(10, 6, 40 + s), fam.colouring(5, 50 + s),
                fam.assignment_qap(3, 60 + s)]
    out += [fam.assignment_qap(4, 7)]            # 16 qubits
    return [P for P in out if P.nF >= 2]       # drop empty / trivial feasible sets


def kind(P):
    return P.name.split(' (')[0]


def main():
    rows = []
    for P in instances():
        A = build_automaton(P.feas, P.n)
        N = 2 ** P.n
        for rank in RANKS:
            G = P.good(rank)
            m = int(G.sum())
            a_unif = m / N
            k_std = best_k(a_unif)
            p_std = grover_curve(np.ones(N) / math.sqrt(N), G, k_std)[0][-1]
            best = None
            for q in QS:
                mu, D = mu_q(A, float(q))
                a = float(mu[G].sum())
                if best is None or a > best[0]:
                    best = (a, float(q))
            mu5, D5 = mu_q(A, 0.5)
            a5 = float(mu5[G].sum())
            k5 = best_k(a5)
            kq = best_k(best[0])
            rows.append(dict(
                family=kind(P), name=P.name, n=P.n, nF=P.nF, nG=m, rank=rank, width=A.w, widths=A.width,
                frac_feasible=P.nF / N,
                a_unif=a_unif, k_std=k_std, p_std=p_std,
                a_half=a5, k_half=k5, p_half=grover_curve(np.sqrt(mu5), G, k5)[0][-1],
                q_best=best[1], a_best=best[0], k_best=kq,
                p_best=grover_curve(np.sqrt(mu_q(A, best[1])[0]), G, kq)[0][-1],
                cls_unif=1 / a_unif, cls_half=1 / a5, cls_best=1 / best[0],
                maxD=int(D5[P.feas].max()), meanD=float(D5[P.feas].mean())))
    json.dump(rows, open('results/general_grover.json', 'w'), indent=1)
    # summary
    print(f"{'family':24s} n   |F|   |G| rank width  a_unif   a(q=.5)  a(q*)   q*   k_std k_.5 k_q*   spd(.5) spd(q*)")
    for fam_name in dict.fromkeys(r['family'] for r in rows):
        for rank in RANKS:
            R = [r for r in rows if r['family'] == fam_name and r['rank'] == rank and r['n'] == 10 or
                 (r['family'] == fam_name and r['rank'] == rank and fam_name.startswith('quadratic') and r['n'] == 9)]
            if not R:
                continue
            med = lambda k: float(np.median([r[k] for r in R]))
            print(f"{fam_name[:24]:24s} {R[0]['n']:2d} {med('nF'):6.0f} {med('nG'):5.1f} {rank:4.1f} {med('width'):4.0f} "
                  f"{med('a_unif'):9.2e} {med('a_half'):8.2e} {med('a_best'):8.2e} {med('q_best'):4.2f} "
                  f"{med('k_std'):5.0f} {med('k_half'):4.0f} {med('k_best'):4.0f}  {med('k_std')/max(1,med('k_half')):6.1f}x {med('k_std')/max(1,med('k_best')):6.1f}x")


if __name__ == '__main__':
    main()

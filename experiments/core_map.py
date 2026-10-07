"""Map of the residual quantum core after the exact classical reduction (interface conditioning), for portfolio families of growing size and
looseness.  Nothing is compiled here: it counts, for interface tuples sampled with their weight,
  free variables (not fixed by the interface), log2 |F_core| (feasible strings of the residual problem), non-trivial blocks, objective couplings among free variables.
Families differ in how loose the local constraints are (cap = longs per sector, Bmax = gross positions per period, Dmax = net exposure):
  tight  cap 1, Bmax 3, Dmax 3   (the family used so far)
  medium cap 2, Bmax 6, Dmax 4
  loose  cap 4, Bmax 12, Dmax 6
Writes results/core_map.json."""
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT)
import sys; sys.path.insert(0, ROOT)
import json, math, itertools
import numpy as np

FAMILIES = {'tight': dict(cap=1, Bmax=3, Dmax=3), 'medium': dict(cap=2, Bmax=6, Dmax=4), 'loose': dict(cap=4, Bmax=12, Dmax=6)}
SIZES = [(4, 2), (6, 4), (8, 4), (12, 4), (12, 8), (16, 8), (24, 8)]        # (assets per sector=m, periods)  with 2 sectors -> A = 2m assets, n = 2*A*T


def nF(m, l, s): return math.comb(m, l) * math.comb(m - l, s)


def block_free(m, l, s):
    """number of free x-variables (long+short) of a block with counts (l,s)"""
    if nF(m, l, s) == 1: return 0
    lf = 0 if l in (0, m) else m
    sf = 0 if (s == 0 or l == m) else m
    return lf + sf


def period_options(m, nsec, p):
    opts = []
    for choice in itertools.product([(l, s) for l in range(p['cap'] + 1) for s in range(p['Bmax'] + 1) if l + s <= m], repeat=nsec):
        L = sum(c[0] for c in choice); S = sum(c[1] for c in choice)
        if 0 <= L - S <= p['Dmax'] and L + S <= p['Bmax']:
            opts.append((choice, L + S, math.prod(nF(m, *c) for c in choice)))
    return opts


def study(fam, m, T, nsec=2, ntup=200, seed=0):
    p = FAMILIES[fam]; rng = np.random.default_rng(seed)
    opts = period_options(m, nsec, p)
    if not opts: return None
    w = np.array([o[2] for o in opts], float); w /= w.sum()
    Q = int(round(0.6 * T * p['Bmax'])); rows = []
    for _ in range(ntup):
        rem = Q; pick = []
        for t in range(T):
            ok = [i for i, o in enumerate(opts) if o[1] <= rem]
            if not ok: break
            ww = w[ok] / w[ok].sum(); o = opts[ok[rng.choice(len(ok), p=ww)]]; pick.append(o); rem -= o[1]
        if len(pick) < T: continue
        free = 0; logF = 0.0; nontriv = 0
        for o in pick:
            for (l, s) in o[0]:
                f = block_free(m, l, s); free += f; logF += math.log2(nF(m, l, s)); nontriv += f > 0
        rows.append((free, logF, nontriv))
    if not rows: return None
    r = np.array(rows)
    n = 2 * m * nsec * T
    return dict(family=fam, m=m, T=T, n=n, free_mean=float(r[:, 0].mean()), free_max=int(r[:, 0].max()), log2F_mean=float(r[:, 1].mean()),
                log2F_max=float(r[:, 1].max()), nontrivial_blocks=float(r[:, 2].mean()), blocks=nsec * T, zz_free=float((r[:, 0] ** 2 / 2).mean()), sampled=len(rows))


if __name__ == '__main__':
    out = []
    print(f"{'family':7s} {'n':>4s} {'free':>6s} {'max':>4s} {'log2|F|':>8s} {'nontriv/blocks':>15s} {'ZZ among free':>14s}")
    for fam in FAMILIES:
        for m, T in SIZES:
            r = study(fam, m, T)
            if r is None: continue
            out.append(r)
            print(f"{fam:7s} {r['n']:4d} {r['free_mean']:6.1f} {r['free_max']:4d} {r['log2F_mean']:8.1f} {r['nontrivial_blocks']:7.1f}/{r['blocks']:<6d} {r['zz_free']:14.0f}", flush=True)
    json.dump(out, open('results/core_map.json', 'w'), indent=1)

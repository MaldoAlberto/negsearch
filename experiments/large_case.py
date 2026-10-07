"""Large multi-constraint case (multi-period long/short portfolio with sector caps, exclusivity and a global trading budget).
For several sizes: constraints, variables, automaton width for two variable orders, qubits and (estimated) CZ of A_q, the share of the
uniform state that is feasible (exact), qubits of the slack-QUBO encoding, and the effect of classical preprocessing:
 (1) variable order, (2) conditioning on the interface (per-period position counts drawn classically -> independent blocks).
CZ are estimates: per multi-controlled gate with Qiskit's default ancilla-free synthesis, before routing (routing roughly doubles them).
Writes results/large_case.json."""
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT)
import sys; sys.path.insert(0, ROOT)
import json, math, time, warnings; warnings.filterwarnings('ignore')
import numpy as np
from negsearch.large_case import case_automaton, block_automaton, aq_cost, sector_block_automaton, onehot_cost
from negsearch.structural import count_feasible

CASES = [(6, 2, [3, 3]), (8, 3, [4, 4]), (10, 4, [5, 5]), (10, 6, [5, 5]), (12, 8, [4, 4, 4])]
cap, Dmax, Bmax = 1, 3, 3
out = []
print(f"{'A,T':7s} {'n':>4s} {'constr':>6s} {'width':>6s} {'qubits':>6s} {'#mcg':>6s} {'CZ(A_q)':>9s} {'uniform feas.':>14s} {'QUBO qubits':>11s} | blocks: m->CZ  max/parallel  E[sum]")
for A, T, sectors in CASES:
    t0 = time.time()
    Q = int(round(0.6 * T * Bmax))
    Aut, var = case_automaton(A, T, sectors, cap, Dmax, Bmax, Q, 'period')
    n = Aut.n
    nF = count_feasible(Aut)
    q, ng, cz = aq_cost(Aut)
    constraints = T * (2 + len(sectors) + A) + 1
    slack = T * (2 + 2 + len(sectors) * 1) + math.ceil(math.log2(Q + 1))
    # classical preprocessing (2): conditioned blocks, interface = number of positions m_t of each period
    blocks = {}
    for m in range(0, Bmax + 1):
        B = block_automaton(A, sectors, m, cap, Dmax, Bmax)
        if B.width[B.n] == 0 or count_feasible(B) == 0:
            continue
        qb, ngb, czb = aq_cost(B)
        blocks[m] = dict(width=B.w, qubits=qb, cz=czb, nF=count_feasible(B))
    mx = max(b['cz'] for b in blocks.values()); mean = float(np.mean([b['cz'] for b in blocks.values()]))
    # classical preprocessing (3): condition on the per-sector counts (l, s) too -> tiny independent blocks, one-hot registers
    sect = {}
    for size in sorted(set(sectors)):
        for l in range(0, cap + 1):
            for s_ in range(0, Bmax + 1):
                B = sector_block_automaton(size, l, s_)
                if B.width[B.n] == 0 or count_feasible(B) == 0:
                    continue
                qo, ngo, czo = onehot_cost(B); qb2, ngb2, czb2 = aq_cost(B)
                sect[(size, l, s_)] = dict(width=B.w, qubits_oh=qo, cz_oh=czo, cz_bin=czb2)
    sect_max = max(v['cz_oh'] for v in sect.values()); sect_mean = float(np.mean([v['cz_oh'] for v in sect.values() if v['cz_oh'] > 0]))
    sect_q = float(np.mean([v['qubits_oh'] for v in sect.values()])); nblocks = T * len(sectors)
    row = dict(A=A, T=T, n=n, constraints=constraints, width=Aut.w, qubits=q, n_mcg=ng, cz=cz, log2_F=math.log2(nF), log2_uniform_feas=math.log2(nF) - n,
               qubo_qubits=n + slack, blocks=blocks, block_cz_max=mx, cz_conditioned_sum_mean=T * mean, sector_blocks={str(k): v for k, v in sect.items()},
               sector_cz_max_oh=sect_max, sector_cz_total_typ_oh=nblocks * sect_mean, sector_qubits_total_typ_oh=nblocks * sect_q, secs=time.time() - t0)
    out.append(row)
    print(f"{A},{T:<5d} {n:4d} {constraints:6d} {Aut.w:6d} {q:6d} {ng:6d} {cz:9d} {'2^%.1f' % row['log2_uniform_feas']:>14s} {n+slack:11d} | {','.join(f'{m}:{b['cz']}' for m,b in blocks.items())}  {mx}  {T*mean:.0f}   | sectors(one-hot): max {sect_max} total~{nblocks*sect_mean:.0f} qubits~{nblocks*sect_q:.0f} ({row['secs']:.0f}s)", flush=True)
# (1) variable order, small sizes
print('\nvariable order (width of the exact automaton)')
orders = []
for A, T, sectors in [(4, 2, [2, 2]), (4, 3, [2, 2]), (5, 2, [3, 2])]:
    r = {}
    for order in ('period', 'asset'):
        Aut, _ = case_automaton(A, T, sectors, cap, Dmax, Bmax, int(round(0.6 * T * Bmax)), order)
        r[order] = Aut.w
    r.update(A=A, T=T); orders.append(r); print(f"A={A} T={T}: period-major {r['period']}  asset-major {r['asset']}", flush=True)
json.dump(dict(cases=out, orders=orders), open('results/large_case.json', 'w'), indent=1)

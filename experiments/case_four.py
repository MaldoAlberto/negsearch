"""ONE case, four methods, ibm_fez topology (156 qubits).  Problem: multi-period long/short portfolio with n=64 binary variables (A=16 assets, T=2 periods, 2 sectors of 8),
sector caps, gross/net limits per period, global budget.  Classical front end (exact): sample an interface tuple, eliminate the variables it fixes; what remains is the quantum core.
Methods, all on the SAME core and the SAME reduced dense objective, p=1, angles trained in ideal simulation:
  ours-QAOA     symmetric preparation + objective phase + pair-exchange mixers          (support F, feasible by construction)
  ours-Grover   one iteration A^dag, reflect |0>, A is compiled; ideal amplification of the optimum is computed (iterations needed in the fault-tolerant regime)
  Fourier-LCU   single-basis-circuit variant: uniform start, objective + exclusivity phase + one Rz(theta_c) per count constraint (theta_c trained), X mixer
  unbalanced    penalty -l1 g + l2 g^2 for each count constraint (both directions) + exclusivity penalty, uniform start, X mixer
Hardware numbers are COMPILED ESP on ibm_fez; the 'noisy' distribution is the global-depolarising model  ESP*ideal + (1-ESP)*uniform  (a model, not a measurement).
Writes results/case_four.json."""
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT)
import sys; sys.path.insert(0, ROOT)
import json, math, itertools, warnings; warnings.filterwarnings('ignore')
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from negsearch.hw_large import qubo_cost, ising_from_qubo
from negsearch.depth_study import phase_poly
import experiments.prune_circuit as P
from experiments.four_methods import sample
from experiments.core_map import block_free
from experiments.qaoa_realistic import best as compile_best

import os
m = int(os.environ.get('CF_M', 8)); T = int(os.environ.get('CF_T', 2)); A = 2 * m; S = [m, m]; n = 2 * A * T
FAM = os.environ.get('CF_FAM', 'tight'); TARGET_FREE = (int(os.environ.get('CF_FMIN', 14)), int(os.environ.get('CF_FMAX', 16)))
TAG = os.environ.get('CF_TAG', 'case_four')


def compact(qc):
    used = sorted({qc.find_bit(q).index for i in qc.data for q in i.qubits}); mp = {q: k for k, q in enumerate(used)}
    out = QuantumCircuit(len(used))
    for i in qc.data: out.append(i.operation, [mp[qc.find_bit(q).index] for q in i.qubits])
    return out, used


def core_problem(seed_obj=5, seed_tuple=3):
    h0, J0 = qubo_cost(A, T, seed_obj)
    from experiments.core_map import period_options, nF as nFb, FAMILIES
    p = FAMILIES[FAM]; opts = period_options(m, 2, p); Q = int(round(0.6 * T * p['Bmax'])); cands = []
    rr = np.random.default_rng(int(os.environ.get("CF_TUPLE_SEED", 7)))
    fo = np.array([sum(block_free(m, l, s) for l, s in o[0]) for o in opts], float); wo = np.exp(-float(os.environ.get('CF_BIAS', 0)) * fo); wo /= wo.sum()
    for _ in range(300000):
        combo = [opts[i] for i in rr.choice(len(opts), T, p=wo)]
        if sum(o[1] for o in combo) > Q: continue
        blocks = [c for o in combo for c in o[0]]
        f = sum(block_free(m, l, s) for l, s in blocks)
        if TARGET_FREE[0] <= f <= TARGET_FREE[1] and sum(math.log2(nFb(m, l, s)) for l, s in blocks) <= float(os.environ.get('CF_LOGFMAX', 99)): cands.append((sum(math.log2(nFb(m, l, s)) for l, s in blocks), f, blocks))
    cands.sort(key=lambda c: -c[0]); f, blocks = cands[0][1], cands[0][2]
    iface = [tuple(blocks[t * 2:(t + 1) * 2]) for t in range(T)]
    bl = P.blocks_of(A, T, S, iface); forced = P.forced_values(bl); h1, J1 = P.eliminate(h0, J0, forced)
    free = sorted({v for b in bl for v in b['xq'] if v not in forced}); col = {v: k for k, v in enumerate(free)}
    # feasible strings of the core (product over blocks of the free columns)
    per = []
    for b in bl:
        fc = [p for p, v in enumerate(b['xq']) if v in col]
        if fc: per.append((b, fc, sorted({tuple(r[fc]) for r in b['F']})))
    strs = []
    for combo in itertools.product(*[p[2] for p in per]):
        x = np.zeros(len(free), int)
        for (b, fc, _), r in zip(per, combo):
            for p, val in zip(fc, r): x[col[b['xq'][p]]] = val
        strs.append(x)
    Fm = np.array(strs)
    return dict(iface=iface, bl=bl, forced=forced, h1=h1, J1=J1, h0=h0, J0=J0, free=free, col=col, F=Fm, per=per)


def cost_of(X, h, J, col):
    c = np.zeros(len(X))
    for v, k in col.items(): c += h[v] * X[:, k]
    for (i, j), w in J.items():
        if i in col and j in col: c += w * X[:, col[i]] * X[:, col[j]]
    return c


def allstrings(f): return ((np.arange(2 ** f)[:, None] >> np.arange(f)) & 1)


def main():
    cp = core_problem(); free = cp['free']; col = cp['col']; f = len(free); Fm = cp['F']; nF = len(Fm)
    print(f"n={n} variables, interface {cp['iface']}, free qubits f={f}, |F_core|={nF}", flush=True)
    X = allstrings(f); idx_of = {tuple(r): i for i, r in enumerate(Fm)}
    cF = cost_of(Fm, cp['h1'], cp['J1'], col); cmin, cmax = cF.min(), cF.max(); norm = lambda c: (c - cmin) / (cmax - cmin)
    cA = cost_of(X, cp['h1'], cp['J1'], col)
    featF = np.zeros(2 ** f, bool); codeF = (Fm * (1 << np.arange(f))).sum(1); featF[codeF] = True
    opt_mask = featF & (np.abs(cA - cmin) < 1e-9)
    # constraints (groups) for the baselines: count of free vars of each (block, long/short) group minus the number of ones already forced
    groups = P.groups_of(cp['bl'], cp['forced'])           # list of (free_vars, target)
    G = np.array([sum(X[:, col[v]] for v in vs) - tgt for vs, tgt in groups], float)           # equality targets (0 when satisfied)
    excl = np.zeros(2 ** f)
    for b in cp['bl']:
        for j in range(b['size']):
            a, c = b['xq'][2 * j], b['xq'][2 * j + 1]
            if a in col and c in col: excl += X[:, col[a]] * X[:, col[c]]
    cn = norm(cA)

    def metrics(Pr):
        fe = float(Pr[featF].sum()); q = float((Pr * featF * (1 - cn)).sum())
        return dict(feas=fe, quality_feasible=(q / fe if fe > 0 else 0.0), popt=float(Pr[opt_mask].sum()))

    def rx_layer(psi, b):
        c, s = math.cos(b), -1j * math.sin(b); t = psi.reshape([2] * f)
        for ax in range(f):
            a0 = np.take(t, 0, axis=ax); a1 = np.take(t, 1, axis=ax); t = np.stack([c * a0 + s * a1, s * a0 + c * a1], axis=ax)
        return t.reshape(-1)

    out = dict(n=n, free=f, nF=nF, log2F=math.log2(nF), iface=[[list(c) for c in p] for p in cp['iface']])
    rng = np.random.default_rng(0)

    # ---------- ours QAOA: simulation inside F
    pairs = []
    for b in cp['bl']:
        if len(b['F']) == 1: continue
        for j in range(b['size'] - 1):
            k = j + 1; cols = [(col[b['xq'][2 * j + d]], col[b['xq'][2 * k + d]]) for d in (0, 1) if b['xq'][2 * j + d] in col]
            perm = np.empty(nF, int)
            for i, r in enumerate(Fm):
                r2 = r.copy()
                for a_, c_ in cols: r2[a_], r2[c_] = r[c_], r[a_]
                perm[i] = idx_of[tuple(r2)]
            pairs.append(perm)

    def qaoa_state(g, bt):
        psi = np.ones(nF, complex) / math.sqrt(nF); psi = psi * np.exp(-1j * g * cF)
        for perm in pairs: psi = math.cos(bt) * psi - 1j * math.sin(bt) * psi[perm]
        return psi
    best = None
    for r in range(30):
        x0 = [rng.uniform(-2, 2), rng.uniform(0, 1.6)]
        o = minimize(lambda x: float((np.abs(qaoa_state(*x)) ** 2 * cF).sum()), x0, method='Nelder-Mead')
        if best is None or o.fun < best.fun: best = o
    gq, bq = best.x; PF = np.abs(qaoa_state(gq, bq)) ** 2; Pq = np.zeros(2 ** f); Pq[codeF] = PF
    out['ours_qaoa'] = dict(ideal=metrics(Pq), angles=[float(gq), float(bq)])
    # uniform-on-F reference (what sampling the prepared state gives with no optimisation)
    Pu = np.zeros(2 ** f); Pu[codeF] = 1 / nF; out['uniform_on_F'] = metrics(Pu)

    # ---------- baselines on the full 2^f space
    w_ex = 1.0
    def lcu_state(par):
        g, bt, th = par[0], par[1], par[2:]
        psi = np.ones(2 ** f, complex) / math.sqrt(2 ** f)
        psi = psi * np.exp(-1j * g * (cn + w_ex * excl)) * np.exp(1j * (th @ G)); return rx_layer(psi, bt)
    def unb_state(par, lam):
        g, bt = par; l1, l2 = lam
        pen = np.zeros(2 ** f)
        for row in G:
            for sgn in (1, -1): gg = sgn * row; pen += -l1 * gg + l2 * gg ** 2          # count <= target and count >= target
        psi = np.ones(2 ** f, complex) / math.sqrt(2 ** f)
        psi = psi * np.exp(-1j * g * (cn + 0.1 * (pen + 5 * excl))); return rx_layer(psi, bt)
    bestl = None
    for r in range(int(os.environ.get('CF_RESTARTS', 8))):
        x0 = np.concatenate([[rng.uniform(-1, 1), rng.uniform(0, 1.5)], rng.uniform(-1, 1, len(G))])
        o = minimize(lambda x: -metrics(np.abs(lcu_state(x)) ** 2)['quality_feasible'] * metrics(np.abs(lcu_state(x)) ** 2)['feas'] , x0, method='L-BFGS-B', options=dict(maxiter=int(os.environ.get("CF_MAXIT", 80))))
        if bestl is None or o.fun < bestl.fun: bestl = o
    out['lcu'] = dict(ideal=metrics(np.abs(lcu_state(bestl.x)) ** 2))
    bestu = None
    for lam in [(0.96, 0.037), (0.5, 0.2), (2.0, 0.5)]:
        for r in range(int(os.environ.get('CF_RESTARTS', 8))):
            x0 = [rng.uniform(-1, 1), rng.uniform(0, 1.5)]
            o = minimize(lambda x: -metrics(np.abs(unb_state(x, lam)) ** 2)['quality_feasible'] * metrics(np.abs(unb_state(x, lam)) ** 2)['feas'], x0, method='Nelder-Mead', options=dict(maxiter=120))
            if bestu is None or o.fun < bestu[0].fun: bestu = (o, lam)
    out['unbalanced'] = dict(ideal=metrics(np.abs(unb_state(bestu[0].x, bestu[1])) ** 2), lam=list(bestu[1]))
    # ---------- Grover (ideal, from uniform on F, optimum marked)
    k_opt = max(1, int(round(math.pi / 4 * math.sqrt(nF)))); M = int(opt_mask.sum())
    out['ours_grover'] = dict(marked=M, iters_opt=k_opt, iters_std_over_ours_log2=(f - math.log2(nF)) / 2, p_after_opt=float(math.sin((2 * k_opt + 1) * math.asin(math.sqrt(M / nF))) ** 2))

    # ---------- compile all four to ibm_fez
    g_, b_ = gq, bq
    qc_q = P.circuit(A, T, S, cp['bl'], cp['h1'], cp['J1'], [g_], [b_], True)
    co = ising_from_qubo(cp['h1'], cp['J1'])
    def baseline_circ(extra_h=None, extra_J=None, rz_theta=None):
        h, J = cp['h1'].copy(), dict(cp['J1'])
        if extra_h is not None:
            for v, w in extra_h.items(): h[v] += w
        if extra_J is not None:
            for k, w in extra_J.items(): J[k] = J.get(k, 0) + w
        qc = QuantumCircuit(n)
        for v in free: qc.h(v)
        phase_poly(qc, ising_from_qubo(h, J), 0.3)
        if rz_theta is not None:
            for gi, (vs, tgt) in enumerate(groups):
                for v in vs: qc.rz(2 * rz_theta[gi], v)
        for v in free: qc.rx(0.8, v)
        return qc
    exJ = {}
    for b in cp['bl']:
        for j in range(b['size']):
            a, c = b['xq'][2 * j], b['xq'][2 * j + 1]
            if a in col and c in col: exJ[(a, c)] = 0.1
    # penalty couplings among the variables of each group
    penJ = dict(exJ); penh = {}
    for vs, tgt in groups:
        for i, u in enumerate(vs):
            penh[u] = penh.get(u, 0) + 0.1
            for v2 in vs[i + 1:]: penJ[(min(u, v2), max(u, v2))] = penJ.get((min(u, v2), max(u, v2)), 0) + 0.1
    circuits = {'ours_qaoa': qc_q, 'lcu': baseline_circ(extra_J=exJ, rz_theta=bestl.x[2:]), 'unbalanced': baseline_circ(extra_h=penh, extra_J=penJ)}
    # one Grover iteration (ancilla-free reflection) and the preparation alone
    prep = QuantumCircuit(n)
    for b in cp['bl']:
        from negsearch.symlower import symmetric_block_prep
        prep.compose(symmetric_block_prep(b['size'], b['l'], b['s']), qubits=b['xq'], inplace=True)
    gc, used = compact(prep); fq = gc.num_qubits
    it = QuantumCircuit(fq); it.compose(gc.inverse(), inplace=True); it.x(range(fq)); it.h(fq - 1); it.mcx(list(range(fq - 1)), fq - 1); it.h(fq - 1); it.x(range(fq)); it.compose(gc, inplace=True)
    for k, qc in list(circuits.items()) + [('ours_grover_iter', it)]:
        r = compile_best(qc, seeds=4); out.setdefault(k, {})['compiled'] = r
        print(f"{k:18s} CZ {r['cz']:5d} depth {r['depth']:5d} qubits {r['qubits']:3d} ESP {r['esp']:.3g}", flush=True)
    # noisy (global depolarising) model on the feasible-fraction / quality
    for k, Pm in (('ours_qaoa', Pq), ('lcu', np.abs(lcu_state(bestl.x)) ** 2), ('unbalanced', np.abs(unb_state(bestu[0].x, bestu[1])) ** 2)):
        e = out[k]['compiled']['esp']; Pn = e * Pm + (1 - e) / 2 ** f; out[k]['noisy_model'] = metrics(Pn)
    # verify that the compiled circuit of ours equals the ideal F-basis simulation
    gc2, used2 = compact(qc_q)
    if gc2.num_qubits <= 22:
        sv = Statevector(gc2).probabilities(); Pcirc = np.zeros(2 ** f)
        # map compact qubit order to free-variable order
        loc = {v: k for k, v in enumerate(used2)}; probe = 0.0
        pr = np.zeros(2 ** f)
        for s_i, pv in enumerate(sv):
            if pv < 1e-15: continue
            code = 0
            for v in free:
                if (s_i >> loc[v]) & 1: code |= 1 << col[v]
            pr[code] += pv
        out['ours_qaoa']['circuit_vs_ideal_tv'] = float(0.5 * np.abs(pr - Pq).sum())
    print(json.dumps({k: v for k, v in out.items() if k in ('uniform_on_F', 'ours_qaoa', 'lcu', 'unbalanced', 'ours_grover')}, indent=1, default=float))
    json.dump(out, open(f'results/{TAG}.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()

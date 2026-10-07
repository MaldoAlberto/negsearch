"""LaTeX tables for the portfolio section (all numbers read from results/*.json)."""
import os as _os, sys as _sys
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT); _sys.path.insert(0, ROOT)
import json
from experiments.portfolio_analysis import merged
R = merged()
NM = {'a003_t02': r'\texttt{a003\_t02}', 'a004_t02': r'\texttt{a004\_t04} (2 per.)', 'a005_t02': r'\texttt{a005\_t04} (2 per.)'}
M = [('penalty_slack', 'Penalty, slack QUBO'), ('penalty_unbalanced', 'Penalty, unbalanced'),
     ('A_init_Xmixer', r'$A_q$ + X mixer'), ('A_init_XYmixer', r'$A_q$ + XY (hybrid)'), ('A_init_Grover_mixer', r'$A_q$ + Grover (CG-QAOA)')]
L = [r'\begin{table*}[t]\centering\small', r'\caption{Exact simulation on QOBLIB portfolio instances: P(feasible) / normalised quality (infeasible samples count 0) / P(optimum) for $p=1,2,3$. Penalty weights tuned over $\alpha\in[1/64,8]$; best value kept. $p=0$: uniform sampling (penalty rows) or $A_q$ alone (classically samplable).}\label{tab:pf_exact}',
     r'\resizebox{\linewidth}{!}{\begin{tabular}{llcccc}\toprule instance (qubits ours/slack) & method & $p=0$ & $p=1$ & $p=2$ & $p=3$\\\midrule']
f = lambda v: f"{v['p_feas']:.2f} / {v['quality']:.2f} / {v['p_opt']:.1e}".replace('e-0', 'e-')
for k, nm in NM.items():
    r = R[k]; first = True
    for m, lab in M:
        if m not in r['p1']: continue
        p0 = r['uniform'] if m.startswith('penalty') else r['A_alone']
        L.append((f"{nm} ({r['qubits_x']}/{r['qubits_slack_qubo']})" if first else '') + f" & {lab} & {f(p0)} & " + ' & '.join(f(r[f'p{p}'][m]) for p in (1, 2, 3)) + r'\\')
        first = False
    L.append(r'\midrule')
L[-1] = r'\bottomrule\end{tabular}}\end{table*}'
open('tables/pf_exact.tex', 'w').write('\n'.join(L) + '\n')

# depth progression
res = json.load(open('results/portfolio_resources.json')); H = json.load(open('results/portfolio_hardware_opt.json'))
P = json.load(open('results/portfolio_prune.json')); MM = json.load(open('results/portfolio_midmeasure.json'))
L = [r'\begin{table}[t]\centering\small', r'\caption{CZ gates / depth on \texttt{ibm\_fez} (FakeFez, level 3) for one and two layers. v1: counters uncomputed; v2: shorts-first order, lean counters; v3: v2 + path XY + linear phase; counts+Dicke: counts drawn classically, Dicke preparation (most expensive count sector).}\label{tab:pf_depth}',
     r'\resizebox{\linewidth}{!}{\begin{tabular}{lccc}\toprule circuit & a003 (12) & a004 (16) & a005 (20)\\\midrule']
row = lambda lab, vals: L.append(lab + ' & ' + ' & '.join(vals) + r'\\')
ks = ('a003_t02', 'a004_t02', 'a005_t02')
for p in (1, 2):
    s = '' if p == 1 else '_p2'
    row(f'penalty, slack QUBO, $p={p}$', [f"{res[k]['penalty_slack']['twoq' if p==1 else 'p2_twoq']}/{res[k]['penalty_slack']['depth' if p==1 else 'p2_depth']}" for k in ks])
    row(f'penalty, unbalanced, $p={p}$', [f"{H[k]['hardware'][f'unbalanced p={p}']['cz']}/{H[k]['hardware'][f'unbalanced p={p}']['depth']}" for k in ks])
    row(f'hybrid v1, $p={p}$', [f"{res[k]['A_init_XYmixer']['twoq' if p==1 else 'p2_twoq']}/{res[k]['A_init_XYmixer']['depth' if p==1 else 'p2_depth']}" for k in ks])
    row(f'hybrid v2, $p={p}$', [f"{H[k]['hardware'][f'hybrid p={p} (lean)']['cz']}/{H[k]['hardware'][f'hybrid p={p} (lean)']['depth']}" for k in ks])
    row(f'hybrid v3, $p={p}$', [(f"{P[k]['path_J1.01'][f'p{p}']['cz']}/{P[k]['path_J1.01'][f'p{p}']['depth']}" if f'p{p}' in P[k]['path_J1.01'] else ('601/690' if k == 'a005_t02' else '--')) for k in ks])
    row(f'counts+Dicke, $p={p}$', [(f"{MM[k][f'p{p} | sector + Dicke']['cz']}/{MM[k][f'p{p} | sector + Dicke']['depth']}" if k in MM else '--') for k in ks])
    L.append(r'\midrule')
row(r'CG-QAOA (Grover mixer), $p=1$', [f"{res[k]['A_init_Grover_mixer']['twoq']}/{res[k]['A_init_Grover_mixer']['depth']}" for k in ks])
L.append(r'\bottomrule\end{tabular}}\end{table}')
open('tables/pf_depth.tex', 'w').write('\n'.join(L) + '\n')

# scaling
S = json.load(open('results/portfolio_scaling_mps.json'))
L = [r'\begin{table*}[t]\centering\small', r'\caption{Aer MPS simulation (bond dimension 64), $p=1$, 2000 shots. Quality: normalised, infeasible = 0; gap: best of 2000 shots, $(f_{best}-f_{opt})/(f_{worst}-f_{opt})$, exact $f_{opt}$ by dynamic programming. CZ on \texttt{ibm\_fez} only where the circuit fits (156 qubits).}\label{tab:pf_scaling}',
     r'\resizebox{\linewidth}{!}{\begin{tabular}{lrrcccccccc}\toprule & & & \multicolumn{2}{c}{P(feasible)} & \multicolumn{3}{c}{quality} & \multicolumn{2}{c}{best-shot gap} & CZ (pen./hyb./per shot)\\',
     r'instance & qubits & slack QUBO & pen. & hyb. & pen. & classical & hyb. & classical & hyb. & \\\midrule']
for k, r in sorted(S.items(), key=lambda z: z[1]['qubits']):
    u, h, a = r['unbalanced p=1'], r['hybrid p=1'], r['A_q classical']
    cz = f"{u['hardware']['cz']}/{h['hardware']['cz']}/{h['hardware_per_shot_version']['cz']:.0f}" if u['hardware'] else '--'
    L.append(rf"\texttt{{{k.replace('_', chr(92)+'_')}}} & {r['qubits']} & {r['qubits_slack_qubo']} & {u['p_feas']:.4f} & {h['p_feas']:.0f} & {u['quality']:.3f} & {a['quality']:.3f} & {h['quality']:.3f} & {a['best_gap']:.3f} & {h['best_gap']:.3f} & {cz}\\")
L.append(r'\bottomrule\end{tabular}}\end{table*}')
open('tables/pf_scaling.tex', 'w').write('\n'.join(L) + '\n')

# noise
N = []
for k, v in json.load(open('results/portfolio_noisy.json')).items():
    if k.startswith('penalty'): N.append((k.replace('penalty_', 'penalty, ').replace(' p=', ', $p=') + '$', v['cz'], v['noisy']))
for k, v in json.load(open('results/portfolio_hardware_opt.json'))['a003_t02']['hardware'].items():
    if k == 'A_q alone (lean)': N.append(('$A_q$ alone (lean, classically samplable)', v['cz'], v['noisy']))
N.append(('hybrid v1, $p=1$', json.load(open('results/portfolio_noisy.json'))['A_init_XYmixer p=1']['cz'], json.load(open('results/portfolio_noisy.json'))['A_init_XYmixer p=1']['noisy']))
for k, v in json.load(open('results/portfolio_noisy_lean.json')).items():
    N.append((f"hybrid v3, $p={k[9]}$", v['cz'], v['noisy']))
for k, v in json.load(open('results/portfolio_midmeasure_noisy.json')).items():
    N.append((f"counts+Dicke, $p={k[-1]}$", round(v['mean_cz_per_shot']), v['noisy']))
L = [r'\begin{table}[t]\centering\small', r'\caption{\texttt{po\_a003\_t02\_orig} under the \texttt{ibm\_fez} noise model (FakeFez, 2000--4000 shots).}\label{tab:pf_noise}',
     r'\resizebox{\linewidth}{!}{\begin{tabular}{lcccc}\toprule circuit & CZ & P(feas.) & quality & P(opt)\\\midrule']
for lab, cz, n in N:
    L.append(f"{lab} & {cz} & {n['p_feas']:.2f} & {n['quality']:.2f} & {n['p_opt']*100:.1f}\\%\\\\")
L.append(r'\bottomrule\end{tabular}}\end{table}')
open('tables/pf_noise.tex', 'w').write('\n'.join(L) + '\n')
print('ok')

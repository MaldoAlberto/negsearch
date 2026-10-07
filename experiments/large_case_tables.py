"""LaTeX tables for the large-case section: tables/large_scaling.tex, tables/large_hw.tex, tables/large_quality.tex (if results exist)."""
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT)
import json, numpy as np

d = json.load(open('results/large_case.json'))
rows = []
for c in d['cases']:
    rows.append(f"{c['A']},{c['T']} & {c['n']} & {c['constraints']} & {c['width']} & {c['qubits']} & {c['cz']/1e3:,.1f} & $2^{{{c['log2_uniform_feas']:.0f}}}$ & {c['qubo_qubits']} & {c['cz_conditioned_sum_mean']/1e3:,.1f} & {c['sector_cz_total_typ_oh']/1e3:,.1f} & {c['sector_cz_max_oh']}\\\\")
tab = r"""\begin{table*}[t]
\centering\footnotesize\setlength{\tabcolsep}{2.6pt}
\caption{Multi-period portfolio with sector caps, long/short exclusivity and a global trading budget ($A$ assets, $T$ periods, sectors of $3$--$5$ assets). Left: the coherent state $A_q$ over all constraints (binary register, period-major order). Right: classical preprocessing that conditions on the interface. CZ counts (thousands) use Qiskit's default ancilla-free multi-controlled synthesis, before routing (routing roughly doubles them). ``Per-period'' conditions on the number of positions of each period (binary registers, sum over periods); ``per-sector'' also on the counts of each sector (one-hot registers; typical total over the blocks, and the largest single block in CZ). The uniform column is the share of feasible strings in the uniform state; the slack column is the number of qubits of a slack-variable QUBO.}\label{tab:large_scaling}
\begin{tabular}{lrrrrrrrrrr}
\toprule
 & & & \multicolumn{3}{c}{coherent $A_q$} & & & \multicolumn{3}{c}{conditioned (k CZ)}\\
\cmidrule(lr){4-6}\cmidrule(lr){9-11}
$A,T$ & $n$ & constr. & width & qubits & kCZ & unif. & slack q. & per-period & per-sector & max block\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
open('tables/large_scaling.tex', 'w').write(tab)

hw = []
for tag, label in (('_n24', '$n=24$'), ('_n8', '$n=8$')):
    m = json.load(open(f'hardware_circuits/large_case{tag}_meta.json'))
    for c in m['circuits']:
        hw.append((label, c['name'], c['qubits'], c['cz'], c['depth'], c['duration_us'], c['esp'], c['weight']))
lines = []
for tag, label in (('_n24', '$n=24$'), ('_n8', '$n=8$')):
    for nm in ('prep', 'qaoa1'):
        R = [r for r in hw if r[0] == label and r[1] == nm]
        if not R:
            continue
        f = lambda i: (min(r[i] for r in R), max(r[i] for r in R))
        rng = lambda i, fmt: (fmt % f(i)[0]) if f(i)[0] == f(i)[1] else (fmt % f(i)[0]) + '--' + (fmt % f(i)[1])
        lines.append(f"{label} & {'$A_q$ only' if nm=='prep' else '$A_q$ + QAOA $p{=}1$'} & {rng(2,'%d')} & {rng(3,'%d')} & {rng(4,'%d')} & {rng(5,'%.1f')} & {rng(6,'%.2f')}\\\\")
tab = r"""\begin{table*}[t]
\centering\small\setlength{\tabcolsep}{4pt}
\caption{Hardware circuits generated for the conditioned multi-constraint case, compiled on the \texttt{ibm\_fez} topology (FakeFez snapshot; best of 10 transpiler seeds by estimated success probability; ranges over the interface tuples compiled). Nothing was executed on a device. The QAOA layer is not executable at $n=24$ (about $3.8$--$4.4$ thousand CZ, estimated success probability $\approx0$), so only the preparation was generated there.}\label{tab:large_hw}
\begin{tabular}{llrrrrr}
\toprule
instance & circuit & qubits & CZ & depth & $\mu$s & ESP\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
open('tables/large_hw.tex', 'w').write(tab)

if _os.path.exists('results/large_case_quality.json'):
    q = json.load(open('results/large_case_quality.json'))
    meths = ['A_q + Grover mixer', 'conditioned + GM', 'penalty', 'unbalanced']
    lab = {'A_q + Grover mixer': r'$A_q$ + Grover mixer', 'conditioned + GM': 'conditioned + Grover mixer', 'penalty': 'penalty QAOA', 'unbalanced': 'unbalanced QAOA'}
    ln = []
    for p in ('1', '2', '3'):
        for m in meths:
            vs = [q[s][p][m] for s in q if p in q[s]]
            if not vs:
                continue
            ln.append(f"{p} & {lab[m]} & {np.mean([v['feas'] for v in vs]):.2f} & {np.mean([v['quality'] for v in vs]):.2f} & {np.mean([v['popt'] for v in vs]):.3f}\\\\")
        if p != '3':
            ln.append(r"\midrule")
    base = [q[s]['uniform'] for s in q]; aq = [q[s]['A_q alone'] for s in q]
    tab = r"""\begin{table}[t]
\centering\small\setlength{\tabcolsep}{4pt}
\caption{Quality on the multi-constraint case at $n=16$ (4 assets, 2 periods, 2 sectors; $521$ feasible strings; uniform feasible share $0.8$\,\%; mean over """ + str(len(q)) + r""" random objectives). QAOA depth $p$, exact numerics with locally optimised angles. Quality counts infeasible samples as $0$; $P(\mathrm{opt})$ is the probability of an optimal string (uniform feasible sampling: """ + f"{np.mean([b['popt'] for b in aq]):.4f}" + r"""). Preparation alone ($p=0$): feasible $1.00$, quality """ + f"{np.mean([b['quality'] for b in aq]):.2f}" + r""".}\label{tab:large_quality}
\begin{tabular}{llrrr}
\toprule
$p$ & method & feasible & quality & $P(\mathrm{opt})$\\
\midrule
""" + "\n".join(ln) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    open('tables/large_quality.tex', 'w').write(tab)
print('ok')

"""Tables (LaTeX + markdown) and figure for the generality study.  Reads results/general_*.json."""
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT)
import json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

G = json.load(open('results/general_grover.json'))
Q = json.load(open('results/general_qaoa.json'))
C = json.load(open('results/general_circuits.json'))
R = json.load(open('results/general_relaxation.json'))
ORDER = ['cardinality', 'knapsack', 'weighted MIS', 'set packing', 'exact cover', '3-colouring', 'quadratic assignment']
LABEL = {'cardinality': 'cardinality ($\\sum x=4$)', 'knapsack': 'knapsack (weights $w_i$)', 'weighted MIS': 'independent set',
         'set packing': 'set packing', 'exact cover': 'exact cover', '3-colouring': '3-colouring',
         'quadratic assignment': 'assignment (permutations)'}
KIND = {'cardinality': 'count', 'knapsack': 'linear ineq.', 'weighted MIS': 'pairwise', 'set packing': 'overlap $\\le1$',
        'exact cover': 'overlap $=1$', '3-colouring': 'one-hot + edges', 'quadratic assignment': 'row/col sums $=1$'}
med = lambda rows, k: float(np.median([r[k] for r in rows]))
fam = lambda r: r['name'].split(' (')[0]
f3 = lambda x: (f'{x:.1e}'.replace('e-0', 'e-').replace('e+0', 'e')) if x < 0.01 else f'{x:.3f}'


def sci(x):
    if x == 0:
        return '0'
    e = int(np.floor(np.log10(x)))
    if -2 <= e <= 0:
        return f'{x:.3f}'
    m = x / 10 ** e
    return f'${m:.1f}\\!\\cdot\\!10^{{{e}}}$'


# ---------------- Table 1: Grover -----------------
rows_tex, rows_md = [], []
for f in ORDER:
    R0 = [r for r in G if fam(r) == f and r['rank'] == 0.0 and r['n'] <= 10]
    if not R0:
        continue
    n = R0[0]['n']
    row = (LABEL[f], KIND[f], len(R0), int(med(R0, 'width')), med(R0, 'frac_feasible'), med(R0, 'a_unif'), med(R0, 'a_half'),
           med(R0, 'a_best'), int(med(R0, 'k_std')), int(med(R0, 'k_half')), int(med(R0, 'k_best')))
    rows_tex.append(row)
# 4x4 permutation instance, n=16
r16 = [r for r in G if r['n'] == 16 and r['rank'] == 0.0]
for r in r16:
    rows_tex.append(('assignment $4\\times4$ ($n{=}16$)', 'row/col sums $=1$', 1, r['width'], r['frac_feasible'], r['a_unif'],
                     r['a_half'], r['a_best'], r['k_std'], r['k_half'], r['k_best']))
with open('tables/general_grover.tex', 'w') as fh:
    fh.write('\\begin{table*}[t]\\centering\\small\\setlength{\\tabcolsep}{3.5pt}\n\\caption{Amplitude amplification on seven constraint types with the same pipeline (exact numerics, $n=10$ '
             'decision variables unless noted; median over the instances, optimal solutions marked). $w$: automaton width; '
             '$a_{\\rm unif}$, $a_{1/2}$, $a_{q^*}$: probability of the marked set in the uniform state, in $A_{1/2}|0\\rangle$ and in the best-biased $A_q|0\\rangle$; '
             '$k$: optimal number of amplification iterations. Standard Grover must also verify feasibility in its oracle; $A_q$ does not.}\\label{tab:gen_grover}\n'
             '\\begin{tabular}{llrrrrrrrrr}\\toprule\nfamily & constraint & inst. & $w$ & $|F|/2^n$ & $a_{\\rm unif}$ & $a_{1/2}$ & $a_{q^*}$ & $k_{\\rm std}$ & $k_{1/2}$ & $k_{q^*}$\\\\\\midrule\n')
    for (lab, kind, ni, w, ff, a0, a5, aq, ks, k5, kq) in rows_tex:
        fh.write(f'{lab} & {kind} & {ni} & {w} & {sci(ff)} & {sci(a0)} & {sci(a5)} & {sci(aq)} & {ks} & {k5} & {kq}\\\\\n')
    fh.write('\\bottomrule\\end{tabular}\\end{table*}\n')

# ---------------- Table 2: QAOA -----------------
def agg(f, meth, p, key, n_max=12):
    rows = [r for r in Q if fam(r) == f and r['n'] <= n_max and str(p) in r[meth]]
    return float(np.median([r[meth][str(p)][key] for r in rows])), len(rows)


with open('tables/general_qaoa.tex', 'w') as fh:
    fh.write('\\begin{table*}[t]\\centering\\small\\setlength{\\tabcolsep}{3.5pt}\n\\caption{QAOA at depth $p=3$ on the same families (exact numerics, median over instances). '
             'Penalty: $|+\\rangle^{\\otimes n}$, X mixer, best of four penalty weights. $A_q$+GM: negation-forced initial state, objective-only phase, Grover mixer '
             '(valid for any constraint). $A_q$+NB: same state, mixer over single flips and swaps inside $F$ (numerics only). '
             'Quality: normalised objective with infeasible samples counted as 0; $p{=}0$ is $A_q$ alone.}\\label{tab:gen_qaoa}\n'
             '\\begin{tabular}{lrrrrrrrrrr}\\toprule\n& & \\multicolumn{3}{c}{P(feasible)} & \\multicolumn{4}{c}{quality} & \\multicolumn{2}{c}{P(optimum)}\\\\'
             '\\cmidrule(lr){3-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-11}\n'
             'family & inst. & penalty & $A_q$+GM & $A_q$+NB & $A_q$ alone & penalty & $A_q$+GM & $A_q$+NB & penalty & $A_q$+GM\\\\\\midrule\n')
    for f in ORDER:
        if f == 'quadratic assignment':
            pass
        pf, ni = agg(f, 'penalty', 3, 'p_feas')
        if ni == 0:
            continue
        gf, _ = agg(f, 'gm', 3, 'p_feas'); nf, _ = agg(f, 'nb', 3, 'p_feas')
        qp, _ = agg(f, 'penalty', 3, 'quality'); qg, _ = agg(f, 'gm', 3, 'quality'); qn, _ = agg(f, 'nb', 3, 'quality')
        qa = float(np.median([r['base']['Aq']['quality'] for r in Q if fam(r) == f and r['n'] <= 12]))
        op, _ = agg(f, 'penalty', 3, 'p_opt'); og, _ = agg(f, 'gm', 3, 'p_opt')
        fh.write(f'{LABEL[f]} & {ni} & {pf:.2f} & {gf:.2f} & {nf:.2f} & {qa:.2f} & {qp:.2f} & {qg:.2f} & {qn:.2f} & {op:.3f} & {og:.3f}\\\\\n')
    fh.write('\\bottomrule\\end{tabular}\\end{table*}\n')

# ---------------- Table 3: circuits -----------------
with open('tables/general_circuits.tex', 'w') as fh:
    fh.write('\\begin{table*}[t]\\centering\\small\n\\caption{Generic compiler (automaton $\\to$ circuit), $n=6$ decision variables, Qiskit Aer. '
             '$q$: total qubits (decisions + automaton registers); CZ and depth of $A_q$ after transpilation to $\\{$CZ, $R_z$, $\\sqrt X$, $X$, $R_y\\}$ without any hand optimisation; '
             'errors are the largest deviation of the simulated circuit probabilities from the exact formula over all Grover iterations / the $p{=}2$ Grover-mixer QAOA circuit.}\\label{tab:gen_circ}\n'
             '\\begin{tabular}{lrrrrrr}\\toprule\nfamily & $q$ & $w$ & CZ($A_q$) & $k_{\\rm std}\\to k_{A_q}$ & err.\\ Grover & err.\\ QAOA\\\\\\midrule\n')
    SHORT = {'cardinality': 'cardinality', 'knapsack': 'knapsack', 'weighted MIS': 'independent set', 'set packing': 'set packing',
             'exact cover': 'exact cover', '3-colouring': '3-colouring', 'quadratic assignment': 'assignment'}
    for r in C:
        f = fam(r)
        fh.write(f"{SHORT.get(f, f)} & {r['nq']} & {r['width']} & {r['cz_Aq']} & ${r['k_std']}\\to{r['k_aq']}$ & {r['max_err_grover']:.0e} & {r['max_err_qaoa']:.0e}\\\\\n")
    fh.write('\\bottomrule\\end{tabular}\\end{table*}\n')

# ---------------- Table 4: relaxation -----------------
with open('tables/general_relax.tex', 'w') as fh:
    fh.write('\\begin{table}[t]\\centering\\small\\setlength{\\tabcolsep}{4pt}\n\\caption{Width\\,--\\,completeness dial on knapsack ($n=12$, six instances, median). The automaton tracks the weight in units of $r$ '
             '(a lower bound), so forcing is sound but incomplete: $r=1$ is exact, $r$ above the largest weight keeps no information.}\\label{tab:gen_relax}\n'
             '\\begin{tabular}{rrrrr}\\toprule\n$r$ & width & P(feas.) & $a_{\\rm opt}$ & $k_{A_q}$ (std.\\ 50)\\\\\\midrule\n')
    for r in (1, 2, 3, 4, 6, 12):
        RR = [o for o in R if o['r'] == r]
        fh.write(f"{r} & {int(med(RR,'width'))} & {med(RR,'p_feas'):.2f} & {sci(med(RR,'a_opt'))} & {int(med(RR,'k_amp'))}\\\\\n")
    fh.write('\\bottomrule\\end{tabular}\\end{table}\n')

# ---------------- Table 5: cost per iteration -----------------
CO = json.load(open('results/general_cost.json'))
SHORT2 = {'cardinality': 'cardinality', 'knapsack': 'knapsack', 'weighted MIS': 'independent set', 'set packing': 'set packing',
          'exact cover': 'exact cover', '3-colouring': '3-colouring', 'quadratic assignment': 'assignment'}
def cz(x):
    return f'{x:,}'.replace(',', '\\,')
with open('tables/general_cost.tex', 'w') as fh:
    fh.write('\\begin{table*}[t]\\centering\\small\\setlength{\\tabcolsep}{3.5pt}\n\\caption{Gate cost of amplitude amplification, standard Grover against $A_q$ (CZ after transpilation to $\\{$CZ, $R_z$, $\\sqrt X$, $X$, $R_y\\}$, no hand optimisation; the objective-threshold oracle is common to both and not counted). '
             'Standard iteration: feasibility oracle (automaton compute, phase, uncompute) + diffusion. $A_q$ iteration: $A_q^\\dagger$, $A_q$, and the phase flip on all qubits. '
             '``default'': Qiskit\'s ancilla-free multi-controlled gates; ``V-chain'': the two all-qubit phase flips use Toffoli V-chains with clean ancillas (extra $N-3$ qubits for $A_q$). Totals = iterations $\\times$ cost per iteration (+ one $A_q$).}\\label{tab:gen_cost}\n'
             '\\begin{tabular}{lrrrrrrrrrr}\\toprule\n& & & & & & \\multicolumn{2}{c}{total CZ, default} & \\multicolumn{2}{c}{total CZ, V-chain}\\\\\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}\n'
             'family & $n$ & $w$ & CZ($A_q$) & CZ(oracle) & $k_{\\rm std}\\!\\to\\!k_{A_q}$ & standard & $A_q$ & standard & $A_q$\\\\\\midrule\n')
    for r in CO:
        if r['n'] < 9:
            continue
        f = fam(r)
        lab = SHORT2.get(f, f) + (' $4\\times4$' if r['n'] == 16 else (' $3\\times3$' if f.startswith('quadratic') else ''))
        fh.write(f"{lab} & {r['n']} & {r['width']} & {cz(r['cz_Aq'])} & {cz(r['cz_oracle'])} & ${r['k_std']}\\to{r['k_aq']}$ & {cz(r['total_std'])} & {cz(r['total_aq'])} & {cz(r['total_std_v'])} & {cz(r['total_aq_v'])}\\\\\n")
    fh.write('\\bottomrule\\end{tabular}\\end{table*}\n')

# ---------------- Table 6: MPS scaling -----------------
SC = json.load(open('results/general_scaling_mps.json'))
sel = [('cardinality', 24), ('cardinality', 60), ('independent set (band d=2)', 24), ('independent set (band d=2)', 60),
       ('independent set (band d=2)', 80), ('knapsack', 18), ('knapsack', 30)]
LAB = {'cardinality': 'cardinality (count)', 'independent set (band d=2)': 'independent set (band)', 'knapsack': 'knapsack'}
with open('tables/general_scaling.tex', 'w') as fh:
    fh.write('\\begin{table*}[t]\\centering\\small\\setlength{\\tabcolsep}{3.5pt}\n\\caption{The generic compiler at scale (Aer matrix-product states, 2000 shots, automata written from the problem structure). '
             'Qubits include the automaton registers (interleaved with the variables); the bond dimension of the final state equals the automaton width. '
             'Feasible: share of samples accepted by the automaton. Uniform: share of feasible strings among all $2^n$ (exact). '
             'Mean: mean sample objective $\\pm$ standard error against the exact expectation under $A_q$ (dynamic programme over the automaton). Best / opt.: best of 2000 shots / exact optimum.}\\label{tab:gen_scale}\n'
             '\\begin{tabular}{lrrrrrrrrr}\\toprule\nfamily & $n$ & qubits & $w$ & bond & feasible & uniform & mean (MPS) & mean (exact) & best / opt.\\\\\\midrule\n')
    for k, n_ in sel:
        r = [x for x in SC if x['kind'] == k and x['n'] == n_][0]
        fh.write(f"{LAB[k]} & {r['n']} & {r['qubits']} & {r['width']} & {r['bond']} & {r['p_feas']:.3f} & {sci(r['p_uniform_feas'])} & "
                 f"${r['mean_sample']:.1f}\\pm{r['se']:.1f}$ & {r['mean_exact']:.1f} & {r['best']:.0f} / {r['opt']:.0f}\\\\\n")
    fh.write('\\bottomrule\\end{tabular}\\end{table*}\n')

# ---------------- Table 7: limits (market split) -----------------
LM = json.load(open('results/general_limits.json'))
with open('tables/general_limits.tex', 'w') as fh:
    fh.write('\\begin{table}[t]\\centering\\small\\setlength{\\tabcolsep}{4pt}\n\\caption{Limit: market split ($m=3$ equality constraints, random integers up to 99, one planted solution). '
             'The automaton is the sound relaxation that forces only when a bound makes a value impossible (state = vector of partial sums). Width: distinct states; '
             'P(feas.): probability that $A_{1/2}|0\\rangle$ ends on a solution (dead ends lose mass); medians over three instances.}\\label{tab:gen_limits}\n'
             '\\begin{tabular}{rrrrr}\\toprule\n$n$ & width & P(feas.) & uniform & $k_{\\rm std}\\to k_{A_q}$\\\\\\midrule\n')
    for r in [x for x in LM if x['m'] == 3]:
        fh.write(f"{r['n']} & {int(r['width']):,} & {sci(r['p_feas'])} & {sci(r['p_uniform'])} & ${r['k_std']}\\to{r['k_aq']}$\\\\\n".replace(',', '\\,'))
    fh.write('\\bottomrule\\end{tabular}\\end{table}\n')

# ---------------- Figure -----------------
INK = '#52514e'
fams = [f for f in ORDER]
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
ax = axs[0]
xs = np.arange(len(rows_tex))
w = 0.26
ax.bar(xs - w, [r[8] for r in rows_tex], w, color='#9a9893', label='standard Grover on $|+\\rangle^{\\otimes n}$')
ax.bar(xs, [r[9] for r in rows_tex], w, color='#1baf7a', label='amplification on $A_{1/2}|0\\rangle$')
ax.bar(xs + w, [r[10] for r in rows_tex], w, color='#4a3aa7', label='amplification on $A_{q^*}|0\\rangle$')
ax.set_yscale('symlog', linthresh=1); ax.set_xticks(xs)
short = ['count', 'knapsack', 'indep.\nset', 'set\npacking', 'exact\ncover', '3-col.', 'assign.\n3x3', 'assign.\n4x4 (16q)']
ax.set_xticklabels(short[:len(xs)], fontsize=7.5); ax.set_ylabel('amplification iterations $k$', fontsize=9)
ax.set_title('(a) Grover: iterations to find an optimum', fontsize=9, loc='left'); ax.legend(fontsize=7.5, frameon=False)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
ax.grid(axis='y', color='#e4e3df', lw=0.8); ax.set_axisbelow(True)
ax = axs[1]
fl = [f for f in ORDER if agg(f, 'penalty', 3, 'quality')[1] > 0]
xs = np.arange(len(fl))
vals = {m: [agg(f, m, 3, 'quality')[0] for f in fl] for m in ('penalty', 'gm', 'nb')}
base = [float(np.median([r['base']['Aq']['quality'] for r in Q if fam(r) == f and r['n'] <= 12])) for f in fl]
ax.bar(xs - 1.5 * w, base, w, color='#d7d6d2', label='$A_q$ alone ($p=0$)')
ax.bar(xs - 0.5 * w, vals['penalty'], w, color='#eb6834', label='penalty QAOA')
ax.bar(xs + 0.5 * w, vals['gm'], w, color='#4a3aa7', label='$A_q$ + Grover mixer')
ax.bar(xs + 1.5 * w, vals['nb'], w, color='#1baf7a', label='$A_q$ + neighbour mixer')
ax.set_xticks(xs); ax.set_xticklabels([short[ORDER.index(f)] for f in fl], fontsize=7.5)
ax.set_ylabel('normalised quality (infeasible = 0)', fontsize=9); ax.set_ylim(0, 1.18)
ax.set_title('(b) QAOA, $p=3$', fontsize=9, loc='left'); ax.legend(fontsize=7.2, frameon=False, ncol=2, loc='upper left')
for s in ('top', 'right'): ax.spines[s].set_visible(False)
ax.grid(axis='y', color='#e4e3df', lw=0.8); ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig('figures/fig_general.pdf'); fig.savefig('figures/fig_general.png', dpi=200)
print('ok', len(Q), 'qaoa rows')

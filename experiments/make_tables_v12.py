"""Tables of the reduction / residual-core / four-method / hardware-estimate section (sec_core.tex) from results/*.json."""
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT)
import json, math
import numpy as np
J = lambda f: json.load(open(f'results/{f}.json'))


def sci(x, d=1):
    if x == 0: return '0'
    e = int(math.floor(math.log10(abs(x)))); m = x / 10 ** e
    return f'{m:.{d}f}\\cdot10^{{{e}}}' if e < -2 or e > 4 else f'{x:.{max(0, 3 - len(str(int(abs(x)))) if abs(x) >= 1 else 3)}f}'


def tex_e(x):
    m, e = f'{x:.1e}'.split('e'); return f'${m}\\cdot10^{{{int(e)}}}$'


def w(name, cap, label, cols, head, rows, star=False):
    env = 'table*' if star else 'table'
    L = [f'\\begin{{{env}}}[t]', '\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}', f'\\caption{{{cap}}}\\label{{{label}}}', f'\\begin{{tabular}}{{{cols}}}', '\\toprule', head + '\\\\', '\\midrule']
    for r in rows: L.append(r if r == '\\midrule' else r + '\\\\')
    L += ['\\bottomrule', '\\end{tabular}', f'\\end{{{env}}}']
    open(f'tables/{name}.tex', 'w').write('\n'.join(L) + '\n')


# ---- 1. exact pruning stages
pt = J('prune_circuit_tuples'); fv = {r['n']: r for r in J('prune_free_vars')}
rows = []
for r in pt:
    n = r['n']; rows.append(f"{n} & {fv[n]['free_mean']:.1f} & {r['S0']['cz_mean']:.0f} & {r['S2']['cz_mean']:.0f} & {r['S3']['cz_mean']:.0f} & {r['S0']['esp_median']:.2f} & {r['S2']['esp_median']:.2f} & {r['S3']['esp_median']:.2f}")
w('core_prune', 'Exact pruning of one QAOA layer ($p=1$) for the multi-constraint case, averaged over interface tuples sampled with their weight (8--12 per size), compiled to the \\texttt{ibm\\_fez} topology (best of three seeds; nothing executed). S0: symmetric preparation, pair-exchange mixers, full objective. S2: variables fixed by the interface eliminated from the objective, blocks with a single feasible string get no mixer, two-qubit swap rotations when a block has no long or no short position. S3: S2 plus a constraint gauge that rewrites couplings with the constant group counts. All three give the same state on the feasible set (defect $\\le10^{-14}$ against a dense simulation for $n\\le16$ and $1.3\\cdot10^{-14}$ by two statevector simulations at $n=24$). The reduction is large because the interface fixes most variables: the last column of the left block is the mean number of qubits that remain free.', 'tab:core_prune', 'rrrrrrrr',
  '$n$ & free & \\multicolumn{3}{c}{CZ (mean)} & \\multicolumn{3}{c}{ESP (median)}\\\\ & qubits & S0 & S2 & S3 & S0 & S2 & S3', rows)

# ---- 2. residual core map
cm = J('core_map'); sel = []
for fam in ('tight', 'medium', 'loose'):
    for n in (96, 192, 384, 768):
        for r in cm:
            if r['family'] == fam and r['n'] == n: sel.append(r)
rows = [f"{r['family']} & {r['n']} & {r['free_mean']:.0f} & {r['log2F_mean']:.0f} & {r['zz_free']:.0f}" for r in sel]
w('core_map', 'Residual quantum core after the exact classical reduction (interface drawn classically), counted for 200 interface tuples per row sampled with their weight under the global budget. Families differ in how loose the local constraints are: tight (at most one long position per sector, at most $3$ positions per period), medium ($2$, $6$) and loose ($4$, $12$). ``free\'\' is an upper bound on the qubits that remain (all variables of a block that is not fully determined); $\\log_2|F|$ is the number of feasible strings of the core; ZZ is the number of couplings among free variables if the objective is dense (an upper bound). Counting only; nothing compiled.', 'tab:core_map', 'lrrrr',
  'family & $n$ & free & $\\log_2|F|$ & ZZ', rows)

# ---- 3. cases
C = [('case_four', 'ours_qaoa'), ('case_four_n100', 'ours_qaoa'), ('case_large_n192', 'ours_qaoa')]
rows = []
for tag in ('case_four', 'case_four_n100', 'case_large_n192'):
    d = J(tag); n, f, nF = d['n'], d['free'], d['nF']
    first = True
    for k, name in (('ours_qaoa', 'ours (QAOA)'), ('lcu', 'Fourier-LCU'), ('unbalanced', 'unbalanced')):
        c = d[k]['compiled']; ide = d[k].get('ideal')
        fe = f"{100 * ide['feas']:.0f}\\,\\%" if ide else '--'; q = f"{ide['quality_feasible']:.2f}" if ide else '--'; po = (f"{100 * ide['popt']:.2f}\\,\\%" if 100 * ide['popt'] >= 0.01 else tex_e(ide['popt'])) if ide else '--'
        head = f"{n} & {f} & {int(nF)}" if first else ' & & '
        rows.append(f"{head} & {name} & {c['cz']} & {c['depth']} & {c['esp']:.3f} & {fe} & {q} & {po}")
        first = False
    gtag = {'case_four': 'grover_cheaper', 'case_four_n100': 'grover_cheaper_n100', 'case_large_n192': 'grover_cheaper_n192'}[tag]; gc = J(gtag); gi = d['ours_grover']
    for mode, lab in (('noancilla', 'Grover, 1 it., no ancilla'), ('v-chain', 'Grover, 1 it., V-chain')):
        g = gc[mode]; rows.append(f" & & & ours ({lab}) & {g['cz']} & {g['depth']} & {tex_e(g['esp']) if g['esp'] < 0.01 else format(g['esp'], '.3f')} & \\multicolumn{{3}}{{l}}{{{g['qubits']} qubits; {gi.get('iters_opt', gi.get('iters'))} iterations}}")
    rows.append('\\midrule')
rows.pop()
w('core_cases', 'Three cases, four methods, one $p=1$ layer on the residual core, compiled to \\texttt{ibm\\_fez} (156 qubits; best of 3--4 seeds; nothing executed). All methods receive the same classically reduced objective; angles are trained in ideal simulation (ours: inside $F$; the baselines: dense simulation, only for $n\\le100$, with few restarts, so they may be under-trained; at $n=192$ they are compiled only). Feasible, quality (1 is the optimum, among feasible samples) and $P(\\mathrm{opt})$ are ideal-simulation values. The interface tuples are chosen so that the core fits ($16$, $20$, $30$ free qubits); a typical tuple at these sizes leaves $24$--$40$ or more free qubits. $|F|$ is the number of feasible strings of the core, which for the first two is small enough to enumerate. The Grover rows give one compiled iteration, with an ancilla-free reflection and with a Toffoli V-chain reflection (clean ancillas), and the iterations needed to amplify the single optimum.', 'tab:core_cases', 'rrrlrrrrrr',
  '$n$ & free & $|F|$ & method & CZ & depth & ESP & feas. & qual. & $P$(opt)', rows, star=True)

# ---- 4. noise validation
nv = J('noise_validation'); rows = []
for k, name in (('ours_qaoa', 'ours (QAOA layer)'), ('lcu_like', 'single-basis LCU-like layer')):
    r = nv[k]
    rows.append(f"{name} & {r['cz']} & {r['esp']:.3f} & {100 * r['ideal_feasible']:.1f} & {100 * r['model_feasible']:.1f} & {100 * r['sim_feasible']:.1f}$\\pm${100 * r['sim_feasible_se']:.1f} & {r['ideal_quality']:.3f} & {r['sim_quality_postselected']:.3f}")
w('core_noise', 'The global-depolarising model against a gate-level noisy simulation (Aer, noise model of \\texttt{FakeFez}: gate errors, thermal relaxation, readout; $600$ shots) on the $n=64$ case with $16$ free qubits. Model: feasible fraction $=\\mathrm{ESP}\\cdot P_\\mathrm{ideal}+(1-\\mathrm{ESP})|F|/2^f$. Quality after post-selecting the feasible shots is compared with its ideal value. The LCU-like row has untrained angles and is only a check of the noise model, not of that method.', 'tab:core_noise', 'lrrrrrrr',
  'circuit & CZ & ESP & ideal feas.\\,\\% & model\\,\\% & simulated\\,\\% & ideal qual. & post-sel. qual.', rows, star=True)

# ---- 5. hardware estimate
eps = []
for tag in ('case_four', 'case_four_n100', 'case_large_n192'):
    d = J(tag)
    for k in ('ours_qaoa', 'lcu', 'unbalanced'): c = d[k]['compiled']; eps.append(-math.log(c['esp']) / c['cz'])
eps_med = float(np.median(eps)); json.dump(dict(eps_med=eps_med, eps_min=min(eps), eps_max=max(eps)), open('results/hw_estimate.json', 'w'))
cases = [(16, 473), (20, 669), (30, 923)]
rows = []
for fac, lab in ((1, 'today ($\\varepsilon\\approx%.4f$)' % eps_med), (1 / 3, '$3\\times$ better'), (1 / 10, '$10\\times$ better')):
    e = eps_med * fac; cells = ' & '.join(f"{math.exp(-e * cz):.2f}" if math.exp(-e * cz) >= 0.005 else f"{math.exp(-e * cz):.0e}" for _, cz in cases)
    c1 = math.log(10) / e / 31; c3 = math.log(1 / 0.3) / e / 31
    rows.append(f"{lab} & {cells} & {c3:.0f} & {c1:.0f}")
w('core_hw', 'Estimate of what one QAOA layer needs. Effective error per CZ $\\varepsilon=-\\ln(\\mathrm{ESP})/\\#\\mathrm{CZ}$ over the nine compiled circuits of \\cref{tab:core_cases} (all gates, routing and readout included), median %.4f (range %.4f--%.4f). ESP of our layer at $473$, $669$ and $923$ CZ ($16$, $20$, $30$ free qubits) if the error per CZ were reduced by the factor shown; last two columns: free qubits whose layer keeps ESP $\\ge0.3$ and $\\ge0.1$, assuming the $\\approx31$ CZ per free qubit observed in these cores (objective coupling only within a period; a dense objective grows quadratically and would give far fewer). Scaling of the error rate is an assumption, not a prediction of hardware.' % (eps_med, min(eps), max(eps)), 'tab:core_hw', 'lrrrrr',
  'error per CZ & ESP$_{16}$ & ESP$_{20}$ & ESP$_{30}$ & free qubits, ESP$\\ge0.3$ & ESP$\\ge0.1$', rows, star=True)
print('eps', eps_med, min(eps), max(eps))

# ---- 6. generic automaton preparation against the symmetric lowering (and measurement-based uncomputation)
rows = []
for tag, lab, f in (('n64', 64, 16), ('n100', 100, 20), ('n192', 192, 30)):
    d = J(f'generic_vs_symmetric_{tag}'); m = J(f'mbu_compare_{tag}'); s0, s1, g0, g1 = d['symmetric_prep'], d['symmetric_layer'], d['generic_prep'], d['generic_layer']
    mb, ml = m['mbuU_prep'], m['mbuU_layer']
    rows.append(f"{lab} & {f} & {s0['qubits']} / {g0['qubits']} / {mb['qubits']} & {s0['cz']} / {g0['cz']} / {m['mbu_expected_prep_cz']:.0f} & {s1['cz']} / {g1['cz']} / {ml['cz']}")
w('core_generic', 'Our construction with and without the symmetric (Dicke-based) lowering, on the three cores. Entries: symmetric lowering / generic automaton preparation $A_q$ with binary registers uncomputed in reverse / generic one-hot register with measurement-based uncomputation and one pool of reused register qubits. ``qubits\'\' are those of the preparation alone; CZ are for the preparation alone and for one QAOA layer (same objective and mixers; for the measurement-based version the layer uses the worst case, all corrections applied, and the preparation column the expected value, half of the corrections). Compiled to \\texttt{ibm\\_fez} (best of three seeds). The one-hot register with reverse uncomputation needs $90$, $119$ and $179$ qubits (it does not fit at $n=192$) and $1611$, $2385$ CZ at $n=64$, $100$.', 'tab:core_generic', 'rrrrr',
  '$n$ & free & qubits (prep) & CZ (prep) & CZ (layer)', rows, star=True)

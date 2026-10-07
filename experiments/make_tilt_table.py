"""tables/tilted_prep.tex from results/{cost_informed_prep,tilted_prep,baselines_biased,retrain_relphase}.json"""
import os, json
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); os.chdir(ROOT)
C = json.load(open('results/cost_informed_prep.json'))['mean']; T = json.load(open('results/tilted_prep.json')); B = json.load(open('results/baselines_biased.json'))
Tm = T['mean']
L = ['\\begin{table*}[t]\\centering\\small',
     '\\caption{\\orig{exact simulation of the logical circuits, $12$ blocks of the $120$-qubit instance; no noise; nothing executed} Cost-informed preparation. Feasible: ideal fraction of feasible samples; quality among feasible samples ($1$ is the best of the block); $P(\\mathrm{opt})$: probability of the block optimum at $p=0$. Tilt: product tilt $\\exp(-\\tau\\sum_i g_ix_i)$ with the mean-field field $g$ of the block, $\\tau=4$ (edge of the grid), realised by changing only the angles of the Dicke blocks. Boltzmann: $\\exp(-\\tau c(x))$, an upper bound (its smallest probability on $F$ is $4\\cdot10^{-10}$, so the support is exact only in name). Baselines: warm start with $k/m$ (plain) or with the marginals of the tilted distribution (biased), $p=1$.}\\label{tab:tilted_prep}',
     '\\begin{tabular}{lrrrr}\\toprule', 'preparation / method & feasible & quality $p{=}0$ & quality $p{=}1$ & $P$(opt) $p{=}0$\\\\\\midrule',
     f"ours, uniform on $F$ & $1$ & {C['uniform']['q_p0']:.2f} & {C['uniform']['q_p1']:.2f} & {100*C['uniform']['popt_p0']:.0f}\\,\\%\\\\",
     f"ours, product tilt & $1$ & {Tm['product']['q_p0']:.2f} & {Tm['product']['q_p1']:.2f} & {100*Tm['product']['popt_p0']:.0f}\\,\\%\\\\",
     f"ours, Boltzmann (upper bound) & $1$ & {Tm['boltzmann']['q_p0']:.2f} & {Tm['boltzmann']['q_p1']:.2f} & {100*Tm['boltzmann']['popt_p0']:.0f}\\,\\%\\\\\\midrule"]
nm = {'lcu': 'Fourier-LCU variant', 'unbalanced': 'unbalanced penalty', 'xy': 'XY mixer'}
for m in ('lcu', 'unbalanced', 'xy'):
    for mode in ('plain', 'biased'):
        L.append(f"{nm[m]}, {mode} & {B[m][mode]['feas']:.2f} & -- & {B[m][mode]['quality']:.2f} & --\\\\")
L += ['\\bottomrule\\end{tabular}\\end{table*}']
open('tables/tilted_prep.tex', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))

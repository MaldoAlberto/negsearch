"""tables/core_baselines.tex from results/bs_*.json (mean +- std over instances, p=1 unless stated)."""
import os, sys, json, glob
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); os.chdir(ROOT)
import numpy as np
runs = {}
for f in sorted(glob.glob('results/bs_n*_s*.json')):
    tag = os.path.basename(f).split('_s')[0]            # bs_n64 / bs_n100
    runs.setdefault(tag, []).extend(json.load(open(f)))
names = [('ours_p1', 'ours ($F$)'), ('lcu_ws_p1', 'Fourier-LCU, ws'), ('unb_ws_p1', 'unbalanced, ws'), ('xy_ws_p1', 'XY mixer, ws')]
def ms(v, fmt):
    v = np.array(v, float); return ('$' + fmt % v.mean() + ('\\pm' + fmt % v.std() if len(v) > 1 else '') + '$')
L = ['\\begin{table*}[t]\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}',
     '\\caption{\\orig{compiled counts; ideal feasibility, quality and $P(\\mathrm{opt})$ from exact dense simulation; per-shot values from the ESP and depolarising model; nothing executed} Stronger baselines on the residual core: ws = warm start: all baselines start from the warm-started product state of the Fourier-LCU paper (RY$(2\\arcsin\\sqrt{k/m})$ per qubit) and are trained with $12$ restarts ($p=1$). Mean $\\pm$ standard deviation over instances (objective seeds $\\times$ interface tuples). Feasible, quality (among feasible samples) and $P(\\mathrm{opt})$: ideal. Last column: probability of the optimum per shot under the depolarising model with the ESP of the compiled circuit.}\\label{tab:core_baselines}',
     '\\begin{tabular}{llrrrrrr}\\toprule', 'core & method & CZ & ESP & feas. & quality & $P$(opt) ideal & $P$(opt) per shot (model)\\\\\\midrule']
for tag, rs in sorted(runs.items(), key=lambda kv: kv[1][0]['n']):
    f = rs[0]['free']; first = True
    for key, nm in names:
        d = [r['methods'][key] for r in rs if key in r['methods'] and 'model' in r['methods'][key]]
        if not d: continue
        L.append((f"$n={rs[0]['n']}$, $f={f}$ ({len(d)} inst.)" if first else '') + f" & {nm} & {ms([x['compiled']['cz'] for x in d], '%.0f')} & {ms([x['compiled']['esp'] for x in d], '%.3f')} & "
                 f"{ms([100*x['ideal']['feas'] for x in d], '%.0f')}\\,\\% & {ms([x['ideal']['quality_feasible'] for x in d], '%.2f')} & {ms([100*x['ideal']['popt'] for x in d], '%.2f')}\\,\\% & "
                 f"{ms([100*x['model']['popt'] for x in d], '%.3f')}\\,\\%\\\\")
        first = False
    L.append('\\midrule')
L[-1] = '\\bottomrule'; L += ['\\end{tabular}\\end{table*}']
open('tables/core_baselines.tex', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))

"""tables/hw100_pred.tex: compiled counts and noise-model prediction for the 120-qubit experiment (reads hardware_circuits/hw100)."""
import os, sys, json, pickle
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); os.chdir(ROOT); sys.path.insert(0, ROOT)
import numpy as np
from negsearch import hw100 as H
meta = json.load(open('hardware_circuits/hw100/meta.json')); D = pickle.load(open('hardware_circuits/hw100/instance.pkl', 'rb')); inst, ideal = D['inst'], D['ideal']
names = dict(ours='ours, $p{=}1$ light', ours_full='ours, $p{=}1$ full', ours_p0='ours, preparation only', uniform='uniform (Hadamards)', lcu='Fourier-LCU, warm start', unbalanced='unbalanced, warm start', xy='XY mixer, warm start', ours_tilt_p0='ours, tilted preparation only', ours_tilt='ours, tilted, $p{=}1$ light', lcu_b='Fourier-LCU, tilted start', unbalanced_b='unbalanced, tilted start', xy_b='XY mixer, tilted start')
rows = []
for m in meta['methods']:
    st = meta['compile'][m]; esp = [s['esp'] for s in st]
    P = H.model_probs(ideal, esp, m)
    ideal_m = [H.metrics(b, d[m]) for b, d in zip(inst['blocks'], ideal)]; mod = [H.metrics(b, p) for b, p in zip(inst['blocks'], P)]
    rows.append((names[m], np.median([s['cz'] for s in st]), max(s['depth'] for s in st), np.median(esp), np.mean([x['feas'] for x in ideal_m]),
                 np.mean([x['feas'] for x in mod]), np.mean([x['quality'] for x in ideal_m]), np.mean([x['quality'] for x in mod])))
L = ['\\begin{table*}[t]\\centering\\small', '\\orig{compiled counts; ideal values from exact simulation; noisy values from the depolarising model; nothing executed}'.join(['\\caption{', ' Predictions for the 120-qubit experiment (twelve blocks of $10$ qubits in one circuit; averages over blocks; CZ and ESP: median per block, depth: maximum). Feasible: share of shots of a block that satisfy its constraints; quality: $1-$normalised cost among feasible shots (uniform on the feasible set: about $0.5$). Not measured.}\\label{tab:hw100_pred}']),
     '\\begin{tabular}{lrrrrrrr}\\toprule', 'method & CZ & depth & ESP & feas. ideal & feas. model & qual. ideal & qual. model\\\\\\midrule']
for r in rows: L.append(f"{r[0]} & {r[1]:.0f} & {r[2]} & {r[3]:.2f} & {r[4]:.3f} & {r[5]:.3f} & {r[6]:.2f} & {r[7]:.2f}\\\\")
L += ['\\bottomrule\\end{tabular}\\end{table*}']
open('tables/hw100_pred.tex', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))

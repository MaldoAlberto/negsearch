"""figures/constraint_scaling.pdf from results/constraint_scaling*.json, constraint_interface.json"""
import os, json
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); os.chdir(ROOT)
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
S = json.load(open('results/constraint_scaling_structured.json'))['structured']
R = json.load(open('results/constraint_scaling.json'))['random']
I = json.load(open('results/constraint_interface.json'))
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.1))
m = [r['m'] for r in S]
ax[0].plot(m, [r['ours']['cz'] for r in S], 'o-', label='ours, CZ'); ax[0].plot(m, [r['penalty']['cz'] for r in S], 's-', label='penalty layer, CZ')
ax[0].plot(m, [r['ours']['depth'] for r in S], 'o--', color='C0', alpha=.6, label='ours, depth'); ax[0].plot(m, [r['penalty']['depth'] for r in S], 's--', color='C1', alpha=.6, label='penalty, depth')
ax[0].set_xlabel('disjoint cardinality constraints $m$ ($n=24$)'); ax[0].set_yscale('log'); ax[0].set_title('(a) decomposable'); ax[0].legend(fontsize=7)
ms = range(9)
for key, lab, st in (('ours', 'ours, generic automaton', 'o-'), ('penalty', 'penalty layer', 's-')):
    ax[1].plot(list(ms), [np.mean([R[s][k][key]['cz'] for s in R]) for k in ms], st, label=lab + ', CZ')
ax[1].set_xlabel('overlapping constraints $m$ ($n=14$)'); ax[1].set_yscale('log'); ax[1].set_title('(b) overlapping'); ax[1].legend(fontsize=7)
ss = [r['s'] for r in I['1']]
ax[2].plot(ss, [np.mean([I[s][k]['cz_mean'] for s in I]) for k in range(len(ss))], 'o-', label='CZ per branch')
ax[2].plot(ss, [np.mean([I[s][k]['branches'] for s in I]) for k in range(len(ss))], 'd-', label='classical branches')
ax[2].set_xlabel('interface variables fixed classically $s$ ($m=8$)'); ax[2].set_yscale('log'); ax[2].set_title('(c) interface conditioning'); ax[2].legend(fontsize=7)
for a in ax: a.grid(alpha=.3)
plt.tight_layout(); os.makedirs('figures', exist_ok=True); plt.savefig('figures/constraint_scaling.pdf')

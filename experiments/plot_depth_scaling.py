"""Depth vs number of variables (results/depth_scaling.json -> figures/fig_depth_scaling.{pdf,png})."""
import os as _os, sys as _sys
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT)
import json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
D = json.load(open('results/depth_scaling.json'))
INK2, GRID = '#52514e', '#e6e5e0'
STY = {'Penalty QAOA layer (p=1)': ('#eb6834', 'o', '-'), 'NF preparation A_q': ('#2a78d6', 's', '-'),
       'NF preparation A_q (ancillas)': ('#2a78d6', 's', '--'), 'Hadfield QAOA layer (p=1)': ('#1baf7a', '^', '-'),
       'CG-QAOA layer (p=1)': ('#4a3aa7', 'D', '-'), 'CG-QAOA layer (ancillas)': ('#4a3aa7', 'D', '--'),
       'NF preparation A (3-SAT)': ('#2a78d6', 's', '-'), 'Grover iterate Q (3-SAT)': ('#eda100', 'v', '-')}
LAB = {'CG-QAOA layer (p=1)': 'CG-QAOA layer (no ancillas)', 'NF preparation A_q': 'NF preparation A_q (no ancillas)'}
def style(ax, title, yl):
    ax.set_title(title, loc='left', fontsize=9); ax.set_xlabel('number of variables n', color=INK2); ax.set_ylabel(yl, color=INK2)
    ax.set_yscale('log'); ax.grid(axis='y', color=GRID, lw=0.6)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
fig, axs = plt.subplots(1, 3, figsize=(13, 4.6))
mis = [r for r in D['mis'] if r['source'] == 'karate']
for key, ax, title in (('logical', axs[0], 'MIS (QOBLIB karate subgraphs): all-to-all'), ('routed', axs[1], 'MIS: routed on ibm_fez (heavy-hex)')):
    for m, (c, mk, ls) in STY.items():
        pts = [(r['n'], r[key]['depth']) for r in mis if r['method'] == m and r.get(key)]
        if pts: ax.plot(*zip(*pts), color=c, marker=mk, ls=ls, lw=1.6, ms=4, label=LAB.get(m, m))
    style(ax, title, 'circuit depth')
for m in ('NF preparation A (3-SAT)', 'Grover iterate Q (3-SAT)'):
    c, mk, ls = STY[m]; pts = [(r['n'], r['logical']['depth']) for r in D['sat'] if r['method'] == m]
    axs[2].plot(*zip(*pts), color=c, marker=mk, ls=ls, lw=1.6, ms=4, label=m)
style(axs[2], '3-SAT (density 4.2): all-to-all', 'circuit depth')
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.36, 0.0))
axs[2].legend(fontsize=7, frameon=False, loc='lower right')
plt.tight_layout(rect=(0, 0.14, 1, 1)); plt.savefig('figures/fig_depth_scaling.pdf'); plt.savefig('figures/fig_depth_scaling.png', dpi=150)
# second figure: two-qubit gates
fig, ax = plt.subplots(figsize=(5.5, 3.6))
for m, (c, mk, ls) in STY.items():
    pts = [(r['n'], r['routed']['twoq']) for r in mis if r['method'] == m and r.get('routed')]
    if pts: ax.plot(*zip(*pts), color=c, marker=mk, ls=ls, lw=1.6, ms=4, label=LAB.get(m, m))
style(ax, 'MIS: two-qubit gates on ibm_fez', 'CZ count'); ax.legend(fontsize=6.5, frameon=False)
plt.tight_layout(); plt.savefig('figures/fig_twoq_scaling.png', dpi=150); plt.savefig('figures/fig_twoq_scaling.pdf')
print('ok')

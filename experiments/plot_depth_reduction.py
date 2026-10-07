import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')))
import json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
D = json.load(open('results/depth_reduction.json')); n = [r['n'] for r in D]
INK2, GRID = '#52514e', '#e6e5e0'
fig, axs = plt.subplots(1, 3, figsize=(13, 3.9))
for k, c, mk in (('A[low-degree first]', '#eb6834', 'o'), ('A[degeneracy]', '#2a78d6', 's'), ('A[colouring layers]', '#1baf7a', '^')):
    axs[0].plot(n, [r[k]['routed']['depth'] for r in D], color=c, marker=mk, lw=1.6, ms=4, label=k[2:-1] + ' order')
axs[0].set_title('Preparation A_q: vertex order (ibm_fez)', loc='left', fontsize=9)
for k, c, mk in (('CG layer chain', '#4a3aa7', 'D'), ('CG layer tree', '#2a78d6', 's')):
    axs[1].plot(n, [r[k]['routed']['depth'] for r in D], color=c, marker=mk, lw=1.6, ms=4, label=k.replace('CG layer ', 'reflection: ') + (' (O(n))' if 'chain' in k else ' (O(log n))'))
axs[1].set_title('CG-QAOA layer: reflection synthesis (ibm_fez)', loc='left', fontsize=9)
for k, c, ls in (('penalty default', '#eb6834', '-'), ('penalty swap strategy', '#eda100', '--')):
    axs[2].plot(n, [r[k]['routed']['depth'] for r in D], color=c, ls=ls, marker='o', lw=1.6, ms=4, label=k.replace('penalty ', '') + ': depth')
    axs[2].plot(n, [r[k]['routed']['twoq'] for r in D], color=c, ls=ls, marker='x', lw=1.0, ms=5, alpha=0.7, label=k.replace('penalty ', '') + ': CZ count')
axs[2].set_title('Penalty QAOA layer: SABRE vs line swap strategy', loc='left', fontsize=9)
for ax in axs:
    ax.set_yscale('log'); ax.set_xlabel('number of variables n', color=INK2); ax.grid(axis='y', color=GRID, lw=0.6)
    ax.legend(fontsize=7, frameon=False); [ax.spines[s].set_visible(False) for s in ('top', 'right')]
axs[0].set_ylabel('routed depth / CZ', color=INK2)
plt.tight_layout(); plt.savefig('figures/fig_depth_reduction.png', dpi=150); plt.savefig('figures/fig_depth_reduction.pdf'); print('ok')

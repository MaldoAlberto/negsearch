import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')))
import json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
D = json.load(open('results/depth_to_solution_3sat.json')); n = [r['n'] for r in D]
INK2, GRID = '#52514e', '#e6e5e0'
fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.9))
axs[0].plot(n, [r['depth_round_Grover'] for r in D], 'v-', color='#eb6834', lw=1.6, ms=4, label='Grover round (uniform H)')
axs[0].plot(n, [r['depth_round_NF'] for r in D], 's-', color='#2a78d6', lw=1.6, ms=4, label='NF round (oracle + A + A^dagger + S0)')
axs[0].plot(n, [r['depth_A'] for r in D], 's--', color='#2a78d6', lw=1.2, ms=3, label='NF preparation A only')
axs[0].set_title('Depth of one amplification round (3-SAT)', loc='left', fontsize=9)
axs[1].plot(n, [r['total_depth_Grover'] for r in D], 'v-', color='#eb6834', lw=1.6, ms=4, label='Grover: rounds x depth')
axs[1].plot(n, [r['total_depth_NF'] for r in D], 's-', color='#2a78d6', lw=1.6, ms=4, label='Negation-forced: A + rounds x depth')
for r in D[::2]:
    axs[1].annotate((lambda x: f'{x:.1f}x' if x < 10 else f'{x:.0f}x')(r['total_depth_Grover']/r['total_depth_NF']), (r['n'], r['total_depth_NF']), textcoords='offset points', xytext=(0, 7), ha='center', fontsize=7, color=INK2)
axs[1].set_title('Total depth to reach a solution (median of 3 instances)', loc='left', fontsize=9)
for ax in axs:
    ax.set_yscale('log'); ax.set_xlabel('number of variables n', color=INK2); ax.set_ylabel('circuit depth (all-to-all)', color=INK2)
    ax.grid(axis='y', color=GRID, lw=0.6); ax.legend(fontsize=7.5, frameon=False, loc='upper left')
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig('figures/fig_depth_to_solution.png', dpi=150); plt.savefig('figures/fig_depth_to_solution.pdf'); print('ok')

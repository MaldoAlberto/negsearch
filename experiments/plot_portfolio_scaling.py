"""Figure for the MPS scaling study: figures/fig_portfolio_scaling.{png,pdf}."""
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT)
import json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = json.load(open('results/portfolio_scaling_mps.json'))
keys = sorted(R, key=lambda k: R[k]['qubits'])
n = [R[k]['qubits'] for k in keys]
SER = [('uniform', 'uniform random', '#9a9893', 'o', '-'),
       ('unbalanced p=1', 'penalty QAOA (unbalanced), p=1', '#eb6834', 'o', '-'),
       ('A_q classical', 'counts from A_q + random subset (classical)', '#1baf7a', '^', '-'),
       ('hybrid p=1', 'hybrid: A_q counts + Dicke + XY, p=1', '#4a3aa7', 's', '-')]


def style(ax):
    ax.grid(axis='y', color='#e4e3df', lw=0.8); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'): ax.spines[s].set_color('#9a9893')
    ax.tick_params(colors='#52514e', labelsize=8)
    ax.set_xscale('log'); ax.set_xticks(n); ax.set_xticklabels([str(v) for v in n]); ax.minorticks_off()
    ax.set_xlabel('decision qubits (QOBLIB instance)', fontsize=8, color='#52514e')


fig, axs = plt.subplots(2, 2, figsize=(10, 7.2))
FLOOR = 2e-4
ax = axs[0][0]; style(ax)
for k, lab, c, m, ls in SER:
    y = [max(R[i][k]['p_feas'], FLOOR) for i in keys]
    ax.plot(n, y, ls, marker=m, color=c, lw=2, ms=6, mec='white', mew=1.2, label=lab)
ax.set_yscale('log'); ax.set_ylim(FLOOR * 0.7, 1.6); ax.set_ylabel('P(feasible)', fontsize=9)
ax.axhline(FLOOR, color='#d7d6d2', lw=0.8, ls=':'); ax.text(n[0], FLOOR * 1.25, '0 of 2000 shots', fontsize=7, color='#52514e')
ax.set_title('(a) feasible samples', fontsize=9, loc='left')

ax = axs[0][1]; style(ax)
for k, lab, c, m, ls in SER:
    ax.plot(n, [R[i][k]['quality'] for i in keys], ls, marker=m, color=c, lw=2, ms=6, mec='white', mew=1.2)
ax.set_ylim(0, 1); ax.set_ylabel('normalised quality (infeasible = 0)', fontsize=9)
ax.set_title('(b) solution quality per shot', fontsize=9, loc='left')

ax = axs[1][0]; style(ax)
for k, lab, c, m, ls in SER[1:]:
    xs = [R[i]['qubits'] for i in keys if R[i][k]['best_gap'] is not None]
    ys = [R[i][k]['best_gap'] for i in keys if R[i][k]['best_gap'] is not None]
    ax.plot(xs, ys, ls, marker=m, color=c, lw=2, ms=6, mec='white', mew=1.2)
ax.set_ylabel('best of 2000 shots: gap to optimum\n' r'$(f_{best}-f_{opt})/(f_{worst}-f_{opt})$', fontsize=9)
ax.set_ylim(bottom=0); ax.set_title('(c) best solution found (lower is better)', fontsize=9, loc='left')

ax = axs[1][1]; style(ax)
H = [(k, R[k]) for k in keys if R[k]['hybrid p=1'].get('hardware')]
xs = [r['qubits'] for _, r in H]
ax.plot(xs, [r['unbalanced p=1']['hardware']['cz'] for _, r in H], '-o', color='#eb6834', lw=2, ms=6, mec='white', mew=1.2,
        label='penalty QAOA (unbalanced), p=1')
ax.plot(xs, [r['hybrid p=1']['hardware']['cz'] for _, r in H], '-s', color='#4a3aa7', lw=2, ms=6, mec='white', mew=1.2,
        label='hybrid, one coherent circuit')
ax.plot(xs, [r['hybrid p=1']['hardware_per_shot_version']['cz'] for _, r in H], '--s', color='#4a3aa7', lw=2, ms=6,
        mfc='white', mew=1.4, label='hybrid, counts drawn per shot')
ax.set_yscale('log'); ax.set_ylabel('CZ gates on ibm_fez (156 qubits)', fontsize=9)
ax.legend(fontsize=7.5, frameon=False, loc='upper left'); ax.set_title('(d) hardware cost (circuits that fit)', fontsize=9, loc='left')
labels = ', '.join(f"{R[k]['qubits']}: {k}" for k in keys)
h, l = axs[0][0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.025))
fig.text(0.5, 0.005, 'instances (decision qubits: key) — ' + labels, ha='center', fontsize=6.8, color='#52514e')
fig.tight_layout(rect=(0, 0.095, 1, 1))
fig.savefig('figures/fig_portfolio_scaling.png', dpi=200); fig.savefig('figures/fig_portfolio_scaling.pdf')
print('ok')

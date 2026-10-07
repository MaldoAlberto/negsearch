"""Figures and tables for the QOBLIB portfolio study.
Reads results/portfolio_qaoa.json, portfolio_penalty_scan*.json, portfolio_resources.json, portfolio_noisy.json."""
import os as _os, sys as _sys
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..'))
_os.chdir(ROOT); _sys.path.insert(0, ROOT)
import json, glob
import matplotlib; matplotlib.use('Agg'); import matplotlib.ticker
import matplotlib.pyplot as plt

METHODS = [  # key, label, colour (reference categorical palette, fixed order)
    ('penalty_slack', 'Penalty QAOA, slack QUBO (QOBLIB model)', '#2a78d6'),
    ('penalty_unbalanced', 'Penalty QAOA, unbalanced (no slack)', '#eb6834'),
    ('A_init_Xmixer', 'A_q initial state + X mixer', '#1baf7a'),
    ('A_init_XYmixer', 'A_q initial state + XY mixer (hybrid)', '#4a3aa7'),
    ('A_init_Grover_mixer', 'A_q + Grover mixer (CG-QAOA)', '#e34948'),
]
NAMES = {'a003_t02': 'po_a003_t02_orig', 'a004_t02': 'po_a004_t04_orig (2 periods)', 'a005_t02': 'po_a005_t04_orig (2 periods)'}


def merged():
    """Main results with the penalty baselines replaced by the best alpha over all scans (fair tuning)."""
    R = json.load(open('results/portfolio_qaoa.json'))
    for key in R:                                  # attach optimal angles to each penalty entry
        for pk, res in R[key].items():
            if pk.startswith('p') and pk[1:].isdigit():
                for m, pre in (('penalty_slack', 'sl'), ('penalty_unbalanced', 'unb')):
                    if m in res: res[m]['x'] = res['params'][f"{pre}{res[m]['alpha']}"]
                for m, pre in (('A_init_Xmixer', 'warm'), ('A_init_XYmixer', 'xy'), ('A_init_Grover_mixer', 'gm')):
                    if m in res: res[m]['x'] = res['params'][pre]
    for fn in sorted(glob.glob('results/portfolio_penalty_scan*.json')):
        for key, rows in json.load(open(fn)).items():
            if key not in R: continue
            for pk, res in rows.items():
                if pk not in R[key]: continue
                for m in ('penalty_slack', 'penalty_unbalanced'):
                    if m in res and (m not in R[key][pk] or res[m]['quality'] > R[key][pk][m]['quality']):
                        R[key][pk][m] = dict(res[m], x=res['params'][f"{m}{res[m]['alpha']}"])
    return R


def style(ax):
    ax.grid(axis='y', color='#e4e3df', lw=0.8); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'): ax.spines[s].set_color('#9a9893')
    ax.tick_params(colors='#52514e', labelsize=8)


if __name__ == '__main__':
    R = merged()
    keys = [k for k in NAMES if k in R]
    os_ = _os
    os_.makedirs('figures', exist_ok=True); os_.makedirs('tables', exist_ok=True)
    # ---------------- quality and P(opt) vs p ----------------
    fig, axs = plt.subplots(2, len(keys), figsize=(3.6 * len(keys), 5.6), squeeze=False)
    for j, key in enumerate(keys):
        ps = sorted(int(k[1:]) for k in R[key] if k.startswith('p') and k[1:].isdigit())
        for row, metric, ylab in ((0, 'quality', 'normalised quality'), (1, 'p_opt', 'P(optimum)')):
            ax = axs[row][j]; style(ax)
            for m, lab, col in METHODS:
                ys = [R[key][f'p{p}'][m][metric] for p in ps if m in R[key][f'p{p}']]
                if not ys: continue
                y0 = R[key]['A_alone'][metric] if m.startswith('A_') else R[key]['uniform'][metric]
                ax.plot([0] + ps[:len(ys)], [y0] + ys, '-o', color=col, lw=2, ms=4.5, label=lab,
                        markeredgecolor='white', markeredgewidth=1.2)
            ax.set_xticks([0] + ps); ax.set_xlabel('QAOA layers p (0 = initial state)', fontsize=8, color='#52514e')
            if j == 0: ax.set_ylabel(ylab, fontsize=9, color='#0b0b0b')
            if metric == 'p_opt': ax.set_yscale('log'); ax.set_ylim(bottom=1e-6)
            else: ax.set_ylim(0, 1)
            if row == 0: ax.set_title(f"{NAMES[key]}\n{R[key]['qubits_x']} decision qubits", fontsize=9)
    h, l = axs[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig('figures/fig_portfolio_quality.png', dpi=200); fig.savefig('figures/fig_portfolio_quality.pdf')
    # ---------------- two-qubit gates on ibm_fez vs p ----------------
    if os_.path.exists('results/portfolio_resources.json'):
        C = json.load(open('results/portfolio_resources.json'))
        fig, axs = plt.subplots(1, len(keys), figsize=(3.6 * len(keys), 3.2), squeeze=False)
        for j, key in enumerate(keys):
            ax = axs[0][j]; style(ax)
            for m, lab, col in METHODS:
                c = C[key][m]; c1, inc = c['twoq'], c['p2_twoq'] - c['twoq']
                ax.plot([1, 2, 3], [c1 + (p - 1) * inc for p in (1, 2, 3)], '-o', color=col, lw=2, ms=4.5,
                        label=lab, markeredgecolor='white', markeredgewidth=1.2)
            ax.set_yscale('log'); ax.set_xticks([1, 2, 3]); ax.set_xlabel('QAOA layers p', fontsize=8, color='#52514e')
            if j == 0: ax.set_ylabel('CZ gates on ibm_fez', fontsize=9)
            ax.set_title(NAMES[key], fontsize=9)
        h, l = axs[0][0].get_legend_handles_labels()
        fig.legend(h, l, loc='lower center', ncol=3, fontsize=8, frameon=False)
        fig.tight_layout(rect=(0, 0.17, 1, 1))
        fig.savefig('figures/fig_portfolio_cz.png', dpi=200); fig.savefig('figures/fig_portfolio_cz.pdf')
    # ---------------- markdown table ----------------
    qf = lambda r, v: (r['f_worst_feasible'] - v['mean_obj_feasible']) / (r['f_worst_feasible'] - r['f_opt'])
    L = ['| instance | qubits (ours / slack QUBO) | method | p | P(feasible) | quality | quality of feasible samples | P(opt) |',
         '|---|---|---|---|---|---|---|---|']
    for key in keys:
        r = R[key]
        L.append(f"| {NAMES[key]} | {r['qubits_x']} / {r['qubits_slack_qubo']} | uniform sampling | 0 | {r['uniform']['p_feas']:.3f} | {r['uniform']['quality']:.3f} | {qf(r, r['uniform']):.3f} | {r['uniform']['p_opt']:.1e} |")
        L.append(f"| | | A_q alone (q={r['A_alone']['q']:.2f}) | 0 | {r['A_alone']['p_feas']:.3f} | {r['A_alone']['quality']:.3f} | {qf(r, r['A_alone']):.3f} | {r['A_alone']['p_opt']:.1e} |")
        for p in (1, 2, 3):
            if f'p{p}' not in r: continue
            for m, lab, _ in METHODS:
                if m in r[f'p{p}']:
                    v = r[f'p{p}'][m]
                    L.append(f"| | | {lab} | {p} | {v['p_feas']:.3f} | {v['quality']:.3f} | {qf(r, v):.3f} | {v['p_opt']:.1e} |")
    open('tables/portfolio.md', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


def noisy_figure():
    """Noisy (ibm_fez model) quality vs CZ count on po_a003_t02_orig for every circuit simulated with noise."""
    pts = []   # (label, family, cz, quality, p_opt)
    if _os.path.exists('results/portfolio_noisy.json'):
        for k, v in json.load(open('results/portfolio_noisy.json')).items():
            fam = 'A_q alone' if k.startswith('A_q') else ('hybrid' if 'XY' in k else 'penalty')
            lab = {'A_q alone': 'A_q (first version)'}.get(k, k.replace('A_init_XYmixer', 'hybrid v1').replace('penalty_', ''))
            pts.append((lab, fam, v['cz'], v['noisy']['quality'], v['noisy']['p_opt']))
    if _os.path.exists('results/portfolio_hardware_opt.json'):
        for k, v in json.load(open('results/portfolio_hardware_opt.json'))['a003_t02']['hardware'].items():
            if 'noisy' not in v or k.startswith('unbalanced'): continue
            fam = 'A_q alone' if k.startswith('A_q') else 'hybrid'
            pts.append((k.replace(' (lean)', ' v2 (lean)'), fam, v['cz'], v['noisy']['quality'], v['noisy']['p_opt']))
    if _os.path.exists('results/portfolio_noisy_lean.json'):
        for k, v in json.load(open('results/portfolio_noisy_lean.json')).items():
            pts.append((k.split(' (')[0] + ' v3 (path XY, linear phase)', 'hybrid', v['cz'], v['noisy']['quality'], v['noisy']['p_opt']))
    if _os.path.exists('results/portfolio_midmeasure_noisy.json'):
        for k, v in json.load(open('results/portfolio_midmeasure_noisy.json')).items():
            pts.append((k.replace('sector + Dicke', 'counts + Dicke'), 'hybrid', v['mean_cz_per_shot'], v['noisy']['quality'], v['noisy']['p_opt']))
    keep = ('unbalanced p=1', 'unbalanced p=2', 'slack p=1', 'slack p=2', 'A_q alone v2 (lean)', 'hybrid v1 p=1',
            'hybrid p=1 v3 (path XY, linear phase)', 'counts + Dicke p=1', 'counts + Dicke p=2')
    ren = {'A_q alone v2 (lean)': 'A_q alone', 'hybrid v1 p=1': 'hybrid v1', 'hybrid p=1 v3 (path XY, linear phase)': 'hybrid v3',
           'counts + Dicke p=1': 'counts+Dicke p=1', 'counts + Dicke p=2': 'counts+Dicke p=2'}
    pts = [(ren.get(p[0], p[0]),) + p[1:] for p in pts if p[0] in keep]
    style_ = {'penalty': ('#2a78d6', 'o', 'penalty QAOA (slack / unbalanced)'), 'hybrid': ('#eb6834', 's', 'A_q + XY hybrid'),
              'A_q alone': ('#1baf7a', '^', 'A_q alone (classically samplable)')}
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, j, yl in ((axs[0], 3, 'quality under ibm_fez noise'), (axs[1], 4, 'P(optimum) under ibm_fez noise')):
        style(ax)
        for fam, (col, mk, lab) in style_.items():
            P_ = [p for p in pts if p[1] == fam]
            ax.scatter([p[2] for p in P_], [p[j] for p in P_], s=46, color=col, marker=mk, label=lab,
                       edgecolor='white', linewidth=1.2, zorder=3)
            for p in P_:
                ax.annotate(p[0], (p[2], p[j]), xytext=(5, 3), textcoords='offset points', fontsize=6.5, color='#52514e')
        ax.set_xscale('log'); ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_xlabel('CZ gates on ibm_fez', fontsize=8, color='#52514e'); ax.set_ylabel(yl, fontsize=9)
    axs[0].legend(fontsize=7.5, frameon=False, loc='lower left')
    fig.suptitle('po_a003_t02_orig, ibm_fez noise model (FakeFez), 2000–4000 shots', fontsize=9)
    fig.tight_layout(); fig.savefig('figures/fig_portfolio_noisy.png', dpi=200); fig.savefig('figures/fig_portfolio_noisy.pdf')


if __name__ == '__main__':
    noisy_figure()

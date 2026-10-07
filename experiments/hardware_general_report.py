"""Report for results/hardware_general_*.json: P(feasible), P(optimum) with 95 % Wilson intervals, per-seed spread, and three tests.
  H1  A_q feasible share > uniform-state feasible share           (feasibility by construction survives the device)
  H2  Aq+iter P(optimum) > A_q P(optimum)                          (one amplification iteration helps despite noise)
  H3  Aq+iter P(optimum) > noise-control P(optimum)                (the gain is logic, not an artefact of depth)
One-sided two-proportion z-tests on pooled shots; per-seed values are listed because layouts differ.
Usage: python experiments/hardware_general_report.py results/hardware_general_<tag>.json [--pred results/hardware_general_dryrun.json]"""
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT)
import sys, json, math
import numpy as np
from scipy.stats import norm
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES = ['uniform', 'Aq', 'Aq+iter', 'noise ctrl']


def wilson(k, n, z=1.96):
    if n == 0:
        return 0, 0, 0
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0, c - h), min(1, c + h)


def zt(k1, n1, k2, n2):
    """one-sided p-value for p1 > p2"""
    p = (k1 + k2) / (n1 + n2); se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    return 1 - norm.cdf((k1 / n1 - k2 / n2) / se)


def tally(D, mode):
    n = D['n']; feas = np.array(D['feas']); good = np.array(D['good'])
    out = {}
    for key, counts in D['raw'][mode]['counts'].items():
        s, name = key.split('|'); s = int(s)
        kf = kg = tot = 0
        for b, v in counts.items():
            bits = [int(ch) for ch in b.replace(' ', '')[::-1]]
            idx = int(''.join(map(str, bits)), 2)
            tot += v; kf += v * feas[idx]; kg += v * good[idx]
        out.setdefault(name, []).append((s, int(kf), int(kg), int(tot)))
    return out


def report(path, pred=None):
    D = json.load(open(path))
    banner = ['**DRY RUN: noise-model prediction, NOT hardware data.**', ''] if 'Fake' in D['backend'] else []
    lines = banner + [f"# Hardware report: {D['backend']}  ({D['instance']})", '',
             f"shots per circuit {D['shots']}, seeds {D['seeds']}, calibration {D.get('calibration')}", '',
             f"reference: uniform P(feasible) = {D['reference']['uniform_p_feasible']:.4f}, ideal P(optimum): A_q {D['reference']['ideal_Aq_p_optimum']:.2f}, "
             f"after one iteration {D['reference']['ideal_iter_p_optimum']:.2f}", '']
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6)); res_all = {}
    for mi, mode in enumerate(D['modes']):
        T = tally(D, mode); res = {}
        for name in NAMES:
            ks = np.array([[t[1], t[2], t[3]] for t in T[name]])
            kf, kg, tot = ks.sum(0)
            res[name] = dict(feas=wilson(kf, tot), opt=wilson(kg, tot), kf=int(kf), kg=int(kg), tot=int(tot),
                             seeds_feas=[round(t[1] / t[3], 3) for t in T[name]], seeds_opt=[round(t[2] / t[3], 3) for t in T[name]],
                             cz=[D['cz'][f'{t[0]}|{name}'] for t in T[name]])
        res_all[mode] = res
        lines += [f'## mode: {mode}', '', '| circuit | CZ (per seed) | P(feasible) [95% CI] | per seed | P(optimum) [95% CI] | per seed |', '|---|---|---|---|---|---|']
        for name in NAMES:
            r = res[name]
            lines.append(f"| {name} | {r['cz']} | {r['feas'][0]:.3f} [{r['feas'][1]:.3f}, {r['feas'][2]:.3f}] | {r['seeds_feas']} | "
                         f"{r['opt'][0]:.3f} [{r['opt'][1]:.3f}, {r['opt'][2]:.3f}] | {r['seeds_opt']} |")
        u = D['reference']['uniform_p_feasible']
        h1 = zt(res['Aq']['kf'], res['Aq']['tot'], res['uniform']['kf'], res['uniform']['tot'])
        h2 = zt(res['Aq+iter']['kg'], res['Aq+iter']['tot'], res['Aq']['kg'], res['Aq']['tot'])
        h3 = zt(res['Aq+iter']['kg'], res['Aq+iter']['tot'], res['noise ctrl']['kg'], res['noise ctrl']['tot'])
        verdict = lambda p: 'SUPPORTED' if p < 0.01 else ('weak' if p < 0.05 else 'NOT supported')
        lines += ['', f'* H1 (A_q feasible share > measured uniform share): p = {h1:.2g}  -> {verdict(h1)}',
                  f'* H2 (one iteration raises P(optimum) over A_q alone): p = {h2:.2g}  -> {verdict(h2)}',
                  f'* H3 (iteration beats the depth-matched noise control): p = {h3:.2g}  -> {verdict(h3)}', '']
        for ax, key, title in zip(axs, ('feas', 'opt'), ('P(feasible sample)', 'P(optimal sample)')):
            xs = np.arange(len(NAMES)) + (mi - (len(D['modes']) - 1) / 2) * 0.28
            ax.errorbar(xs, [res[nm][key][0] for nm in NAMES],
                        yerr=[[res[nm][key][0] - res[nm][key][1] for nm in NAMES], [res[nm][key][2] - res[nm][key][0] for nm in NAMES]],
                        fmt='o', capsize=3, label=f"{D['backend']} / {mode}")
            ax.set_xticks(range(len(NAMES))); ax.set_xticklabels(NAMES, fontsize=8); ax.set_title(title, fontsize=10, loc='left'); ax.set_ylim(0, 1.05)
    if pred:
        P = json.load(open(pred)); Tp = tally(P, P['modes'][0])
        for ax, j in zip(axs, (1, 2)):
            ax.plot(range(len(NAMES)), [sum(t[j] for t in Tp[nm]) / sum(t[3] for t in Tp[nm]) for nm in NAMES], 'k_', ms=18, mew=2, label='noise-model prediction')
    for ax in axs:
        for s in ('top', 'right'): ax.spines[s].set_visible(False)
    axs[0].axhline(D['reference']['uniform_p_feasible'], color='#9a9893', ls='--', lw=1)
    axs[0].legend(fontsize=7, frameon=False); plt.tight_layout()
    base = _os.path.splitext(_os.path.basename(path))[0]
    fig.savefig(f'figures/{base}.png', dpi=180)
    open(f'tables/{base}.md', 'w').write('\n'.join(lines))
    print('\n'.join(lines)); print(f'\nwrote tables/{base}.md and figures/{base}.png')


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    pred = sys.argv[sys.argv.index('--pred') + 1] if '--pred' in sys.argv else None
    if pred in args: args.remove(pred)
    report(args[0], pred)

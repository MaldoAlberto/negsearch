"""tables/hw100_cost.tex: total depth, two-qubit depth, CZ and scheduled duration of the 120-qubit circuits compiled for the FakeFez snapshot (ibm_fez calibration data),
and the QPU time per run: shots x (circuit duration + repetition delay 250 us).  Estimate; nothing executed."""
import os, json
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); os.chdir(ROOT)
rows = {r[0]: r for r in json.load(open('results/hw100_depths.json'))}
SH = 10000; REP = 250e-6
names = dict(ours='ours $p{=}1$ light', ours_full='ours $p{=}1$ full', ours_p0='ours, preparation', uniform='uniform', lcu='Fourier-LCU, ws', unbalanced='unbalanced, ws', xy='XY mixer, ws',
             ours_tilt_p0='ours tilted, prep.', ours_tilt='ours tilted $p{=}1$', lcu_b='LCU, tilted start', unbalanced_b='unbalanced, tilted start', xy_b='XY, tilted start')
L = ['\\begin{table}[t]\\centering\\scriptsize\\setlength{\\tabcolsep}{2pt}',
     '\\caption{\\orig{compiled counts on the \\texttt{ibm\\_fez} calibration snapshot; time is an estimate; nothing executed} The $120$-qubit circuits (twelve blocks in one circuit): total depth, two-qubit depth, CZ gates (all blocks), scheduled duration and estimated QPU time for $10^4$ shots (duration plus the default repetition delay of $250\\,\\mu$s per shot; job overhead not included).}\\label{tab:hw100_cost}',
     '\\begin{tabular}{lrrrrr}\\toprule', 'circuit & depth & 2q depth & CZ & dur. ($\\mu$s) & QPU s\\\\\\midrule']
tot = 0
for m in names:
    _, d, d2, cz, _, dur = rows[m]; t = SH * (REP + dur); tot += t
    L.append(f"{names[m]} & {d} & {d2} & {cz} & {1e6*dur:.0f} & {t:.1f}\\\\")
L += ['\\midrule', f"all twelve circuits & & & & & {tot:.0f}\\\\", '\\bottomrule\\end{tabular}\\end{table}']
open('tables/hw100_cost.tex', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L)); print('4 circuits (ours p0, ours, lcu, unbalanced):', sum(SH * (REP + rows[m][5]) for m in ('ours_p0', 'ours', 'lcu', 'unbalanced')))

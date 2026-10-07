"""LaTeX tables for the 'ideal / optimised / executable' analysis of the large case (reads results/*.json)."""
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT)
import json, math
import numpy as np


def sci(x):
    if x == 0: return '0'
    m, e = f'{x:.1e}'.split('e'); return f'${m}\\!\\cdot\\!10^{{{int(e)}}}$'


def th(n):
    return f'{n:,}'.replace(',', '\\,')


def status(esp):
    return 'run today' if esp >= 0.3 else ('frontier' if esp >= 0.05 else 'out of reach')


# ---------- table 1: QAOA p=1, ideal vs optimised
rows = json.load(open('results/qaoa_realistic.json')); Q = json.load(open('results/qaoa_realistic_quality.json'))
qual = {8: {'GM (first version)': Q['n8']['GM p=1']['quality'], 'SWAP mixers': Q['n8']['SWAP path p=1']['quality'], 'SWAP + prune 25%': Q['n8']['SWAP path + prune 25% p=1']['quality'], 'preparation only': Q['n8']['preparation only']['quality']},
        16: {'GM (first version)': Q['n16']['GM p=1']['quality'], 'SWAP mixers': Q['n16']['SWAP path p=1']['quality'], 'SWAP + prune 25%': Q['n16']['SWAP path + prune 25%  p=1'.replace('  ', ' ')]['quality'], 'preparation only': Q['n16']['preparation only']['quality']}}
names = {'GM (first version)': 'Grover mixer (first version)', 'SWAP mixers': 'pair-exchange mixers', 'SWAP + prune 10%': None, 'SWAP + prune 25%': '\\quad + objective pruning 25\\%', 'preparation only': 'preparation only'}
L = []
L.append('\\begin{table*}[t]\n\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}')
L.append('\\caption{Multi-constraint case, one QAOA layer ($p=1$), one interface tuple, compiled to the \\texttt{ibm\\_fez} topology (best of eight transpiler seeds; nothing executed). ESP is the product of one minus the error of every gate. Status: ``run today\'\' ESP $\\ge0.3$, ``frontier\'\' $0.05$--$0.3$, ``out of reach\'\' below $0.05$. Quality is measured on the exact feasible set, averaged over all interface tuples (higher is better, $1$ is the optimum); the preparation alone is the uniform feasible state. Pruning drops the objective couplings below $25\\%$ of the largest, which never affects feasibility. The pair-exchange mixer $e^{-i\\beta P}$ with $P=\\mathrm{SWAP}(\\mathrm{long}_j,\\mathrm{long}_k)\\,\\mathrm{SWAP}(\\mathrm{short}_j,\\mathrm{short}_k)$ conserves all counts and the exclusivity and needs the preparation registers uncomputed; the preparation-only row is that circuit without the QAOA layer (generic $A_q$, registers uncomputed).}\\label{tab:large_qaoa_real}')
L.append('\\begin{tabular}{llrrrrrll}\n\\toprule\n$n$ & variant & qubits & CZ & depth & ESP & $d/T_2$ & quality & status\\\\\n\\midrule')
cur = None
for r in rows:
    if r['variant'] == 'SWAP + prune 10%': continue
    n = r['n']
    if cur is not None and n != cur: L.append('\\midrule')
    cur = n
    q = qual.get(n, {}).get(r['variant']); qs = f'{q:.3f}' if q is not None else '--'
    esp = r['esp']; es = f'{esp:.2f}' if esp >= 0.01 else sci(esp)
    L.append(f"{n if (r['variant']=='GM (first version)') else ''} & {names[r['variant']]} & {r['qubits']} & {th(r['cz'])} & {th(r['depth'])} & {es} & {r['dur_t2']:.2f} & {qs} & {status(esp)}\\\\")
L.append('\\bottomrule\n\\end{tabular}\n\\end{table*}')
open('tables/large_qaoa_real.tex', 'w').write('\n'.join(L) + '\n')

# ---------- table 2: block decomposition
B = json.load(open('results/block_decomposition_sym.json'))
L = ['\\begin{table*}[t]\n\\centering\\footnotesize\\setlength{\\tabcolsep}{3.5pt}',
     '\\caption{QAOA as many small blocks. The interface is drawn classically and every (period, sector) block is a separate $p=1$ circuit (pair-exchange mixers, preparation from the symmetry-aware lowering of $A_q$, exact and without registers), while the other blocks are held fixed and enter as linear fields. Compiled to the \\texttt{ibm\\_fez} topology, three random interface tuples per instance; CZ and ESP are over all blocks.}\\label{tab:large_blocks}',
     '\\begin{tabular}{lrrlrrrr}\n\\toprule\n$A,T$ & $n$ & blocks & objective & qubits/block & CZ/block (med.\\,/\\,max) & ESP (med.\\,/\\,min) & CZ per sweep\\\\\n\\midrule']
for r in B:
    for lab, nm in (('full', 'full'), ('pruned', 'pruned 25\\%')):
        cz = []; esp = []; sweep = []
        for it in r['interfaces']:
            c = it['compile'][lab]; cz += [x['cz'] for x in c]; esp += [x['esp'] for x in c]; sweep.append(sum(x['cz'] for x in c))
        qb = 2 * r['sectors'][0] if len(set(r['sectors'])) == 1 else '--'
        nb = len(r['interfaces'][0]['compile'][lab])
        L.append(f"{r['A']},{r['T']} & {r['n']} & {nb} & {nm} & {qb} & {int(np.median(cz))}\\,/\\,{max(cz)} & {np.median(esp):.2f}\\,/\\,{min(esp):.2f} & {th(int(np.mean(sweep)))}\\\\")
L.append('\\bottomrule\n\\end{tabular}\n\\end{table*}')
open('tables/large_blocks.tex', 'w').write('\n'.join(L) + '\n')
# block quality summary (for the text)
for r in B:
    for lab in ('full', 'pruned'):
        g = []
        for it in r['interfaces']:
            m = it['methods'][lab]; rnd = m['random_feasible']; ex = m['exact_blocks']
            g.append(((rnd - m['uniform_same_shots']) / (rnd - ex), (rnd - m['qaoa_ideal']) / (rnd - ex), (rnd - m['qaoa_noisy_model']) / (rnd - ex)))
        g = np.array(g)
        print(f"blocks n={r['n']} [{lab}] share of the random->block-optimum gap closed: uniform {g[:,0].mean():.2f}, QAOA ideal {g[:,1].mean():.2f}, QAOA noisy model {g[:,2].mean():.2f}")

# ---------- table 3: Grover, fault tolerant
G = json.load(open('results/grover_future.json'))
L = ['\\begin{table*}[t]\n\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}',
     '\\caption{Amplitude amplification over the coherent state $A_q$ on a fault-tolerant machine (a model, not a measurement). A multi-controlled gate with $c$ controls costs $2(c-1)$ Toffoli gates, a Toffoli $7$ T gates, and one iteration is $A_q^\\dagger$, the reflection and $A_q$ (the threshold oracle is common to all methods and not counted). With success probability $a$ the amplification needs $\\frac\\pi4/\\sqrt a$ iterations and classical sampling from the same state needs $1/a$ samples. Times assume $1$--$10\\,\\mu$s per T gate and $1$--$10\\,\\mu$s per classical sample. $a^\\star$ is the success probability below which amplification is faster than classical sampling.}\\label{tab:large_grover_ft}',
     '\\begin{tabular}{lrrrrr}\n\\toprule\n$A,T$ & logical qubits & Toffoli / iteration & $a^\\star$ (optimistic\\,/\\,conservative) & time at $a=10^{-9}$ & time at $a=10^{-12}$\\\\\n\\midrule']
def t(x):
    return f'{x:.0e}'.replace('e+0', 'e').replace('e-0', 'e-')
def hum(s):
    if s < 60: return f'{s:.0f}\\,s'
    if s < 3600: return f'{s/60:.0f}\\,min'
    if s < 3600 * 48: return f'{s/3600:.1f}\\,h'
    return f'{s/86400:.0f}\\,d' if s >= 86400 * 10 else f'{s/86400:.1f}\\,d'
for r in G:
    l9 = [x for x in r['levels'] if x['a'] == 1e-9][0]; l12 = [x for x in r['levels'] if x['a'] == 1e-12][0]
    L.append(f"{r['A']},{r['T']} & {th(r['qubits'])} & {th(r['toffoli_iter'])} & {sci(r['a_breakeven_opt'])}\\,/\\,{sci(r['a_breakeven_cons'])} & {hum(l9['t_quantum_s_opt'])}--{hum(l9['t_quantum_s_cons'])} & {hum(l12['t_quantum_s_opt'])}--{hum(l12['t_quantum_s_cons'])}\\\\")
L.append('\\bottomrule\n\\end{tabular}\n\\end{table*}')
open('tables/large_grover_ft.tex', 'w').write('\n'.join(L) + '\n')
print('classical sampling at a=1e-9: 1e3-1e4 s; at 1e-12: 1e6-1e7 s (same for all cases)')

# ---------- table 4: preprocessing
P = json.load(open('results/preprocess_study.json'))
L = ['\\begin{table*}[t]\n\\centering\\footnotesize\\setlength{\\tabcolsep}{3.5pt}',
     '\\caption{Symmetry-aware lowering of $A_q$ for a block of $m$ interchangeable assets conditioned on $l$ longs and $s$ shorts (all units found interchangeable by the automatic check). CZ after routing to \\texttt{ibm\\_fez} (best of three seeds). The generic $A_q$ uses a one-hot automaton register, relative-phase Toffolis, and for QAOA the registers uncomputed; the lowering has no register, no uncomputation and no post-selection.}\\label{tab:large_symmetry}',
     '\\begin{tabular}{rrrrrrrr}\n\\toprule\n$m$ & $(l,s)$ & $|F|$ & generic: qubits & CZ & CZ uncomputed & lowering: qubits & CZ\\\\\n\\midrule']
for b in P['blocks']:
    L.append(f"{b['m']} & ({b['l']},{b['s']}) & {b['F']} & {b['generic_qubits']} & {b['generic_cz']} & {b['generic_cz_uncomputed']} & {b['sym_qubits']} & {b['sym_cz']}\\\\")
L.append('\\bottomrule\n\\end{tabular}\n\\end{table*}')
open('tables/large_symmetry.tex', 'w').write('\n'.join(L) + '\n')
L = ['\\begin{table*}[t]\n\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}',
     '\\caption{Variable order and automaton width for conflict constraints (random pair-exclusion graphs with repeated, subsumed and unit clauses injected). Width of the reduced automaton for the index order, a low-degree-first order, a degeneracy order, a colouring-layer order and the best order found by hill climbing on adjacent swaps (starting from the best of the four). Cleaning removes repeated and subsumed clauses and propagates unit clauses (forced variables stop being qubits); the repeated and subsumed clauses are injected, so the reduction in clauses is not an estimate for natural instances.}\\label{tab:large_order}',
     '\\begin{tabular}{rrrrrrrr}\n\\toprule\n$n$ & clauses (raw$\\to$clean) & qubits & index & low degree & degeneracy & colouring & searched\\\\\n\\midrule']
for r in P['clauses']:
    w = r['width_orders']
    L.append(f"{r['n']} & {r['clauses_raw']}$\\to${r['clauses_clean']} & {r['n_live']} & {w['index']} & {w['degree']} & {w['degeneracy']} & {w['colouring']} & {r['width_best']}\\\\")
L.append('\\bottomrule\n\\end{tabular}\n\\end{table*}')
open('tables/large_order.tex', 'w').write('\n'.join(L) + '\n')

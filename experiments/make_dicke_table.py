"""tables/dicke_opt.tex from results/dicke_optimised.json"""
import os, json
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); os.chdir(ROOT)
R = json.load(open('results/dicke_optimised.json'))
L = ['\\begin{table*}[t]\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}',
     '\\caption{\\orig{compiled counts; exact state checked for $n\\le12$; nothing executed} Dicke block $|D^n_k\\rangle$: CZ count and two-qubit depth, logical (all-to-all) and routed on the \\texttt{ibm\\_fez} coupling map (best of $10$ transpiler seeds; $6$ for $n\\ge16$). ref: generic controlled gates; 4cx: hand-decomposed doubly controlled RY; 3cx: that gate with $3$ CX, exact on the states the block produces.}\\label{tab:dicke_opt}',
     '\\begin{tabular}{lrrrrrr}\\toprule', '$(n,k)$ & logical ref & logical 3cx & routed ref & routed 4cx & routed 3cx & 2q depth ref $\\to$ 3cx\\\\\\midrule']
for r in R:
    L.append(f"$({r['n']},{r['k']})$ & {r['ref']['logical']['cz']} & {r['3cx']['logical']['cz']} & {r['ref']['routed']['cz']} & {r['4cx']['routed']['cz']} & {r['3cx']['routed']['cz']} & {r['ref']['routed']['depth2q']}$\\to${r['3cx']['routed']['depth2q']}\\\\")
L += ['\\bottomrule\\end{tabular}\\end{table*}']
open('tables/dicke_opt.tex', 'w').write('\n'.join(L) + '\n')

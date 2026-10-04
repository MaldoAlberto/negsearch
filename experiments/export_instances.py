"""Write every instance used in E2, E4/E5, E6 and the knapsack notebook as DIMACS files (instances/).
E3 instances are written by e3_scaling.py itself."""
import os as _os, sys as _sys
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..'))
_os.chdir(ROOT); _sys.path.insert(0, ROOT)
import json, os, random, sys
sys.path.insert(0, ROOT)
from negsearch.core import sample_satisfiable, CNF
def dump(path, f_n, clauses, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fh:
        for h in header: fh.write('c ' + h + '\n')
        fh.write(f'p cnf {f_n} {len(clauses)}\n')
        for c in clauses: fh.write(' '.join(map(str, c)) + ' 0\n')
cnt = 0
# E2: first batch (old driver) stored clauses; later batches regenerated from the worker seed
old = json.load(open('results/e2a.json')) if os.path.exists('results/e2a.json') else []
for k, r in enumerate(old):
    dump(f"instances/e2/a_{r['family']}_n{r['n']}_k{k}.cnf", r['n'], r['clauses'], [f"E2a family={r['family']} n={r['n']}", 'order ' + ' '.join(map(str, r['order']))]); cnt += 1
for part in 'ab':
    fn = f'results/e2{part}.jsonl'
    if not os.path.exists(fn): continue
    for line in open(fn):
        r = json.loads(line)
        if 'seed' not in r: continue
        rng = random.Random(r['seed']); f = sample_satisfiable(r['family'], r['n'], rng)
        o = list(range(1, r['n'] + 1)); rng.shuffle(o)
        path = f"instances/e2/{part}_{r['family']}_n{r['n']}_seed{r['seed']}.cnf"
        if not os.path.exists(path):
            dump(path, f.n, f.clauses, [f"E2{part} family={r['family']} n={r['n']} seed={r['seed']} (worker seed)", 'order ' + ' '.join(map(str, o))]); cnt += 1
# E4/E5
for fam in ['3-SAT', '4-SAT', '1-in-3-SAT', '3-COL']:
    fn = f'results/e45_{fam}.json'
    if not os.path.exists(fn): continue
    for k, r in enumerate(json.load(open(fn))):
        dump(f"instances/e45/{fam}_n{r['n']}_k{k}.cnf", r['n'], r['clauses'], [f"E4/E5 family={fam} n={r['n']}", 'order ' + ' '.join(map(str, r['order']))]); cnt += 1
print('written', cnt)

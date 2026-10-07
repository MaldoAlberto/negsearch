"""Generates the three notebooks of the ~100-qubit hardware package into notebooks/ (HW100_01_build.ipynb, HW100_02_run.ipynb, HW100_03_analyze.ipynb)."""
import os, nbformat as nbf
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); os.chdir(ROOT)
md = nbf.v4.new_markdown_cell; code = nbf.v4.new_code_cell

SETUP = '''import os, sys, json, pickle, time, warnings; warnings.filterwarnings('ignore')
root = os.path.abspath('.')
while not os.path.isdir(os.path.join(root, 'negsearch')): root = os.path.dirname(root)      # repository root
os.chdir(root); sys.path.insert(0, root)
import numpy as np
from negsearch import hw100 as H
OUT = 'hardware_circuits/hw100'; RES = 'results/hw100'; os.makedirs(OUT, exist_ok=True); os.makedirs(RES, exist_ok=True)
print('repository root:', root)'''

# ------------------------------------------------------------------------------------------------ notebook 1
nb1 = nbf.v4.new_notebook()
nb1.cells = [
md('''# HW100 · 1/3 — Build and compile the 120-qubit experiment

**Question.** Can feasible-by-construction states be produced *on a real chip at about 100 qubits*, where penalty / unbalanced / Fourier-LCU methods give almost no feasible samples?

**Setting.** Portfolio with 10 assets, 6 periods, 2 sectors, long/short variables: **n = 120 binary variables**. The interface (per period and sector counts of longs and shorts) is drawn classically; conditioned on it the problem splits into **12 independent blocks of 10 qubits**. All 12 blocks run in **one circuit of 120 qubits** on disjoint regions of the chip. Each block carries its own quadratic objective (the neighbours enter as linear fields).

**Methods (same block QUBO, same shots, same regions).**
`ours` (light p=1), `ours_full` (p=1, full objective and mixer), `ours_p0` (support-F preparation only), `uniform` (Hadamards), and three baselines that all use the **warm start of the Fourier-LCU paper** (RY(2·asin√(k/m)) per qubit): `lcu` (single-basis Fourier-LCU variant, trained), `unbalanced` (−l₁g + l₂g² + exclusivity, trained) and `xy` (XY mixer inside each count group, exclusivity by phase, trained).
The baselines receive the same block decomposition and the same count constraints, which is the setting most favourable to them.

**What this notebook does not need:** the IBM key (it uses the FakeFez snapshot) unless you set `USE_LIVE_BACKEND = True` to select regions with the live calibration. Do that right before running notebook 2, since calibration drifts.'''),
code(SETUP),
code('''# ---------------- configuration ----------------
USE_LIVE_BACKEND = False          # True: choose regions with the live calibration of BACKEND_NAME (needs the key file)
BACKEND_NAME = 'ibm_fez'
APIKEY = 'apikey_personal.json'   # local file, never committed:  {"apikey": "..."}
INSTANCE = None                   # optional IBM Cloud instance CRN
SEED_OBJ, SEED_IFACE = 5, 11
COMPILE_SEEDS = 4                 # transpiler seeds per block (best ESP kept)
PACK_TRIES = 3000'''),
md('## 1. Classical part: instance and interface'),
code('''inst = H.build_instance(SEED_OBJ, SEED_IFACE)
print('variables:', inst['nv'], '| blocks:', len(inst['blocks']), '| qubits per block:', H.NB_Q)
print('interface (per period: (longs, shorts) of sector 0 and sector 1):')
for t, row in enumerate(inst['iface']): print('  period', t, row)
print('feasible strings per block:', [int(b['mask'].sum()) for b in inst['blocks']], 'of', 2 ** H.NB_Q)
print('log2 of the number of globally feasible strings:', round(sum(np.log2(b['mask'].sum()) for b in inst['blocks']), 1))'''),
md('## 2. Train the angles (exact simulation of each 10-qubit block; ~1–2 min)'),
code('''t0 = time.time()
params = H.train_all(inst, seed=0)
ideal = H.ideal_probs(inst, params)
print(f'trained in {time.time()-t0:.0f}s')'''),
code('''import pandas as pd
rows = []
for m in H.METHODS:
    mt = [H.metrics(b, d[m]) for b, d in zip(inst['blocks'], ideal)]
    rows.append(dict(method=m, ideal_feasible=np.mean([x['feas'] for x in mt]), ideal_quality=np.mean([x['quality'] for x in mt]),
                     ideal_P_opt=np.mean([x['p_opt'] for x in mt])))
print('IDEAL (noiseless) per-block averages.  quality = 1 - normalised cost among feasible samples (uniform on F gives about 0.5)')
pd.DataFrame(rows).round(3)'''),
md('''## 3. Compile for the chip
Disjoint connected regions of 11 physical qubits (one spare for routing), broken couplers excluded, best regions to the heaviest blocks. Each block is transpiled onto *its own region only* (no cross-block routing) and the blocks are assembled into one circuit with one 120-bit classical register (bit 10·b + k = variable k of block b).'''),
code('''if USE_LIVE_BACKEND:
    svc, backend = H.connect(APIKEY, BACKEND_NAME, INSTANCE)
else:
    from qiskit_ibm_runtime.fake_provider import FakeFez
    backend = FakeFez()
print('backend:', backend.name, backend.num_qubits, 'qubits')
t0 = time.time()
comp = H.compile_all(inst, params, backend, seeds=COMPILE_SEEDS, tries=PACK_TRIES)
print(f'compiled in {time.time()-t0:.0f}s')
for m in H.METHODS: assert H.check_isa(comp[m]['circuit'], backend), m
print('all circuits use only native gates on real couplers')'''),
code('''rows = []
for m in H.METHODS:
    st = comp[m]['blocks']
    rows.append(dict(method=m, CZ_total=sum(s['cz'] for s in st), CZ_per_block_median=np.median([s['cz'] for s in st]),
                     depth_max=max(s['depth'] for s in st), ESP_block_median=np.median([s['esp'] for s in st]), ESP_block_min=min(s['esp'] for s in st),
                     qubits_per_circuit=12 * 10))
pd.DataFrame(rows).round(3)'''),
md('''## 4. Pre-registered prediction (global depolarizing model per block)
`output = ESP·ideal + (1−ESP)·uniform` for each block, with ESP the product of (1 − error) over the compiled gates of that block. Measured results from notebook 3 are compared with this table. The model was validated against gate-level noisy simulation up to about 20 qubits (paper, Sec. 9), where it was somewhat pessimistic.'''),
code('''pred = []
for m in H.METHODS:
    esp = [s['esp'] for s in comp[m]['blocks']]
    P = H.model_probs(ideal, esp, m)
    mt = [H.metrics(b, p) for b, p in zip(inst['blocks'], P)]
    pred.append(dict(method=m, feasible_per_block=np.mean([x['feas'] for x in mt]), quality_feasible=np.mean([x['quality'] for x in mt]),
                     expected_feasible_blocks_per_shot=12 * np.mean([x['feas'] for x in mt])))
pred_df = pd.DataFrame(pred); pred_df.round(3)'''),
md('## 5. Save circuits and metadata (no credentials involved)'),
code('''from qiskit import qpy
for m in H.METHODS:
    with open(f'{OUT}/{m}_120q.qpy', 'wb') as f: qpy.dump(comp[m]['circuit'], f)
    with open(f'{OUT}/{m}_blocks.qpy', 'wb') as f: qpy.dump(comp[m]['subs'], f)
meta = dict(backend=backend.name, live=USE_LIVE_BACKEND, seed_obj=SEED_OBJ, seed_iface=SEED_IFACE, iface=inst['iface'],
            variants=H.VARIANTS, methods=H.METHODS,
            compile={m: [{k: v for k, v in s.items()} for s in comp[m]['blocks']] for m in H.METHODS},
            regions={m: comp[m]['regions'] for m in H.METHODS}, prediction=pred_df.to_dict('records'))
json.dump(meta, open(f'{OUT}/meta.json', 'w'), indent=1, default=float)
pickle.dump(dict(inst=inst, params=params, ideal=ideal), open(f'{OUT}/instance.pkl', 'wb'))
print('saved to', OUT, ':', sorted(os.listdir(OUT)))'''),
]

# ------------------------------------------------------------------------------------------------ notebook 2
nb2 = nbf.v4.new_notebook()
nb2.cells = [
md('''# HW100 · 2/3 — Run

Three modes, same output format (`results/hw100/<mode>/<method>.npz`, an int8 array `bits` of shape (shots, 120)):

* `model`  — samples from the depolarizing model (instant, to test notebook 3).
* `aer`    — gate-level noisy simulation (FakeFez noise model), each 11-qubit block simulated separately; takes minutes.
* `ibm`    — **real hardware**. Needs your key file. One job with all seven 120-qubit circuits.

Run notebook 1 first (ideally with `USE_LIVE_BACKEND = True` right before this one).'''),
code(SETUP),
code('''# ---------------- configuration ----------------
MODE = 'model'                    # 'model' | 'aer' | 'ibm'
SHOTS = 4000
APIKEY = 'apikey_personal.json'   # local file {"apikey": "..."}; never committed, never in the zip
BACKEND_NAME = 'ibm_fez'          # any Heron device with >= 130 usable qubits; must be the one used in notebook 1
INSTANCE = None
DD = True                         # dynamical decoupling
TWIRL = True                      # Pauli twirling of the CZ gates (32 randomizations)
RETRIEVE_JOB_ID = None            # set to a job id string to fetch the results of a job submitted earlier
AER_SHOTS, AER_BLOCKS = 500, None # aer mode: shots per block, and list of block ids (None = all 12)'''),
code('''from qiskit import qpy
meta = json.load(open(f'{OUT}/meta.json')); D = pickle.load(open(f'{OUT}/instance.pkl', 'rb')); inst, ideal = D['inst'], D['ideal']
methods = meta['methods']
big = {m: qpy.load(open(f'{OUT}/{m}_120q.qpy', 'rb'))[0] for m in methods}
subs = {m: qpy.load(open(f'{OUT}/{m}_blocks.qpy', 'rb')) for m in methods}
print('circuits loaded:', {m: (c.num_qubits, c.num_clbits) for m, c in big.items()})
dest = f'{RES}/{MODE}'; os.makedirs(dest, exist_ok=True)'''),
code('''def save(m, bits): np.savez_compressed(f'{dest}/{m}.npz', bits=bits.astype(np.int8))

if MODE == 'model':
    for i, m in enumerate(methods):
        esp = [s['esp'] for s in meta['compile'][m]]
        save(m, H.sample_model(ideal, esp, m, SHOTS, seed=i)); print(m, 'sampled')

elif MODE == 'aer':
    from qiskit_ibm_runtime.fake_provider import FakeFez
    bk = FakeFez()
    for i, m in enumerate(methods):
        print(m)
        save(m, H.aer_blocks(inst, subs[m], meta['regions'][m], bk, shots=AER_SHOTS, blocks=AER_BLOCKS, seed=i))

elif MODE == 'ibm':
    from qiskit_ibm_runtime import SamplerV2 as Sampler
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    svc, backend = H.connect(APIKEY, BACKEND_NAME, INSTANCE)
    print('backend', backend.name, '| status:', backend.status().status_msg, '| queue:', backend.status().pending_jobs)
    if RETRIEVE_JOB_ID:
        job = svc.job(RETRIEVE_JOB_ID)
    else:
        pm = generate_preset_pass_manager(optimization_level=0, backend=backend)       # keeps our layout; only checks ISA / adds nothing
        isa = {m: pm.run(big[m]) for m in methods}
        for m in methods:
            assert H.check_isa(isa[m], backend), f'{m}: not ISA on the live backend (recompile in notebook 1 with USE_LIVE_BACKEND=True)'
        sampler = Sampler(mode=backend)
        if DD: sampler.options.dynamical_decoupling.enable = True; sampler.options.dynamical_decoupling.sequence_type = 'XpXm'
        if TWIRL:
            sampler.options.twirling.enable_gates = True; sampler.options.twirling.num_randomizations = 32
            sampler.options.twirling.shots_per_randomization = max(1, SHOTS // 32)
        job = sampler.run([(isa[m], None, SHOTS) for m in methods])
        json.dump(dict(job_id=job.job_id(), backend=backend.name, shots=SHOTS, dd=DD, twirl=TWIRL, time=time.ctime()), open(f'{dest}/job.json', 'w'))
        print('submitted job', job.job_id(), '(saved to', f'{dest}/job.json)')
    res = job.result()
    for m, r in zip(methods, res):
        counts = r.data[list(r.data.keys())[0]].get_counts()
        save(m, H.counts_to_bits(counts, 120)); print(m, sum(counts.values()), 'shots')
    print('saved. Next: notebook 3 with SOURCE = "ibm".')'''),
md('''### Notes for the real run
* One job, six circuits × `SHOTS`. Estimated QPU time is a few seconds per circuit (the deepest block is about 1000–1400 layers, roughly 0.2–0.4 ms per shot including reset), so the whole job is on the order of a minute or two; the queue dominates.
* If `check_isa` fails, the backend's couplers changed since notebook 1: rerun notebook 1 with `USE_LIVE_BACKEND = True`.
* Twirling on a post-selected experiment is fine (it only randomises the coherent part of the error). Switch `TWIRL`/`DD` off to see their effect; running both settings is a worthwhile control.
* If the job is long in the queue, you can stop the notebook and come back with `RETRIEVE_JOB_ID`.'''),
]

# ------------------------------------------------------------------------------------------------ notebook 3
nb3 = nbf.v4.new_notebook()
nb3.cells = [
md('''# HW100 · 3/3 — Analysis

Per-block feasibility with confidence intervals, quality among feasible samples, how many fully feasible 120-variable assignments can be **assembled** by per-block post-selection, and the comparison with the pre-registered noise-model prediction.

**Important when reading the results.** Blocks are independent: a shot is feasible for the whole problem only if all 12 blocks are, which has probability ≈ ∏ p_b for *every* method. What per-block post-selection buys is that complete feasible assignments are assembled from accepted blocks of different shots; the yield is limited by the worst block.'''),
code(SETUP),
code('''SOURCE = 'model'      # 'ibm' | 'aer' | 'model'  (folder under results/hw100/)
import pandas as pd, matplotlib; import matplotlib.pyplot as plt
meta = json.load(open(f'{OUT}/meta.json')); D = pickle.load(open(f'{OUT}/instance.pkl', 'rb')); inst, ideal = D['inst'], D['ideal']
methods = meta['methods']
bits = {m: np.load(f'{RES}/{SOURCE}/{m}.npz')['bits'] for m in methods}
print(SOURCE, {m: b.shape for m, b in bits.items()})
if SOURCE == 'model': print('WARNING: SOURCE = model only tests this notebook (samples come from the same model as the prediction, so criterion 5 passes by construction).')'''),
code('''R = {m: H.analyse(inst, bits[m]) for m in methods}
pred = {p['method']: p for p in meta['prediction']}
rand_mean, rand_best = H.random_feasible_cost(inst); exact = H.exact_block_cost(inst)
rows = []
for m in methods:
    r = R[m]; N = r['shots']; k = sum(x['n_acc'] for x in r['blocks']); lo, hi = H.wilson(k, N * 12)
    rows.append(dict(method=m, feasible_per_block=r['block_feas_mean'], CI95=f'[{lo:.3f}, {hi:.3f}]', model_prediction=pred[m]['feasible_per_block'],
                     feasible_blocks_per_shot=r['mean_feasible_blocks'], quality=np.nanmean([x['quality'] for x in r['blocks']]),
                     assembled_solutions=r['assembled'], assembled_cost_mean=r['assembled_cost_mean'], global_feasible_shots=r['global_feasible_fraction']))
tab = pd.DataFrame(rows); tab.round(4)'''),
code('''print(f"reference costs of the 120-variable objective: random feasible mean {rand_mean:.2f} (best of 2000: {rand_best:.2f}); exact block-coordinate optimum {exact:.2f}")
print("assembled_solutions = number of complete feasible assignments that can be built from accepted blocks = min over blocks of accepted samples")'''),
code('''fig, ax = plt.subplots(1, 2, figsize=(13, 4))
x = np.arange(12); w = 0.11
for i, m in enumerate(methods):
    ax[0].bar(x + (i - 3) * w, [b['feas'] for b in R[m]['blocks']], w, label=m)
ax[0].set_xlabel('block'); ax[0].set_ylabel('measured feasible fraction'); ax[0].set_title(f'feasibility per block ({SOURCE})'); ax[0].legend(fontsize=7, ncol=2)
for m in methods:
    ax[1].scatter([s['esp'] for s in meta['compile'][m]], [b['feas'] for b in R[m]['blocks']], label=m, s=18)
ax[1].set_xlabel('block ESP (from compiled gates)'); ax[1].set_ylabel('measured feasible fraction'); ax[1].set_title('feasibility vs ESP'); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f'{RES}/{SOURCE}_feasibility.png', dpi=140); plt.show()'''),
md('''## Pre-stated success criteria
Defined before the run. Each line prints PASS/FAIL from the data; none is tuned afterwards.

1. **Feasibility above chance.** `ours_p0` has per-block feasible fraction above the 95 % upper bound of `uniform`.
2. **Feasibility against the quantum baselines.** `ours_p0` and `ours` have per-block feasibility above the upper bound of `lcu`, `unbalanced` and `xy`.
3. **Scale.** All 12 blocks (120 qubits) return at least 50 accepted samples for `ours_p0`, i.e. a complete feasible assembly exists.
4. **Quality.** `ours` has quality among feasible samples above the `ours_p0` value (the objective layer helps despite noise). Comparison with `lcu` is reported, not required.
4b. **Cost-informed preparation (reported).** `ours_tilt_p0` and `ours_tilt` use the classical mean-field tilt realised by Dicke angles only; `lcu_b`, `unbalanced_b` and `xy_b` receive the same information as the start of their product state. 2c requires `ours_tilt_p0` to have the highest feasibility; 4b compares quality.
5. **Model.** Measured feasibility of `ours_p0` lies within 0.1 (absolute) of the model prediction.'''),
code('''def feas_ci(m):
    r = R[m]; k = sum(x['n_acc'] for x in r['blocks']); return H.wilson(k, r['shots'] * 12)
q = lambda m: np.nanmean([x['quality'] for x in R[m]['blocks']])
f = lambda m: R[m]['block_feas_mean']
checks = [
 ('1 feasibility above uniform control', f('ours_p0') > feas_ci('uniform')[1]),
 ('2a ours_p0 above all three baselines', f('ours_p0') > max(feas_ci('lcu')[1], feas_ci('unbalanced')[1], feas_ci('xy')[1])),
 ('2b ours (p=1) above all three baselines', f('ours') > max(feas_ci('lcu')[1], feas_ci('unbalanced')[1], feas_ci('xy')[1])),
 ('3 complete assemblies at 120 qubits (>= 50 per block)', R['ours_p0']['assembled'] >= 50),
 ('2c ours_tilt_p0 above the three biased baselines (same classical information)', f('ours_tilt_p0') > max(feas_ci('lcu_b')[1], feas_ci('unbalanced_b')[1], feas_ci('xy_b')[1])),
 ('4 quality: ours > ours_p0', q('ours') > q('ours_p0')),
 ('4b quality: ours_tilt_p0 vs best biased baseline (reported)', q('ours_tilt_p0') > max(q('lcu_b'), q('unbalanced_b'), q('xy_b'))),
 ('5 model within 0.1 of measured (ours_p0)', abs(f('ours_p0') - pred['ours_p0']['feasible_per_block']) < 0.1)]
for name, ok in checks: print('PASS' if ok else 'FAIL', '-', name)
print(f"\\nquality among feasible samples: " + ', '.join(f'{m} {q(m):.3f}' for m in methods))'''),
md('''## How to read the outcome
* If 1–3 pass: feasible-by-construction states are produced at ~100 qubits on the chip, per block, with feasibility above both the unconstrained control and the warm-started baselines, which were given the favourable block decomposition. The model predicts that 2a passes and 2b (the p=1 layer) fails against `unbalanced`.
* Criterion 4 can fail: the objective layer costs 2–4× the gates of the preparation and may not pay off at these depths. That is a result, not a bug; `ours_full` shows the other end of the trade-off.
* `lcu` and `xy` are expected to be competitive in *quality among their feasible samples* (they have fewer gates), and `unbalanced` in how many feasible samples it produces once it has the warm start; the preparation alone is the clear winner on feasibility.
* What this does **not** show: a quantum advantage over classical solvers (blocks of 10 qubits are trivial classically), or optimisation quality inside blocks beyond the criteria above. It shows scale and feasibility.'''),
]

for name, nb in (('HW100_01_build', nb1), ('HW100_02_run', nb2), ('HW100_03_analyze', nb3)):
    nb.metadata['kernelspec'] = dict(display_name='Python 3', language='python', name='python3')
    nbf.write(nb, f'notebooks/{name}.ipynb'); print('wrote', name)

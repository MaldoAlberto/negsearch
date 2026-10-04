"""E6: logical gate counts for large n (exact Clifford+T accounting of the generated circuits),
expected total T-count with amplitude amplification, and a simple surface-code estimate.
ZX reduction ratios are taken from E4 (measured on smaller instances) and stated as such."""
import os as _os, sys as _sys
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..'))
_os.chdir(ROOT); _sys.path.insert(0, ROOT)
import json, math, random, sys, statistics, glob
sys.path.insert(0, ROOT)
from negsearch.core import *
from negsearch.circuits import component_circuits
from negsearch.zxtools import compact

T_PER = {"ccx": 7, "ch": 2}   # Clifford+T decomposition used by Qiskit (checked below)


def t_count(qc):
    ops = qc.count_ops()
    return sum(T_PER.get(k, 0) * v for k, v in ops.items()) + ops.get("t", 0) + ops.get("tdg", 0)


def check_t_per():
    from qiskit import QuantumCircuit, transpile
    for g in ("ccx", "ch"):
        qc = QuantumCircuit(3)
        getattr(qc, g)(*(range(3) if g == "ccx" else range(2)))
        o = transpile(qc, basis_gates=["cx", "h", "t", "tdg", "s", "sdg", "x", "z"], optimization_level=0).count_ops()
        assert o.get("t", 0) + o.get("tdg", 0) == T_PER[g], (g, o)


def surface_code(T_total, Q_L, p_phys=1e-3, p_th=1e-2, budget=0.01, cycle_us=1.0):
    """Smallest odd d with Q_L * T_total * d * p_L(d) <= budget, p_L(d)=0.1 (p/p_th)^((d+1)/2).
    One 15-to-1 factory (~11 tiles); one T state per d cycles."""
    for d in range(3, 101, 2):
        pL = 0.1 * (p_phys / p_th) ** ((d + 1) / 2)
        if Q_L * T_total * d * pL <= budget:
            phys = 2 * d * d * (math.ceil(1.5 * Q_L) + 11)
            return dict(d=d, physical_qubits=phys, runtime_s=T_total * d * cycle_us * 1e-6)
    return None


if __name__ == "__main__":
    check_t_per()
    zx_ratio = {}
    for fn in glob.glob(f"{ROOT}/results/e45_*.json"):
        for r in json.load(open(fn)):
            for comp in ("O", "S0"):
                zx_ratio.setdefault((r["family"], comp), []).append(r[comp]["zx"]["T"] / r[comp]["orig"]["T"])
            a = r["K"]["None"]
            if a["A_orig"]["T"] > 0:
                zx_ratio.setdefault((r["family"], "A"), []).append(a["A_zx"]["T"] / a["A_orig"]["T"])
    zx_ratio = {k: statistics.mean(v) for k, v in zx_ratio.items()}
    e3 = {fam: json.load(open(f"{ROOT}/results/e3_{fam}.json")) for fam in FAMILIES
          if glob.glob(f"{ROOT}/results/e3_{fam}.json")}
    rng = random.Random(9)
    out = []
    for fam, rows in e3.items():
        for n in sorted(set(r["n"] for r in rows)):
            sel = [r for r in rows if r["n"] == n]
            if n < 20 or n % 10:
                continue
            f = sample_satisfiable(fam, n, rng, cap=20000)
            order = list(range(1, n + 1)); rng.shuffle(order)
            _os.makedirs(f"instances/e6/{fam}", exist_ok=True)
            with open(f"instances/e6/{fam}/n{n:03d}.cnf", "w") as fh:
                fh.write(f"c E6 resource-count instance family={fam} n={n} (Random(9) stream)\nc order {' '.join(map(str, order))}\n")
                fh.write(f"p cnf {f.n} {f.m}\n" + "".join(" ".join(map(str, c)) + " 0\n" for c in f.clauses))
            comp = component_circuits(f, order, None)
            TA, TO, TS = t_count(comp["A"]), t_count(comp["O"]), t_count(comp["S0"])
            QL = max(compact(comp[c]).num_qubits for c in comp)
            pG = statistics.median(r["p_grover"] for r in sel)
            pN = statistics.median(r["p_median"] for r in sel)
            itG, itN = math.pi / (4 * math.sqrt(pG)), math.pi / (4 * math.sqrt(pN))
            rz = lambda c: zx_ratio.get((fam, c), statistics.mean(v for (ff, cc), v in zx_ratio.items() if cc == c) if zx_ratio else 1.0)
            rows_out = {
                "Grover": itG * (TO + TS),
                "Grover+ZX": itG * (TO * rz("O") + TS * rz("S0")),
                "Negation-forced": itN * (2 * TA + TO + TS),
                "Negation-forced+ZX": itN * (2 * TA * rz("A") + TO * rz("O") + TS * rz("S0")),
            }
            rec = dict(family=fam, n=n, m=f.m, logical_qubits=QL, T_A=TA, T_O=TO, T_S0=TS, p_grover=pG, p_nf=pN,
                       iters_grover=itG, iters_nf=itN, zx_ratio={c: rz(c) for c in ("A", "O", "S0")},
                       classical_ppz_repetitions=1 / pN,
                       minisat_conflicts_median=statistics.median(r["minisat_conflicts"] for r in sel),
                       T_total=rows_out, surface={k: surface_code(v, QL) for k, v in rows_out.items()})
            out.append(rec)
            print(fam, n, {k: f"{v:.2e}" for k, v in rows_out.items()},
                  {k: (s["d"], f"{s['physical_qubits']:.2e}", f"{s['runtime_s']:.2e}s") for k, s in rec["surface"].items()}, flush=True)
    json.dump(out, open(f"{ROOT}/results/e6_resources.json", "w"), indent=1)

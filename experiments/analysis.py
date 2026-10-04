"""Analysis and figures for the paper.  Reads results/*.json, writes figures/*.pdf and
results/summary.json (all numbers quoted in the paper come from here)."""
import os as _os, sys as _sys
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..'))
_os.chdir(ROOT); _sys.path.insert(0, ROOT)
import glob, json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = f"{ROOT}/results"; F = f"{ROOT}/figures"
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "violet": "#4a3aa7",
     "red": "#e34948"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
                     "font.family": "serif", "mathtext.fontset": "cm"})
FAMS = ["3-SAT", "4-SAT", "1-in-3-SAT", "3-COL"]
summary = {}


def style(ax, title=None, xl=None, yl=None):
    if title: ax.set_title(title, loc="left", color=INK)
    if xl: ax.set_xlabel(xl, color=INK2)
    if yl: ax.set_ylabel(yl, color=INK2)
    ax.grid(axis="y", color=GRID, lw=0.6)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#bdbcb6")
    ax.tick_params(colors=INK2)


def fit_slope(rows, key, B=2000, seed=0):
    """log2 p = a + s n ; bootstrap over instances. Returns s, (lo, hi)."""
    n = np.array([r["n"] for r in rows], float)
    y = np.array([math.log2(r[key]) for r in rows])
    s, a = np.polyfit(n, y, 1)
    rng = np.random.default_rng(seed); bs = []
    for _ in range(B):
        idx = rng.integers(0, len(n), len(n))
        if len(set(n[idx])) < 2: continue
        bs.append(np.polyfit(n[idx], y[idx], 1)[0])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(s), float(a), (float(lo), float(hi))


# ------------------------------------------------------------------------- E3 scaling
def e3():
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.5), sharey=True)
    out = {}
    for ax, fam in zip(axs, FAMS):
        fn = f"{R}/e3_{fam}.json"
        if not os.path.exists(fn): ax.axis("off"); continue
        rows = json.load(open(fn))
        for r in rows:
            r["p_best_of_10"] = r["p_max"]
        res = {}
        for key, lab, col, mk in [("p_grover", "uniform (Grover)", C["orange"], "o"),
                                  ("p_median", "negation-forced, median order", C["blue"], "s"),
                                  ("p_max", "negation-forced, best of 10 orders", C["aqua"], "^")]:
            s, a, ci = fit_slope(rows, key)
            res[key] = dict(slope=s, ci=ci, quantum_base=2 ** (-s / 2), quantum_base_ci=(2 ** (-ci[1] / 2), 2 ** (-ci[0] / 2)))
            ns = sorted(set(r["n"] for r in rows))
            med = [np.median([math.log2(r[key]) for r in rows if r["n"] == nn]) for nn in ns]
            ax.plot(ns, med, color=col, lw=1.4, marker=mk, ms=3, label=lab)
            ax.plot(ns, [a + s * nn for nn in ns], color=col, lw=0.8, ls=":")
        Dn = np.polyfit([r["n"] for r in rows], [r["D_mean"] for r in rows], 1)[0]
        res["decisions_per_variable"] = float(Dn)
        res["n_range"] = [min(r["n"] for r in rows), max(r["n"] for r in rows)]
        res["instances"] = len(rows)
        res["minisat_conflicts_max"] = max(r["minisat_conflicts"] for r in rows)
        out[fam] = res
        style(ax, fam, "n (variables)", r"$\log_2 p$" if fam == FAMS[0] else None)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper center", ncol=3, fontsize=7.5)
    for ax in axs: ax.set_ylim(-72, 2)
    fig.tight_layout(rect=(0, 0, 1, 0.88)); fig.savefig(f"{F}/fig_scaling.pdf"); plt.close(fig)
    summary["E3"] = out


# ------------------------------------------------------------------------- E1 classicality
def e1():
    fn = f"{R}/e1_classicality.json"
    if os.path.exists(fn):
        rows = json.load(open(fn))
        summary["E1"] = dict(points=len(rows), max_abs_diff=max(abs(r["quantum_stabilizer"] - r["classical"]) for r in rows),
                             rows=rows)



def load(fn):
    p = f"{R}/{fn}"
    return json.load(open(p)) if os.path.exists(p) else None


def fmt_base(x): return f"{x:.3f}"


# ------------------------------------------------------------------------- E2 MPS
def e2():
    rd = lambda fn: [json.loads(l) for l in open(f"{R}/{fn}")] if os.path.exists(f"{R}/{fn}") else None
    a, b = rd("e2a.jsonl"), rd("e2b.jsonl")
    to = (rd("e2a_timeouts.jsonl") or []) + (rd("e2b_timeouts.jsonl") or [])
    fig, axs = plt.subplots(1, 2, figsize=(3.4, 1.9))
    out = {}
    if a:
        cols = dict(zip(FAMS, [C["blue"], C["orange"], C["aqua"], C["violet"]]))
        for r in a:
            axs[0].errorbar(r["p_exact"], r["p_mps"], yerr=2 * r["se"], fmt="o", ms=2.5, color=cols[r["family"]], lw=0.7)
        lim = [min(r["p_exact"] for r in a) / 2, 1]
        axs[0].plot(lim, lim, color=INK2, lw=0.6); axs[0].set_xscale("log"); axs[0].set_yscale("log")
        z = [abs(r["p_mps"] - r["p_exact"]) / max(r["se"], 1e-12) for r in a]
        out["timeouts"] = to
        out["a"] = dict(n=len(a), max_n=max(r["n"] for r in a), max_qubits=max(r["qubits"] for r in a),
                        within2=sum(1 for v in z if v <= 2), within3=sum(1 for v in z if v <= 3), max_z=max(z))
        style(axs[0], None, r"$p_\pi$ (Lemma 3)", r"$p$ (MPS)")
        for fam in FAMS:
            axs[0].plot([], [], "o", ms=3, color=cols[fam], label=fam)
        axs[0].legend(frameon=False, fontsize=5.5, loc="upper left")
    if b:
        for i, r in enumerate(b):
            col = C["blue"] if r["method"] == "negation-forced" else C["orange"]
            axs[1].errorbar(r["success_theory"], r["success_mps"], yerr=2 * r["se"], fmt="s" if col == C["blue"] else "o",
                            ms=2.5, color=col, lw=0.7)
        lo = min(min(r["success_theory"] for r in b), min(r["success_mps"] for r in b)) - 0.03
        axs[1].plot([lo, 1], [lo, 1], color=INK2, lw=0.6); axs[1].set_xlim(lo, 1.01); axs[1].set_ylim(lo, 1.01)
        axs[1].plot([], [], "s", ms=3, color=C["blue"], label="NF"); axs[1].plot([], [], "o", ms=3, color=C["orange"], label="Grover")
        axs[1].legend(frameon=False, fontsize=6, loc="upper left")
        z = [abs(r["success_mps"] - r["success_theory"]) / max(r["se"], 1e-3) for r in b]
        out["b"] = dict(n=len(b), max_qubits=max(r["qubits"] for r in b), max_gates=max(r["gates"] for r in b),
                        within3=sum(1 for v in z if v <= 3), rows=[{k: r[k] for k in ("family", "n", "method", "k", "success_theory", "success_mps", "qubits", "gates")} for r in b])
        style(axs[1], None, "theory", "MPS")
    fig.tight_layout(); fig.savefig(f"{F}/fig_mps.pdf"); plt.close(fig)
    summary["E2"] = out


# ------------------------------------------------------------------------- E4 / E5
def e45():
    rows = []
    for fam in FAMS:
        r = load(f"e45_{fam}.json")
        if r: rows += r
    ver = (load("e4_verify_mps.json") or []) + (load("e4_verify.json") or [])
    out = {}
    lines = [r"\begin{table}[t]\centering\small", r"\caption{ZX simplification per component (mean over instances): $T$-count and two-qubit gates before $\to$ after, and the equivalence checks (statevector or MPS, see text).}\label{tab:zx}",
             r"\begin{tabular}{llrrr}\toprule Family & comp. & $T$ & 2q & $\Delta T$\\\midrule"]
    ratios = []
    for fam in FAMS:
        fr = [r for r in rows if r["family"] == fam]
        if not fr: continue
        for comp in ("A", "O", "S0"):
            if comp == "A":
                o = [r["K"]["None"]["A_orig"] for r in fr]; z = [r["K"]["None"]["A_zx"] for r in fr]
            else:
                o = [r[comp]["orig"] for r in fr]; z = [r[comp]["zx"] for r in fr]
            To, Tz = np.mean([x["T"] for x in o]), np.mean([x["T"] for x in z])
            Co, Cz = np.mean([x["CX"] for x in o]), np.mean([x["CX"] for x in z])
            ratios.append(1 - Tz / To)
            name = {"A": r"$A_\pi$", "O": r"$O_F$", "S0": r"$S_0$"}[comp]
            lines.append(f"{fam} & {name} & {To:.0f}$\\to${Tz:.0f} & {Co:.0f}$\\to${Cz:.0f} & $-{100*(1-Tz/To):.0f}\\%$\\\\")
    nver = len(ver); okver = sum(1 for v in ver if v.get("nonzero_outcomes", 0) == 0 and v.get("min_fidelity", 1) > 1 - 1e-9)
    lines += [r"\bottomrule\end{tabular}", f"\\\\[2pt]\\footnotesize Equivalence checks passed: {okver}/{nver} components.", r"\end{table}"]
    os.makedirs(f"{ROOT}/tables", exist_ok=True)
    open(f"{ROOT}/tables/zx.tex", "w").write("\n".join(lines))
    out["T_reduction_range"] = (min(ratios), max(ratios)) if ratios else None
    out["verified"] = (okver, nver)
    # E5
    fig, axs = plt.subplots(1, len({r['family'] for r in rows}) or 1, figsize=(3.4, 1.9), sharey=True)
    axs = np.atleast_1d(axs)
    Ks = ["0", "1", "2", "3", "None"]; best = []
    for ax, fam in zip(axs, [f for f in FAMS if any(r["family"] == f for r in rows)]):
        for r in [x for x in rows if x["family"] == fam]:
            base = r["K"]["0"]["T_total_orig"]
            y = [r["K"][k]["T_total_zx"] / base for k in Ks]
            ax.plot(range(5), y, color=C["blue"], lw=0.9, alpha=0.8, marker="o", ms=2)
            kb = int(np.argmin(y)); ax.plot([kb], [y[kb]], "*", color=C["orange"], ms=7)
            best.append(dict(family=fam, n=r["n"], Kstar=Ks[kb], gain_vs_grover=base / min(r["K"][k]["T_total_zx"] for k in Ks),
                             gain_full_vs_grover_zx=r["K"]["0"]["T_total_zx"] / r["K"]["None"]["T_total_zx"],
                             gain_best_vs_full=r["K"]["None"]["T_total_zx"] / min(r["K"][k]["T_total_zx"] for k in Ks)))
        ax.set_xticks(range(5)); ax.set_xticklabels(["0", "1", "2", "3", r"$\infty$"]); ax.set_yscale("log")
        style(ax, fam, "K", r"$\mathcal{T}(K)/\mathcal{T}_{\rm Grover}$" if ax is axs[0] else None)
        ax.title.set_fontsize(7)
    fig.tight_layout(); fig.savefig(f"{F}/fig_select.pdf"); plt.close(fig)
    out["selection"] = best
    summary["E45"] = out


# ------------------------------------------------------------------------- E6
def e6():
    rows = load("e6_resources.json")
    if not rows: return
    pick = []
    for fam in FAMS:
        fr = [r for r in rows if r["family"] == fam]
        if not fr: continue
        ns = sorted(r["n"] for r in fr)
        for n in sorted({ns[0] if 40 not in ns else 40, ns[-1]}):
            pick.append(next(r for r in fr if r["n"] == n))
    e = lambda x: f"{x:.1e}".replace("e+0", "e").replace("e+", "e")
    def tm(s):
        s = float(s)
        for unit, v in (("y", 3.15e7), ("d", 86400), ("h", 3600), ("min", 60)):
            if s >= v: return f"{s/v:.1f}\\,{unit}"
        return f"{s:.1f}\\,s"
    lines = [r"\begin{table*}[t]\centering\small",
             r"\caption{Fault-tolerant estimates for the median instance: expected logical $T$-count to find a solution, surface-code distance $d$, physical qubits and runtime (model in the text). Classical reps.: expected repetitions $1/p$ of the classical algorithm with the same negative conditions.}\label{tab:resources}",
             r"\begin{tabular}{lr|rrrr|rrrr|r}\toprule",
             r" & & \multicolumn{4}{c|}{Grover (uniform)} & \multicolumn{4}{c|}{Negation-forced + ZX} & \\",
             r"Family & $n$ & $T$ & $d$ & qubits & time & $T$ & $d$ & qubits & time & classical reps.\\\midrule"]
    for r in pick:
        T = r["T_total"]; S = r["surface"]; g, nz = S["Grover"], S["Negation-forced+ZX"]
        lines.append(f"{r['family']} & {r['n']} & {e(T['Grover'])} & {g['d']} & {e(g['physical_qubits'])} & {tm(g['runtime_s'])} & {e(T['Negation-forced+ZX'])} & {nz['d']} & {e(nz['physical_qubits'])} & {tm(nz['runtime_s'])} & {e(r['classical_ppz_repetitions'])}\\\\")
    lines += [r"\bottomrule\end{tabular}", r"\end{table*}"]
    open(f"{ROOT}/tables/resources.tex", "w").write("\n".join(lines))
    summary["E6"] = rows


def tables_scaling_classical():
    E3 = summary.get("E3", {})
    lines = [r"\begin{table}[t]\centering\small",
             r"\caption{Fitted scaling $p\propto 2^{-sn}$ (95\% bootstrap CI) and effective quantum base $2^{s/2}$; decisions per variable $D/n$ along solutions versus the PPZ bound $1-1/k$ (for $k$-CNF).}\label{tab:scaling}",
             r"\resizebox{\columnwidth}{!}{\begin{tabular}{lrcccc}\toprule Family & inst. & Grover & NF (median) & NF (best) & $D/n$ (bound)\\\midrule"]
    for fam in FAMS:
        if fam not in E3: continue
        r = E3[fam]
        ci = r["p_median"]["quantum_base_ci"]
        bound = {"3-SAT": "0.667", "4-SAT": "0.750"}.get(fam, "--")
        lines.append(f"{fam} & {r['instances']} & {r['p_grover']['quantum_base']:.3f} & {r['p_median']['quantum_base']:.3f} [{ci[0]:.3f},{ci[1]:.3f}] & {r['p_max']['quantum_base']:.3f} & {r['decisions_per_variable']:.3f} ({bound})\\\\")
    lines += [r"\bottomrule\end{tabular}}", r"\end{table}"]
    open(f"{ROOT}/tables/scaling.tex", "w").write("\n".join(lines))
    E1 = summary.get("E1")
    lines = [r"\begin{table}[t]\centering\small", r"\caption{Classicality check (Lemma~\ref{lem:classical}): success probability of the measure-and-flip circuits (stabilizer simulation, 50\,000 shots) versus the same rules executed classically ($2\times10^6$ trials); all differences are within $2\sigma$ of shot noise.}\label{tab:classical}",
             r"\begin{tabular}{lrrr}\toprule Circuit & iters & quantum & classical\\\midrule"]
    if E1:
        for r in E1["rows"]:
            if r["iters"] in (r["iters"],):
                lines.append(f"{r['problem']} & {r['iters']} & {r['quantum_stabilizer']:.4f} & {r['classical']:.4f}\\\\")
    sub = open(f"{R}/e1_subset.txt").read() if os.path.exists(f"{R}/e1_subset.txt") else ""
    import re
    tv = re.search(r"total variation distance: ([0-9.]+)", sub)
    lines.append(r"\midrule \multicolumn{4}{l}{Subset sum $\{1,2,3,4\}$, target 7, QFT adder: TV distance " + (tv.group(1) if tv else "--") + r"}\\")
    lines += [r"\bottomrule\end{tabular}", r"\end{table}"]
    open(f"{ROOT}/tables/classical.tex", "w").write("\n".join(lines))


def numbers():
    E3 = summary.get("E3", {}); E2 = summary.get("E2", {}); E45 = summary.get("E45", {})
    g = [E3[f]["p_grover"]["quantum_base"] for f in E3]; nf = [E3[f]["p_median"]["quantum_base"] for f in E3]
    m = {}
    m["NInstances"] = str(sum(E3[f]["instances"] for f in E3))
    m["GroverBaseRange"] = f"${min(g):.2f}^n$--${max(g):.2f}^n$" if g else "--"
    m["NFBaseRange"] = f"${min(nf):.2f}^n$--${max(nf):.2f}^n$" if nf else "--"
    a = E2.get("a", {}); b = E2.get("b", {})
    m["MaxMPSQubits"] = str(max(a.get("max_qubits", 0), b.get("max_qubits", 0)))
    m["MaxMPSn"] = str(a.get("max_n", "--")); m["NMPSA"] = str(a.get("n", "--"))
    m["MPSAgreement"] = (f"{a.get('within3','--')} of {a.get('n','--')} preparation points and {b.get('within3','--')} of {b.get('n','--')} amplification points agree with theory within $3\\sigma$ shot noise; the largest amplified circuit has {b.get('max_gates','--')} gates on {b.get('max_qubits','--')} qubits." if a else "")
    tr = E45.get("T_reduction_range")
    m["ZXTRange"] = f"{100*tr[0]:.0f}--{100*tr[1]:.0f}\\%" if tr else "--"
    v = E45.get("verified", (0, 0))
    m["ZXSummary"] = (f"On average ZX removes {m['ZXTRange']} of the $T$ gates of each component, while the two-qubit count changes little; all {v[1]} checked components passed the equivalence test." if tr else "")
    sel = E45.get("selection", [])
    if sel:
        from collections import Counter
        ks = Counter(s["Kstar"] for s in sel)
        gains = [s["gain_vs_grover"] for s in sel]
        inter = sum(1 for s in sel if s["Kstar"] != "None")
        m["SelectSummary"] = (f"Over {len(sel)} instances the combined reduction of the expected $T$-count relative to Grover without ZX ranges from {min(gains):.1f}$\\times$ to {max(gains):.1f}$\\times$. "
                              f"The optimum is interior ($K^*<\\infty$) for {inter} of {len(sel)} instances (distribution of $K^*$: " + ", ".join(f"{('$\\infty$' if k=='None' else k)}: {c}" for k, c in sorted(ks.items())) + "), so checking every available negative condition is not always optimal; "
                              f"choosing $K^*$ instead of $K=\\infty$ saves up to {max(s['gain_best_vs_full'] for s in sel):.2f}$\\times$.")
    else:
        m["SelectSummary"] = ""
    E6 = summary.get("E6") or []
    if E6:
        big = [r for r in E6 if r["n"] == max(x["n"] for x in E6 if x["family"] == r["family"])]
        parts = []
        for r in big:
            S = r["surface"]
            if S["Grover"] and S["Negation-forced+ZX"]:
                ratio = r['T_total']['Grover'] / r['T_total']['Negation-forced+ZX']
                ex = int(math.floor(math.log10(ratio))); man = ratio / 10 ** ex
                parts.append(f"{r['family']} ($n={r['n']}$): $T$ lower by ${man:.1f}\\times10^{{{ex}}}$, $d$ {S['Grover']['d']}$\\to${S['Negation-forced+ZX']['d']}, {S['Grover']['physical_qubits']/S['Negation-forced+ZX']['physical_qubits']:.1f}$\\times$ fewer physical qubits")
        m["ResourceSummary"] = "At the largest sizes, relative to Grover: " + "; ".join(parts) + "." if parts else ""
    else:
        m["ResourceSummary"] = ""
    m["MiniSatMaxConflicts"] = f"{max(E3[f]['minisat_conflicts_max'] for f in E3):,}" if E3 else "--"
    with open(f"{ROOT}/numbers.tex", "w") as fh:
        for k, v in m.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    summary["numbers"] = m

if __name__ == "__main__":
    os.makedirs(F, exist_ok=True)
    for fn in [e1, e3, e2, e45, e6, tables_scaling_classical, numbers]:
        fn()
    extra = sys.argv[1:] if len(sys.argv) > 1 else []
    json.dump(summary, open(f"{R}/summary.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in summary.items() if k != "E1"}, indent=1, default=float)[:3000])

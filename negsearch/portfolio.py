"""QOBLIB 06-portfolio (multi-period portfolio optimisation) for small quantum experiments.

Reference model: QOBLIB 06-portfolio/models/binary_quadratic_programming/bqp_u3_c10.zpl, objective recomputed
exactly as the official checker (06-portfolio/check, rounding = Zimpl round-half-away-from-zero).
Here every (asset, direction) has ub = 1 unit copy, so one binary variable x[t, d, i] per period t,
direction d (0 = long, 1 = short) and asset i.  Per period the two linear constraints of the model are

    capital  : 0 <= C - net   <= 2^cs1 - 1      net   = #long - #short
    budget   : 0 <= B - total <= 2^cs2 - 1      total = #long + #short

The QUBO of the reference model enforces them with binary slack registers (cs1 + cs2 extra qubits per period)
and a quadratic penalty.  Here they are instead read as negative conditions: a variable is forced whenever the
other value would make the period's constraints impossible to complete (exact look-ahead), which gives an
in-place preparation A_q whose support is exactly the feasible set (no dead ends)."""
import gzip, os, itertools
from fractions import Fraction
import numpy as np


def zround(x: Fraction) -> int:
    s = 1 if x >= 0 else -1
    a = abs(x) + Fraction(1, 2)
    return s * (a.numerator // a.denominator)


def _lines(path):
    op = gzip.open if path.endswith('.gz') else open
    with op(path, 'rt') as fh:
        for l in fh:
            l = l.strip()
            if l and not l.startswith('#'):
                yield l.split()


def _find(d, base):
    for ext in ('.txt.gz', '.txt'):
        if os.path.exists(os.path.join(d, base + ext)):
            return os.path.join(d, base + ext)
    raise FileNotFoundError(base)


class Instance:
    """Prices / covariances of a QOBLIB portfolio instance directory (optionally restricted to the first
    `periods` days and to the first `assets` symbols; restrictions are labelled as derived sub-instances)."""

    def __init__(self, d, periods=None, assets=None):
        self.dir = d
        syms, raw = [], {}
        for day, s, v in _lines(_find(d, 'stock_prices')):
            if s not in syms: syms.append(s)
            raw[s, int(day)] = Fraction(v)
        T = max(t for _, t in raw) + 1
        self.T = T if periods is None else periods
        self.symbols = syms if assets is None else syms[:assets]
        self.raw = {(s, t): raw[s, t] for s in self.symbols for t in range(self.T)}
        self.cov = {}
        S = set(self.symbols)
        for day, a, b, v in _lines(_find(d, 'covariance_matrices')):
            if a in S and b in S and int(day) < self.T:
                self.cov[a, b, int(day)] = Fraction(v)

    def write(self, out):
        """Write the (restricted) instance in QOBLIB format, so the official checker can be run on it."""
        os.makedirs(out, exist_ok=True)
        hdr = ('# Derived from QOBLIB 06-portfolio/instances/%s (CC BY 4.0): first %d periods, first %d assets.\n'
               % (os.path.basename(self.dir.rstrip('/')), self.T, len(self.symbols)))
        with gzip.open(os.path.join(out, 'stock_prices.txt.gz'), 'wt') as fh:
            fh.write(hdr)
            for t in range(self.T):
                for s in self.symbols:
                    fh.write(f'{t} {s} {_dec(self.raw[s, t])}\n')
        with gzip.open(os.path.join(out, 'covariance_matrices.txt.gz'), 'wt') as fh:
            fh.write(hdr)
            for (a, b, t), v in sorted(self.cov.items(), key=lambda kv: (kv[0][2], kv[0][0], kv[0][1])):
                fh.write(f'{t} {a} {b} {_dec(v)}\n')


def _dec(fr: Fraction):
    """Exact decimal string of a Fraction that came from a decimal literal."""
    from decimal import Decimal, localcontext
    with localcontext() as ctx:
        ctx.prec = 100
        v = Decimal(fr.numerator) / Decimal(fr.denominator)
    assert Fraction(v) == fr
    return format(v, 'f')


class Portfolio:
    """Binary model with ub = 1.  Qubit index q(t, d, i) = t*2n + d*n + i."""

    def __init__(self, inst: Instance, B, lam, cash=300000, unit=100000, delta=Fraction('0.001'),
                 nu=Fraction('0.0001'), rho=Fraction('0.000025'), cs1=2, cs2=2, order='interleaved'):
        self.inst, self.B, self.lam, self.order_kind = inst, B, Fraction(str(lam)), order
        self.unit, self.delta, self.nu, self.rho = Fraction(unit), Fraction(delta), Fraction(nu), Fraction(rho)
        self.C, self.cs1, self.cs2 = cash // unit, cs1, cs2
        self.na, self.T = len(inst.symbols), inst.T
        self.nq = 2 * self.na * self.T
        S = inst.symbols
        self.p = {(i, t): inst.raw[s, t] * self.unit / inst.raw[s, 0] for i, s in enumerate(S) for t in range(self.T)}
        self._build_poly()

    def q(self, t, d, i):
        return t * 2 * self.na + d * self.na + i

    def var(self, k):
        t, r = divmod(k, 2 * self.na); d, i = divmod(r, self.na); return t, d, i

    # ---------------- objective as an exact integer polynomial in x (linear + quadratic) ----------------
    def _build_poly(self):
        n, T, S, p = self.na, self.T, self.inst.symbols, self.p
        h = np.zeros(self.nq, dtype=np.int64); J = {}
        const = 0
        tau = (1, -1)
        tend = T - 1
        def addJ(a, b, c):
            if a == b: h[a] += c; return
            key = (min(a, b), max(a, b)); J[key] = J.get(key, 0) + c
        for t in range(T):
            # risk (ordered pairs, x^2 = x)
            if self.lam:
                for (i, di), (j, dj) in itertools.product(itertools.product(range(n), (0, 1)), repeat=2):
                    c = zround(self.lam * tau[di] * tau[dj] * self.inst.cov[S[i], S[j], t] * p[i, t] * p[j, t])
                    addJ(self.q(t, di, i), self.q(t, dj, j), c)
            # cash interest: - sum_k round(nu*unit*2^k) * bit_k(C - net); here round(nu*unit*2^k) = 10*2^k exactly,
            # so it equals -10*(C - net) and is linear in x (asserted)
            cc = [zround(self.nu * self.unit * 2 ** k) for k in range(self.cs1)]
            assert all(cc[k] == cc[0] * 2 ** k for k in range(self.cs1)), 'cash interest not linear'
            const -= cc[0] * self.C
            for i in range(n):
                h[self.q(t, 0, i)] += cc[0]; h[self.q(t, 1, i)] -= cc[0]
            # short-selling cost
            for i in range(n):
                h[self.q(t, 1, i)] += zround(self.rho * p[i, t])
            # return for t < t_end
            if t < tend:
                for i in range(n):
                    for d in (0, 1):
                        h[self.q(t, d, i)] -= zround(tau[d] * (p[i, t + 1] - p[i, t]))
            # transaction cost
            for i in range(n):
                c = zround(self.delta * p[i, t])
                for d in (0, 1):
                    if t == 0 or t == tend:                 # initial buy-in / liquidation
                        h[self.q(t, d, i)] += c
                    else:                                   # |x_t - x_{t-1}| = x_{t-1} + x_t - 2 x_{t-1} x_t
                        h[self.q(t - 1, d, i)] += c; h[self.q(t, d, i)] += c
                        addJ(self.q(t - 1, d, i), self.q(t, d, i), -2 * c)
        self.h, self.J, self.const = h, J, const

    def energy(self, bits):
        """Objective for an array of bitstrings (shape (..., nq), 0/1)."""
        b = bits.astype(np.int64)
        e = self.const + b @ self.h
        for (a, c), v in self.J.items():
            e = e + v * b[..., a] * b[..., c]
        return e

    def counts(self, bits):
        b = bits.reshape(bits.shape[:-1] + (self.T, 2, self.na))
        L, S = b[..., 0, :].sum(-1), b[..., 1, :].sum(-1)
        return L, S

    def feasible(self, bits):
        L, S = self.counts(bits)
        s1, s2 = self.C - (L - S), self.B - (L + S)
        ok = (s1 >= 0) & (s1 <= 2 ** self.cs1 - 1) & (s2 >= 0) & (s2 <= 2 ** self.cs2 - 1)
        return ok.all(-1)

    def ok_counts(self, L, S):
        s1, s2 = self.C - (L - S), self.B - (L + S)
        return 0 <= s1 <= 2 ** self.cs1 - 1 and 0 <= s2 <= 2 ** self.cs2 - 1

    def solution_lines(self, x):
        out = []
        for t in range(self.T):
            for i, s in enumerate(self.inst.symbols):
                lo, sh = int(x[self.q(t, 0, i)]), int(x[self.q(t, 1, i)])
                if lo or sh: out.append(f'{t} {s} {lo} {sh}')
        return out

    # ---------------- slack QUBO of the reference model (baseline) ----------------
    def slack_qubo(self, P):
        """Return (h, J, const, n_total) of objective + P * sum_t [(C - net - sum 2^c y)^2 + (B - total - sum 2^b s)^2]
        on nq + T*(cs1+cs2) qubits (slack bits appended per period after all x)."""
        n, T = self.na, self.T
        ntot = self.nq + T * (self.cs1 + self.cs2)
        h = np.zeros(ntot, dtype=float); h[:self.nq] = self.h; J = {k: float(v) for k, v in self.J.items()}
        const = float(self.const)
        def sq(terms, rhs):     # P (rhs - sum a_k z_k)^2
            nonlocal const
            const += P * rhs * rhs
            for k, a in terms: h[k] += P * (a * a - 2 * rhs * a)
            for (k1, a1), (k2, a2) in itertools.combinations(terms, 2):
                key = (min(k1, k2), max(k1, k2)); J[key] = J.get(key, 0.0) + 2 * P * a1 * a2
        base = self.nq
        for t in range(T):
            y = [base + c for c in range(self.cs1)]; s = [base + self.cs1 + b for b in range(self.cs2)]
            base += self.cs1 + self.cs2
            sq([(self.q(t, 0, i), 1) for i in range(n)] + [(self.q(t, 1, i), -1) for i in range(n)]
               + [(y[c], 2 ** c) for c in range(self.cs1)], self.C)
            sq([(self.q(t, d, i), 1) for i in range(n) for d in (0, 1)] + [(s[b], 2 ** b) for b in range(self.cs2)], self.B)
        return h, J, const, ntot

    # ---------------- unbalanced penalisation (Montanez-Barrera et al.), no slack ----------------
    def unbalanced_poly(self, l1, l2):
        """objective + sum_t sum_{g in {C-net, B-total}} (-l1*g + l2*g^2): penalises g<0 strongly, slightly
        rewards g>0 small; quadratic in x, no slack qubits."""
        n = self.na
        h = self.h.astype(float).copy(); J = {k: float(v) for k, v in self.J.items()}; const = float(self.const)
        def pen(terms, rhs):    # g = rhs - sum a_k x_k
            nonlocal const
            const += -l1 * rhs + l2 * rhs * rhs
            for k, a in terms: h[k] += l1 * a + l2 * (a * a - 2 * rhs * a)
            for (k1, a1), (k2, a2) in itertools.combinations(terms, 2):
                key = (min(k1, k2), max(k1, k2)); J[key] = J.get(key, 0.0) + 2 * l2 * a1 * a2
        for t in range(self.T):
            pen([(self.q(t, 0, i), 1) for i in range(n)] + [(self.q(t, 1, i), -1) for i in range(n)], self.C)
            pen([(self.q(t, d, i), 1) for i in range(n) for d in (0, 1)], self.B)
        return h, J, const

    # ---------------- negation-forced preparation ----------------
    def period_order(self):
        """Variable order inside a period as (d, i): 'interleaved' = (long_0, short_0, long_1, ...);
        'shorts_first' = (short_0..short_{n-1}, long_0..long_{n-1}) (fewer forced states, cheaper circuit)."""
        if self.order_kind == 'shorts_first':
            return [(1, i) for i in range(self.na)] + [(0, i) for i in range(self.na)]
        return [(d, i) for i in range(self.na) for d in (0, 1)]

    def plan(self):
        """For one period: decision table act[pos][(L,S)] in {'coin', 0, 1} with exact look-ahead
        (value v is allowed iff some completion of the period satisfies both constraints)."""
        order = self.period_order(); m = len(order)
        nl_rem = [sum(1 for d, _ in order[k:] if d == 0) for k in range(m + 1)]
        ns_rem = [sum(1 for d, _ in order[k:] if d == 1) for k in range(m + 1)]
        def extendable(k, L, S):
            return any(self.ok_counts(L + a, S + b) for a in range(nl_rem[k] + 1) for b in range(ns_rem[k] + 1))
        act = []
        reach = {(0, 0)}
        assert extendable(0, 0, 0)
        for k, (d, i) in enumerate(order):
            a = {}; nxt = set()
            for (L, S) in reach:
                o0 = extendable(k + 1, L, S)
                o1 = extendable(k + 1, L + (d == 0), S + (d == 1))
                assert o0 or o1, 'dead end'
                a[L, S] = 'coin' if (o0 and o1) else (1 if o1 else 0)
                if o0: nxt.add((L, S))
                if o1: nxt.add((L + (d == 0), S + (d == 1)))
            act.append(a); reach = nxt
        return order, act

    def _coin_stats(self):
        """For every basis state: valid (follows all forced decisions), #coins that came up 1, #coins that came up 0."""
        if hasattr(self, '_cs'): return self._cs
        order, act = self.plan()
        N = 2 ** self.nq
        idx = np.arange(N, dtype=np.int64)
        valid = np.ones(N, bool); n1 = np.zeros(N, np.int16); n0 = np.zeros(N, np.int16)
        for t in range(self.T):
            L = np.zeros(N, np.int16); S = np.zeros(N, np.int16)
            for k, (d, i) in enumerate(order):
                x = ((idx >> self.q(t, d, i)) & 1).astype(np.int16)
                for (l, s), dec in act[k].items():
                    sel = (L == l) & (S == s)
                    if dec == 'coin':
                        n1 += sel & (x == 1); n0 += sel & (x == 0)
                    else:
                        valid &= ~sel | (x == dec)
                L += (d == 0) * x; S += (d == 1) * x
        self._cs = (valid, n1, n0)
        return self._cs

    def prep_state(self, q):
        """A_q|0> as a dense vector (bit k of the basis index = qubit k).  q: coin probability of '1'."""
        valid, n1, n0 = self._coin_stats()
        amp = np.where(valid, np.sqrt(q) ** n1 * np.sqrt(1 - q) ** n0, 0.0)
        return amp.astype(complex)

    def feasible_count_per_period(self):
        order, _ = self.plan(); m = len(order)
        return sum(1 for bits in itertools.product((0, 1), repeat=m)
                   if self.ok_counts(sum(b for b, (d, _) in zip(bits, order) if d == 0),
                                     sum(b for b, (d, _) in zip(bits, order) if d == 1)))

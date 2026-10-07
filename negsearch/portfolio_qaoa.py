"""Exact statevector simulation (numpy) of QAOA variants on the ub=1 QOBLIB portfolio model."""
import numpy as np
from .portfolio_circuits import xy_edges


class Sim:
    def __init__(self, P, extra=0):
        """extra: number of slack qubits appended (slack-QUBO baseline)."""
        self.P = P; self.n = P.nq + extra
        idx = np.arange(2 ** self.n, dtype=np.int64)
        self.bits = ((idx[:, None] >> np.arange(self.n)) & 1).astype(np.int8)
        xb = self.bits[:, :P.nq]
        self.f = P.energy(xb).astype(float)
        self.feas = P.feasible(xb)
        ff = self.f[self.feas]
        self.fopt, self.fworst = ff.min(), ff.max()
        self.opt = self.feas & (self.f == self.fopt)

    def poly(self, h, J, const=0.0):
        e = float(const) + self.bits @ np.asarray(h, float)
        for (a, b), v in J.items():
            e = e + float(v) * (self.bits[:, a] & self.bits[:, b])
        return e

    def x_mixer(self, psi, beta):
        c, s = np.cos(beta), -1j * np.sin(beta)
        for q in range(self.n):
            v = psi.reshape(-1, 2, 2 ** q)
            a0, a1 = v[:, 0, :].copy(), v[:, 1, :].copy()
            v[:, 0, :] = c * a0 + s * a1; v[:, 1, :] = s * a0 + c * a1
        return psi

    def xy_mixer(self, psi, beta):
        c, s = np.cos(beta), -1j * np.sin(beta)
        for L in xy_edges(self.P):
            for a, b in L:
                ma, mb = 1 << a, 1 << b
                idx = np.nonzero(((np.arange(2 ** self.n) & ma) > 0) & ((np.arange(2 ** self.n) & mb) == 0))[0] \
                    if not hasattr(self, '_xy') or (a, b) not in self._xy else self._xy[a, b]
                if not hasattr(self, '_xy'): self._xy = {}
                self._xy[a, b] = idx
                j = idx - ma + mb
                u, w = psi[idx].copy(), psi[j].copy()
                psi[idx] = c * u + s * w; psi[j] = s * u + c * w
        return psi

    def state(self, kind, gam, bet, cost, psi0=None):
        psi = psi0.copy() if psi0 is not None else np.ones(2 ** self.n, complex) / np.sqrt(2 ** self.n)
        for g, b in zip(gam, bet):
            psi = psi * np.exp(-1j * g * cost)
            if kind == 'x': psi = self.x_mixer(psi, b)
            elif kind == 'xy': psi = self.xy_mixer(psi, b)
            elif kind == 'grover': psi = psi - (1 - np.exp(-1j * b)) * psi0 * np.vdot(psi0, psi)
        return psi

    def metrics(self, psi):
        pr = np.abs(psi) ** 2
        qual = (self.fworst - self.f) / (self.fworst - self.fopt)
        return dict(p_feas=float(pr[self.feas].sum()), p_opt=float(pr[self.opt].sum()),
                    quality=float((pr * self.feas * qual).sum()),
                    mean_obj_feasible=float((pr * self.feas * self.f).sum() / max(pr[self.feas].sum(), 1e-300)))

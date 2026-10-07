# Constraint-guided QAOA for maximum independent set: exact numpy simulation.
"""Constraint-guided QAOA for MIS (exact statevector, numpy).
Methods: penalty QAOA (QUBO, X mixer, |+>), Hadfield constraint-preserving QAOA (|0>, neighbour-controlled RX),
negation-forced preparation A|0> alone, and NF-GM-QAOA (|psi0>=A|0>, Grover mixer about psi0)."""
import numpy as np, itertools
from scipy.optimize import minimize

def read_gph(path):
    n=None; E=[]
    for l in open(path):
        if l.startswith('p'): n=int(l.split()[2])
        elif l.startswith('e'):
            a,b=map(int,l.split()[1:3])
            if a!=b: E.append((min(a,b)-1,max(a,b)-1))
    return n, sorted(set(E))

class MIS:
    def __init__(s,n,E):
        s.n,s.E=n,E; s.N=[set() for _ in range(n)]
        for a,b in E: s.N[a].add(b); s.N[b].add(a)
        idx=np.arange(2**n); s.bits=((idx[:,None]>>np.arange(n))&1).astype(np.int8)   # bit i = qubit i
        s.size=s.bits.sum(1)
        viol=np.zeros(2**n,int)
        for a,b in E: viol+= s.bits[:,a]&s.bits[:,b]
        s.viol=viol; s.feas=(viol==0); s.opt=s.size[s.feas].max()
    # ---- operators on state psi (length 2^n) ----
    def _apply_1q(s,psi,q,U,mask=None):
        psi=psi.reshape([2]*s.n[::-1] if False else [2]*s.n,order='F')  # axis q <-> bit q
        a0=np.take(psi,0,axis=q); a1=np.take(psi,1,axis=q)
        b0=U[0,0]*a0+U[0,1]*a1; b1=U[1,0]*a0+U[1,1]*a1
        if mask is not None:   # mask over remaining axes: apply only where mask True
            m=mask; b0=np.where(m,b0,a0); b1=np.where(m,b1,a1)
        out=np.stack([b0,b1],axis=q)
        return out.reshape(-1,order='F')
    def _zero_mask(s,q,ctrls):
        # boolean array over all axes except q (same shape as np.take(psi,0,axis=q)) : all ctrls == 0
        shape=[2]*s.n; shape.pop(q)
        m=np.ones(shape,bool)
        for c in ctrls:
            ax=c if c<q else c-1
            sl=[slice(None)]*(s.n-1); sl[ax]=1; m[tuple(sl)]=False
        return m
    def prep_A(s,order,psi=None,inverse=False):
        H=np.array([[1,1],[1,-1]])/np.sqrt(2)
        if psi is None: psi=np.zeros(2**s.n,complex); psi[0]=1
        seq=list(order)[::-1] if inverse else list(order)
        pos={v:i for i,v in enumerate(order)}
        for v in seq:
            earlier=[u for u in s.N[v] if pos[u]<pos[v]]
            psi=s._apply_1q(psi,v,H,s._zero_mask(v,earlier) if earlier else None)
        return psi
    def rx(s,beta): c,si=np.cos(beta),np.sin(beta); return np.array([[c,-1j*si],[-1j*si,c]])
    def x_mixer(s,psi,beta):
        for q in range(s.n): psi=s._apply_1q(psi,q,s.rx(beta))
        return psi
    def hadfield_mixer(s,psi,beta):
        for v in range(s.n): psi=s._apply_1q(psi,v,s.rx(beta),s._zero_mask(v,s.N[v]))
        return psi
    # ---- ansatze ----
    def penalty_state(s,params,lam):
        p=len(params)//2; C=-s.size+lam*s.viol
        psi=np.ones(2**s.n,complex)/np.sqrt(2**s.n)
        for l in range(p):
            psi=psi*np.exp(-1j*params[l]*C); psi=s.x_mixer(psi,params[p+l])
        return psi
    def hadfield_state(s,params):
        p=len(params)//2; psi=np.zeros(2**s.n,complex); psi[0]=1; C=-s.size
        for l in range(p):
            psi=psi*np.exp(-1j*params[l]*C); psi=s.hadfield_mixer(psi,params[p+l])
        return psi
    def gm_state(s,params,psi0):
        p=len(params)//2; psi=psi0.copy(); C=-s.size
        for l in range(p):
            psi=psi*np.exp(-1j*params[l]*C)
            psi=psi-(1-np.exp(-1j*params[p+l]))*psi0*np.vdot(psi0,psi)
        return psi
    def metrics(s,psi):
        pr=np.abs(psi)**2
        return dict(p_feas=float(pr[s.feas].sum()), p_opt=float(pr[s.feas&(s.size==s.opt)].sum()),
                    ratio=float((pr*s.size*s.feas).sum()/s.opt))

def optimize(f, dim, restarts=8, seed=0, maxiter=400):
    rng=np.random.default_rng(seed); best=None
    for r in range(restarts):
        x0=rng.uniform(0,np.pi,dim)
        res=minimize(f,x0,method='COBYLA',options=dict(maxiter=maxiter))
        if best is None or res.fun<best.fun: best=res
    return best


def prep_biased(g, order, q, psi=None, inverse=False):
    """Negation-forced preparation with biased coin: RY so that P(include v | not forced) = q."""
    th = 2 * np.arcsin(np.sqrt(q))
    R = np.array([[np.cos(th / 2), -np.sin(th / 2)], [np.sin(th / 2), np.cos(th / 2)]])
    U = R.T if inverse else R
    if psi is None:
        psi = np.zeros(2 ** g.n, complex); psi[0] = 1
    pos = {v: i for i, v in enumerate(order)}
    for v in (list(order)[::-1] if inverse else list(order)):
        earlier = [u for u in g.N[v] if pos[u] < pos[v]]
        psi = g._apply_1q(psi, v, U, g._zero_mask(v, earlier) if earlier else None)
    return psi

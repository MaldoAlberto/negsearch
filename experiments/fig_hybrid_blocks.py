"""Block diagram of the hybrid circuit (A_q + [cost phase, XY mixer]^p) for the paper.
Writes figures/fig_hybrid_blocks.{pdf,png}."""
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Arc

plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 9})
INK, MUTE, WIRE = '#0b0b0b', '#52514e', '#6b6a66'
C_A, C_A_E = '#d8f1e7', '#1baf7a'        # preparation
C_C, C_C_E = '#dce9f9', '#2a78d6'        # cost phase
C_X, C_X_E = '#fbe1d6', '#eb6834'        # XY mixer
C_M, C_M_E = '#ecebe8', '#6b6a66'        # measurement / classical

fig = plt.figure(figsize=(7.2, 4.3))
ax = fig.add_axes([0, 0.30, 1, 0.70]); ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis('off')


def box(a, x0, y0, x1, y1, fc, ec, text=None, fs=9, lw=1.1, ls='-', tc=INK, z=3):
    a.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle='round,pad=0,rounding_size=1.2',
                               fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))
    if text: a.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha='center', va='center', fontsize=fs, color=tc, zorder=z + 1)


def bundle(a, y, x0, x1, label, n=None, dashed=False):
    a.plot([x0, x1], [y, y], color=WIRE, lw=1.0, ls=(0, (2, 2)) if dashed else '-', zorder=1)
    a.text(x0 - 0.8, y, label, ha='right', va='center', fontsize=8.5, color=INK)
    if n: a.plot([x0 + 2.2, x0 + 3.4], [y - 0.9, y + 0.9], color=WIRE, lw=0.9, zorder=1)


# ---- wires: decision registers of periods 1..T on top, their counters below ---------------------------
ax.set_ylim(0, 68)
X0, X1 = 13, 90
xw = {1: (58, 53), 'T': (39, 34)}                    # y of shorts, longs per period
cw = {1: 22, 'T': 14}                                # y of the counter pair c^(t)
for t, (yS, yL) in xw.items():
    tt = '1' if t == 1 else 'T'
    bundle(ax, yS, X0, X1, rf'$|0\rangle^{{\otimes n}}\;\; x^{{({tt})}}_{{\mathrm{{short}}}}$', n=True)
    bundle(ax, yL, X0, X1, rf'$|0\rangle^{{\otimes n}}\;\; x^{{({tt})}}_{{\mathrm{{long}}}}$', n=True)
    bundle(ax, cw[t], X0, X1, rf'$|0\rangle\;\; c^{{({tt})}}=(L,S)$', n=True)
for y in (45.5, 18):
    ax.text(X0 - 9, y + 0.3, r'$\vdots$', ha='center', va='center', fontsize=12, color=MUTE)
ax.text(X0 - 0.5, 28.5, 'counters', ha='right', va='center', fontsize=7.5, color=MUTE, style='italic')
ax.plot([1, X0 - 0.5], [27.3, 27.3], color='#d7d6d2', lw=0.8)

# ---- A_q: product over periods (prepared in parallel) ---------------------------------------------------
box(ax, 16, 10.5, 32, 61.5, C_A, C_A_E, None)
ax.text(24, 51, r'$A_q$', ha='center', va='center', fontsize=13, color=INK, zorder=5)
ax.text(24, 44.5, r'$=\otimes_t\, A_q^{(t)}$', ha='center', va='center', fontsize=9.5, color=INK, zorder=5)
ax.text(24, 33.5, 'negation-\nforced\nconditions;\nperiods in\nparallel', ha='center', va='center',
        fontsize=7.2, color=MUTE, zorder=5, linespacing=1.15)
ax.text(24, 62.4, 'preparation', ha='center', va='bottom', fontsize=8, color=C_A_E, weight='bold')

# ---- one QAOA layer, repeated p times -------------------------------------------------------------------
ax.add_patch(Rectangle((35.5, 8.0), 43, 56, fc='none', ec=MUTE, lw=0.9, ls=(0, (4, 3)), zorder=2))
ax.text(77.6, 9.0, r'$\times\,p$', ha='right', va='bottom', fontsize=10, color=INK)
ax.text(57, 64.6, r'layer $\ell=1,\dots,p$', ha='center', va='bottom', fontsize=8, color=MUTE)
box(ax, 38.5, 30.5, 54, 61.5, C_C, C_C_E, None)
ax.text(46.25, 50, r'$e^{-i\gamma_\ell C(x)}$', ha='center', va='center', fontsize=11, color=INK, zorder=5)
ax.text(46.25, 42.6, 'objective only,\nno penalty:\n' r'$R_Z$ + sparse $R_{ZZ}$', ha='center', va='center',
        fontsize=7.2, color=MUTE, zorder=5, linespacing=1.15)
ax.text(46.25, 20.5, 'counters idle', ha='center', va='center', fontsize=7, color=MUTE, style='italic',
        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none'), zorder=5)
for t, (yS, yL) in xw.items():
    for y in (yS, yL):
        box(ax, 58, y - 2.0, 75.5, y + 2.0, C_X, C_X_E, rf'$e^{{-i\beta_\ell H_{{XY}}}}$', fs=8.5)
ax.text(66.75, 45.6, 'XY inside each (period,\ndirection) block:\nconserves $L$ and $S$', ha='center', va='center',
        fontsize=7.2, color=MUTE, linespacing=1.15)

# ---- measurement + classical post-processing ------------------------------------------------------------
def meter(a, x, y):
    a.add_patch(FancyBboxPatch((x - 1.9, y - 1.9), 3.8, 3.8, boxstyle='round,pad=0,rounding_size=0.5',
                               fc=C_M, ec=C_M_E, lw=0.9, zorder=3))
    a.add_patch(Arc((x, y - 1.0), 2.8, 2.8, theta1=15, theta2=165, color=INK, lw=0.8, zorder=4))
    a.plot([x, x + 1.1], [y - 1.0, y + 1.1], color=INK, lw=0.8, zorder=4)
for t, (yS, yL) in xw.items():
    for y in (yS, yL, cw[t]): meter(ax, 82.5, y)
ax.text(82.5, 62.4, 'measure', ha='center', va='bottom', fontsize=8, color=MUTE)
box(ax, 87, 10.5, 99.5, 61.5, C_M, C_M_E, None, lw=0.9)
ax.text(93.25, 36, 'classical:\n\nscore with\nthe exact\nobjective\n\ncheck\n' r'$c^{(t)}$ vs.' '\nmeasured\n' r'$(L,S)$',
        ha='center', va='center', fontsize=7.2, color=INK, linespacing=1.2, zorder=5)
ax.text(1, 67.5, '(a)', fontsize=10, weight='bold', va='top')

# ---- (b) inside A_q^{(t)}: one variable step --------------------------------------------------------------
bx = fig.add_axes([0, 0.0, 1, 0.29]); bx.set_xlim(0, 100); bx.set_ylim(0, 26); bx.axis('off')
bx.text(1, 25.5, '(b)', fontsize=10, weight='bold', va='top')
bx.text(5.5, 25.2, r'inside $A_q^{(t)}$: for each variable $x_k$ (shorts first), with $(L,S)$ = positions chosen so far',
        fontsize=8.5, va='top', color=INK)
yx, yc = 15.5, 6.5
bundle(bx, yx, 14, 80, r'$x_k\;|0\rangle$')
bundle(bx, yc, 14, 80, r'$c^{(t)}=(L,S)$', n=True)
box(bx, 18, yx - 2.6, 29, yx + 2.6, C_A, C_A_E, r'$R_Y(\theta_q)$', fs=9)
bx.text(23.5, yx - 5.6, 'coin, ' r'$\sin^2\!\frac{\theta_q}{2}=q$', ha='center', fontsize=7.2, color=MUTE)
box(bx, 35, yx - 2.6, 49, yx + 2.6, C_A, C_A_E, r'$R_Y(-\theta_q)$', fs=9)
bx.plot([42, 42], [yc, yx - 2.6], color=C_A_E, lw=1.1, zorder=2); bx.plot(42, yc, 'o', ms=5, color=C_A_E, zorder=3)
bx.text(42, yc - 3.2, r'if $(L,S)\in F_0(k)$', ha='center', fontsize=7.4, color=INK)
box(bx, 54, yx - 2.6, 69, yx + 2.6, C_A, C_A_E, r'$R_Y(\pi-\theta_q)$', fs=9)
bx.plot([61.5, 61.5], [yc, yx - 2.6], color=C_A_E, lw=1.1, zorder=2); bx.plot(61.5, yc, 'o', ms=5, color=C_A_E, zorder=3)
bx.text(61.5, yc - 3.2, r'if $(L,S)\in F_1(k)$', ha='center', fontsize=7.4, color=INK)
# counter increment
bx.plot([74, 74], [yc, yx], color=INK, lw=1.0, zorder=2); bx.plot(74, yx, 'o', ms=5, color=INK, zorder=3)
box(bx, 71, yc - 2.4, 77, yc + 2.4, 'white', INK, r'$+1$', fs=8.5, lw=1.0, z=4)
bx.text(74, yc - 5.4, r'$L$ or $S$ += $x_k$', ha='center', fontsize=7.4, color=INK)
bx.text(89.5, 11.5, r'$F_0(k)$: $x_k=1$ would make' '\nperiod $t$ infeasible' r' $\Rightarrow x_k=0$' '\n'
        r'($F_1$: vice versa).' '\nNo dead ends. Counters are' '\nnot uncomputed (XY' '\n' r'conserves $L,S$).',
        ha='center', va='center', fontsize=6.6, color=MUTE, linespacing=1.25,
        bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#d7d6d2', lw=0.8))
fig.savefig('figures/fig_hybrid_blocks.pdf'); fig.savefig('figures/fig_hybrid_blocks.png', dpi=220)
print('ok')

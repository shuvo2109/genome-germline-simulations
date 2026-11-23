"""
Supplementary figure (Fig. S2): reversible vs. categorical de-Darwinization.

Addresses the reviewer point that a chromosome transcribing non-replicating
ribozymes could de-Darwinize the catalytic workforce before translation.

Architecture C ("transcribed-ribozyme chromosome"): a single low-copy chromosome
of k genes (replicated once per protocell generation, per-gene mutation rate U,
purifying selection at the protocell level, exactly as the post-genome model).
Genes are transcribed into a ribozyme workforce. Transcripts are NON-replicating
(de-Darwinized) so long as no replicase acts on them. Because a transcribed
ribozyme is the same chemical class as the chromosome, it can become replicable
again: each transcript type (re)acquires replicability with per-gene probability p
per generation, so a cell re-Darwinizes with per-cell probability 1-(1-p)^k (keeping
all k transcript types non-replicable becomes harder as k grows). Once a replicase
acts, that lineage's workforce evolves under the pre-genome within-cell dynamics
(Muller's ratchet + cheaters). Between-protocell selection still purges re-Darwinized
lineages. No horizontal parasite spread between cells is modeled, which is
conservative (it would only make re-emergence more damaging).

Limits: p = 0 is identical to the post-genome (translation) model (categorical,
irreversible de-Darwinization); large p reproduces the pre-genome model.

Usage:  python make_supplement_chromosome.py
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'font.family': 'serif', 'font.size': 9, 'axes.labelsize': 10,
    'axes.titlesize': 11, 'legend.fontsize': 7, 'xtick.labelsize': 8,
    'ytick.labelsize': 8, 'lines.linewidth': 1.5, 'figure.dpi': 300,
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'mathtext.default': 'regular',
})

pre_color = '#D55E00'    # orange  (pre-genome)
post_color = '#0072B2'   # blue    (post-genome / translation, p = 0)
chrom_colors = ['#9CD3B8', '#1B9E77', '#0B5A43']  # greens, light->dark for rising p


# ── models (pre- and post-genome identical to make_figures.py) ──────
def simulate_pregenome(M, k, n, mu, s, tau, G, seed):
    rng = np.random.default_rng(seed); x = np.ones((M, k)); mf = np.zeros(G)
    for g in range(G):
        for _ in range(tau):
            x = x * (1 - mu); x = x / (x + (1 - x) * (1 + s))
        x = rng.binomial(n, np.clip(x, 0, 1)).astype(float) / n
        f = np.prod(x, axis=1); mf[g] = f.mean(); tot = f.sum()
        if tot > 0:
            x = x[rng.choice(M, M, p=f / tot, replace=True)].copy()
    return mf


def simulate_postgenome(M, k, U, mu_t, G, seed, delta=0.01):
    rng = np.random.default_rng(seed); gen = np.ones((M, k), np.int8); mf = np.zeros(G)
    for g in range(G):
        gf = gen * (1 - mu_t) + (1 - gen) * delta
        f = np.prod(gf, axis=1); mf[g] = f.mean(); tot = f.sum()
        if tot > 0:
            gen = gen[rng.choice(M, M, p=f / tot, replace=True)].copy()
        gen[rng.random((M, k)) < U] = 0
    return mf


def simulate_chromosome(M, k, n, mu, s, tau, U, mu_t, p, G, seed, delta=0.01):
    rng = np.random.default_rng(seed)
    gen = np.ones((M, k), np.int8)
    darw = np.zeros(M, bool)            # has a replicase re-emerged in this lineage?
    x = np.ones((M, k))                 # workforce functional fraction
    mf = np.zeros(G)
    p_cell = 1 - (1 - p) ** k           # per-cell re-emergence rate from per-gene p
    for g in range(G):
        newly = (~darw) & (rng.random(M) < p_cell)
        if newly.any():
            x[newly] = gen[newly] * (1 - mu_t) + (1 - gen[newly]) * delta
        darw |= newly
        nd = ~darw
        x[nd] = gen[nd] * (1 - mu_t) + (1 - gen[nd]) * delta           # fresh transcripts
        if darw.any():                                                  # re-Darwinized: pre-genome dynamics
            xd = x[darw]
            for _ in range(tau):
                xd = xd * (1 - mu); xd = xd / (xd + (1 - xd) * (1 + s))
            xd = rng.binomial(n, np.clip(xd, 0, 1)).astype(float) / n
            gd = gen[darw]; xd = np.where(gd == 1, xd, delta)
            x[darw] = xd
        f = np.prod(x, axis=1); mf[g] = f.mean(); tot = f.sum()
        if tot > 0:
            par = rng.choice(M, M, p=f / tot, replace=True)
            gen = gen[par].copy(); darw = darw[par].copy(); x = x[par].copy()
        gen[rng.random((M, k)) < U] = 0
    return mf


def make_figure():
    M, G = 100, 500
    mu, s, tau, n = 0.005, 0.02, 10, 10
    U, mu_t = 0.005, 0.005
    reps = 5
    p_vals = [1e-3, 3e-3, 1e-2]                 # per-gene re-emergence rates
    eq = lambda mf: mf[400:].mean()

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.6, 3.4))

    # ── Panel A: trajectories at fixed k = 50 ───────────────────────
    kA = 50
    pre = np.mean([simulate_pregenome(M, kA, n, mu, s, tau, G, 10 + r) for r in range(reps)], axis=0)
    post = np.mean([simulate_postgenome(M, kA, U, mu_t, G, 50 + r) for r in range(reps)], axis=0)
    axA.plot(np.arange(G), post, color=post_color, lw=2, label='Translation (p = 0)')
    for c, p in zip(chrom_colors, p_vals):
        ch = np.mean([simulate_chromosome(M, kA, n, mu, s, tau, U, mu_t, p, G, 100 + r + int(p * 1e5))
                      for r in range(reps)], axis=0)
        axA.plot(np.arange(G), ch, color=c, label=f'Chromosome, p = {p:g}')
    axA.plot(np.arange(G), pre, color=pre_color, ls=':', label='Pre-genome')
    axA.set_xlabel('Protocell generations'); axA.set_ylabel('Mean protocell fitness')
    axA.set_xlim(0, 500); axA.set_ylim(0, 1.02)
    axA.set_title('A', loc='left', fontweight='bold', pad=6)
    axA.text(0.5, 0.08, f'k = {kA}', transform=axA.transAxes, ha='center', fontsize=8)
    axA.legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.95,
               edgecolor='0.8', handlelength=1.6, fontsize=6.3)

    # ── Panel B: equilibrium fitness vs k ───────────────────────────
    k_scan = np.array([5, 10, 15, 20, 30, 40, 50, 75, 100])

    def scan(fn):
        m, sd = [], []
        for k in k_scan:
            vals = [eq(fn(k, 1000 + r + k * 7)) for r in range(reps)]
            m.append(np.mean(vals)); sd.append(np.std(vals))
        return np.array(m), np.array(sd)

    post_m, post_s = scan(lambda k, sd: simulate_postgenome(M, k, U, mu_t, G, sd))
    pre_m, pre_s = scan(lambda k, sd: simulate_pregenome(M, k, n, mu, s, tau, G, sd + 3000))
    axB.errorbar(k_scan, post_m, yerr=post_s, color=post_color, marker='s',
                 markersize=4, capsize=2.5, lw=1.8, label='Translation (p = 0)', zorder=5)
    for c, p in zip(chrom_colors, p_vals):
        cm, cs = scan(lambda k, sd, p=p: simulate_chromosome(M, k, n, mu, s, tau, U, mu_t, p, G, sd + int(p * 1e6)))
        axB.errorbar(k_scan, cm, yerr=cs, color=c, marker='o', markersize=3.5,
                     capsize=2.5, label=f'Chromosome, p = {p:g}')
    axB.errorbar(k_scan, pre_m, yerr=pre_s, color=pre_color, marker='^',
                 markersize=4, capsize=2.5, ls=':', label='Pre-genome')
    axB.set_xlabel('Network complexity, k'); axB.set_ylabel('Equilibrium fitness')
    axB.set_xlim(0, 105); axB.set_ylim(0, 1.02)
    axB.set_title('B', loc='left', fontweight='bold', pad=6)
    axB.legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.95,
               edgecolor='0.8', handlelength=1.6, fontsize=6.3)

    fig.tight_layout(w_pad=2.0)
    fig.savefig('supplement_chromosome.pdf')
    plt.close(fig)
    print('Figure S2 (reversible vs categorical de-Darwinization) done.')


if __name__ == '__main__':
    make_figure()

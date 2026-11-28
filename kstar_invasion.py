"""
Generate new figure (Fig 5) for complexity ceiling, invasion fitness, critical complexity k_hat and evolutionarily
stable h_ESS, and ratchet condition for adding a catalyst
evolutionarily stable templating fraction h_ESS(k).

Figure 5 : Ceiling and invasion. (3 panels).
  - A: Complexity ceiling for a viability threshold
  - B: Critical complexity
  - C: Stable templating fraction

Run:
    python kstar_invasion.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import hmodel as H


# Parameters
p = H.Par()
p_eq = H.Par(G=400)

WC_SIM = 0.5
XC_ODE = 0.0693
c0 = 0.02

# Control
print_results = True


# =========================================================================================
# A. Complexity ceiling k*(h): closed form vs simulation
# =========================================================================================
def equilibrium_fitness(h, k, reps=3):
    """Mean equilibrium fitness across independent simulations."""
    return np.mean([
        H.simulate(h, k, p_eq, 100 + r + 7 * k)[300:].mean()
        for r in range(reps)
    ])


def simulated_ceiling(h, ks):
    """Estimate k* by log-fitness interpolation."""
    W = np.array([equilibrium_fitness(h, k) for k in ks])
    valid = W > 1e-4

    if valid.sum() < 2 or not W[valid].min() < WC_SIM < W[valid].max():
        return np.nan

    return np.interp(
        np.log(WC_SIM), np.log(W[valid])[::-1], ks[valid][::-1]
    )


ks = np.array([1, 2, 3, 5, 8, 12, 18, 25, 35, 50, 70, 100])
h_list = [1.0, 0.95, 0.9, 0.8, 0.5, 0.0]

sim_k = {}

if print_results:
    print("A. Complexity ceiling k*(h): closed form vs simulation")
    print(f"{'h':>6} {'load(h)':>10} {'k* formula':>12} {'k* sim':>10} {'k*(h)/k*(0)':>14}")

for h in h_list:
    sim_k[h] = simulated_ceiling(h, ks)

    load = H.load(h, p) if h < 1 else np.inf
    kf = H.kstar(h, WC_SIM, p) if h < 1 else np.nan
    ratio = H.load(0, p) / load if h < 1 else 0.0

    if print_results:
        print(f"{h:6.2f} {load:10.4f} {kf:12.2f} {sim_k[h]:10.2f} {ratio:14.4f}")

if print_results:
    print("  h = 1: no closed-form fixed point; simulation includes group selection.")
    print("  h > ~0.9: the closed form underestimates fitness by ignoring group selection.")


# =========================================================================================
# B. Invasion fitness: simulation vs analytical prediction
# =========================================================================================
def invasion_sim(h_res, h_inv, k, c0, M=2000, f0=0.05, burn=120, T=9, seed=0):
    rng = np.random.default_rng(seed)

    h = np.full(M, h_res, dtype=float)
    x = np.ones((M, k))
    g = np.ones((M, k), dtype=np.int8)

    # Equilibrate resident population
    for _ in range(burn):
        x = H.within_cell(x, g, h, p, rng)
        W = H.fitness(x, h, c0)
        x, g, h = H.reproduce(x, g, h, W, p, rng)

    # Introduce rare invaders
    h[rng.choice(M, int(f0 * M), replace=False)] = h_inv

    fitness = []

    for _ in range(T):
        x = H.within_cell(x, g, h, p, rng)
        W = H.fitness(x, h, c0)

        inv = h < h_res - 1e-9

        if inv.sum() < 5 or (~inv).sum() < 5:
            break

        fitness.append(W[inv].mean() / W[~inv].mean() - 1)
        x, g, h = H.reproduce(x, g, h, W, p, rng)

    return np.mean(fitness[3:]) if len(fitness) > 5 else np.nan


if print_results:
    print(f"\nB. Invasion fitness s_inv (c0 = {c0}): simulation vs analytical")
    print(f"{'h_res -> h_inv':>16} {'k':>5} {'simulation':>18} {'analytic':>10} {'k_hat':>9}")

    for h_res, h_inv in [(0.8, 0.7), (0.5, 0.4), (0.2, 0.1), (0.9, 0.8), (0.95, 0.9)]:
        for k in [20, 50]:
            sims = [
                invasion_sim(h_res, h_inv, k, c0, seed=r + 3 * k)
                for r in range(3)
            ]

            analytic = H.Wstar(h_inv, k, p, c0) / H.Wstar(h_res, k, p, c0) - 1
            kh = H.k_hat(h_res, h_inv, c0, p)

            print(
                f"  {h_res:.1f} -> {h_inv:.1f} {k:5d}"
                f" {np.nanmean(sims):+8.4f} ± {np.nanstd(sims):.4f}"
                f" {analytic:+10.4f} {kh:9.2f}"
            )

    print("  h_res >= 0.9: analytical fitness can overestimate invasion success.")


# =========================================================================================
# C. Critical complexity k_hat and evolutionarily stable h_ESS
# =========================================================================================
def h_ess(k, c0):
    """Templating fraction maximizing equilibrium fitness."""
    def objective(h):
        return -(np.log(1 - c0 * (1 - h)) - k * H.load(h, p))

    result = minimize_scalar(
        objective, bounds=(0, 0.9999),
        method="bounded", options={"xatol": 1e-8}
    )

    return result.x


costs = [0.05, 0.1, 0.3]

if print_results:
    print(f"\nC. Critical complexity k_hat vs ceiling k*(h) (Xc = {XC_ODE:.4f})")
    print(f"{'h_res':>7} {'k*(h)':>10}" +
          "".join(f"{'c0=' + str(c):>12}" for c in costs))

    for h in [0.95, 0.9, 0.8, 0.6, 0.4, 0.2, 0.1]:
        kh = [H.k_hat(h, max(h - 0.05, 0), c, p) for c in costs]

        print(
            f"{h:7.2f} {H.kstar(h, XC_ODE, p):10.1f}" +
            "".join(f"{v:12.2f}" for v in kh)
        )

    print("\nEvolutionarily stable h_ESS(k) = argmax W*(h,k)")
    print(f"{'k':>6}" + "".join(f"{'c0=' + str(c):>12}" for c in costs))

    for k in [1, 2, 5, 10, 20, 50]:
        print(
            f"{k:6d}" +
            "".join(f"{h_ess(k, c):12.3f}" for c in costs)
        )


# =========================================================================================
# D. Ratchet condition for adding a catalyst
# =========================================================================================
def minimum_benefit(load):
    """Minimum fitness benefit required to retain an additional catalyst."""
    return np.expm1(load)


if print_results:
    print("\nD. Catalyst retention: b > exp(load(h)) - 1")

    W1 = np.mean([
        H.simulate(1.0, 1, p_eq, 500 + r)[300:].mean()
        for r in range(3)
    ])

    load_sim = -np.log(W1)

    print(f"  h = 1.0 (simulation): load = {load_sim:.3f}, b_min = {minimum_benefit(load_sim):.2f}")

    for h in [0.9, 0.5, 0.0]:
        load = H.load(h, p)
        print(f"  h = {h:.1f}: load = {load:.4f}, b_min = {minimum_benefit(load):.4f}")


# =========================================================================================
# Figure: complexity ceiling, invasion threshold, and evolutionary stability
# =========================================================================================
fig, ax = plt.subplots(1, 3, figsize=(10, 3.1))

h = np.linspace(0, 0.97, 100)
colors = ["#56B4E9", "#0072B2", "#003f66"]

# A. Complexity ceiling
ax[0].semilogy(
    h, [H.kstar(v, WC_SIM, p) for v in h],
    color="#0072B2", label=r"$\ln(1/W_c)/\ell(h)$"
)

h_sim = [v for v in h_list if v < 1]
ax[0].semilogy(
    h_sim, [sim_k[v] for v in h_sim],
    "ko", ms=4, label="simulation"
)
ax[0].semilogy(
    [1.0], [sim_k[1.0]], "s",
    color="#D55E00", ms=5, label="$h=1$ (sim.)"
)

ax[0].set(
    xlabel="templating fraction $h$",
    ylabel="ceiling $k^*$ ($W_c=0.5$)",
    title="A"
)
ax[0].legend(frameon=False, fontsize=7)


# B. Critical invasion complexity
for c, color in zip(costs, colors):
    kh = [max(H.k_hat(v, v - 0.02, c, p), 1e-3) for v in h[2:-3]]
    ax[1].semilogy(h[2:-3], kh, color=color, label=fr"$\hat k$, $c_0$={c}")

ax[1].semilogy(
    h, [H.kstar(v, XC_ODE, p) for v in h],
    "--", color="#D55E00", label=fr"$k^*(h)$, $X_c$={XC_ODE:.3f}"
)

ax[1].set(
    xlabel="resident $h$",
    ylabel="complexity $k$",
    ylim=(1e-2, 1e3),
    title="B"
)
ax[1].legend(frameon=False, fontsize=6.5)


# C. Evolutionarily stable templating fraction
k = np.arange(1, 61)

for c, color in zip(costs, colors):
    ax[2].plot(
        k, [h_ess(v, c) for v in k],
        color=color, label=f"$c_0$={c}"
    )

ax[2].set(
    xlabel="complexity $k$",
    ylabel=r"stable templating fraction $h_{ESS}$",
    title="C"
)
ax[2].legend(frameon=False, fontsize=7)

fig.tight_layout()
fig.savefig("fig_h_kstar_invasion.pdf")
print("\nSaved fig_h_kstar_invasion.pdf")
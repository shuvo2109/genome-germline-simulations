"""
Generate new figure (Fig 4) for fixed points and stability of the h-model with an explicit resource

Figure 4 : De-Darwinization as a continuum. (2 panels).
  - A: Within-protocell fixed point of the functional fraction, and
       The heritability of the workforce across a protocell generation.
  - B: Complexity ceiling with an intact genome

Run:
    python fixed_points_stability.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

# Parameters 
mu = 0.005  # mu, s, mu_t from original simulations
s = 0.02
mu_t = 0.005
gamma = 1 - mu_t

phi0 = 1.0
phi = lambda E: phi0 * E

d = 0.4
theta = 1.0
eta = 0.05
dE = 0.1
kappa = 0.02


# Control
print_results = True

# =========================================================================================
# A.  Closed-form within-cell fixed point x*(h)  vs  numerical integration
# =========================================================================================
def xstar(h):
    """Smaller root of  h*s*x^2 - (h*mu + h*s + 1 - h) x + (1-h)*gamma = 0.   (s=0:  gamma (1-h)/(1-h+h mu))"""
    if h <= 0:
        return gamma
    if h >= 1:
        return 0.0

    a = h * s
    b = -(h * mu + h * s + 1 - h)
    c = (1 - h) * gamma

    return (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)

if print_results:
    print("A. x*(h): closed form vs integrating dx/dt (phi=1) from x=1 for 2000 time units")
    print(f"{'h':>6} {'x*(closed)':>12} {'x(t=2000)':>12} {'f′(x*)':>10}")

    for h in [0, 0.25, 0.5, 0.8, 0.9, 0.95, 0.99]:
        f = lambda t, x: (1 - h) * (gamma - x) - h * x * (mu + s * (1 - x))

        xs = xstar(h)

        x_end = solve_ivp(
            f,
            t_span=(0, 2000),
            y0=[1.0],
            rtol=1e-10,
            atol=1e-12
        ).y[0, -1]

        fp = -(1 - h) - h * mu - h * s + 2 * h * s * xs

        if print_results:
            print(f"{h:6.2f} {xs:12.6f} {x_end:12.6f} {fp:+10.4f}")

    if print_results:
        print("  0 <= h < 1: unique, globally stable equilibrium in (0, 1).")
        print("  h = 1: no interior equilibrium; x approaches 0 (collapse).")


# =========================================================================================
# B.  Full (C, D, E) system: fixed points + Jacobian eigenvalues, three versions of the germ channel
# =========================================================================================
def rhs(y, h, g=1.0, share=True, kap=0.0):
    C, D, E = y
    N = C + D
    f = phi(E)

    if share:
        germ = N  # share=False is the germ channel in absolute rate, no factor N
    else:
        germ = 1

    dC = h * f * (1 - mu) * C + (1 - h) * f * germ * g * (1 - mu_t) - d * C - kap * N * C
    dD = h * f * ((1 + s) * D + mu * C) + (1 - h) * f * germ * ((1 - g) + g * mu_t) - d * D - kap * N * D
    dEd = theta * C * (1 - E) - eta * N * E - dE * E

    return np.array([dC, dD, dEd])

def jacobian(y, *a, **k):
    J = np.zeros((3, 3))
    f0 = rhs(y, *a, **k)

    for i in range(3):
        yy = np.array(y, float)
        yy[i] += 1e-6

        J[:, i] = (rhs(yy, *a, **k) - f0) / 1e-6

    return J

def fixed_points(h, **kw):
    roots = []
    rng = np.random.default_rng(0)

    for _ in range(300):
        y0 = [rng.uniform(0, 30), rng.uniform(0, 30), rng.uniform(0, 1)]

        root, _, ier, _ = fsolve(
            lambda y: rhs(y, h, **kw), y0, full_output=True
        )

        valid = (
            ier == 1
            and np.all(root[:2] >= -1e-9)
            and -1e-9 <= root[2] <= 1 + 1e-9
        )

        if valid:
            root = np.maximum(root, 0)
            if not any(np.allclose(root, q, atol=1e-4) for q in roots):
                roots.append(root)

    return roots

if print_results:
    print("\nB. Fixed points (C, D, E) and stability")

    models = [
        ("(i) Absolute germ rate, no crowding", dict(share=False)),
        ("(ii) Share form, no crowding", dict(share=True)),
        ("(iii) Share form + crowding", dict(share=True, kap=kappa)),
    ]

    for label, kw in models:
        print(f"\n  {label}")

        for h in [0.0, 0.5, 0.9]:
            roots = sorted(fixed_points(h, **kw), key=lambda r: -r[0])

            for C, D, E in roots:
                N = C + D
                x = C / N if N > 1e-9 else np.nan

                eig = np.linalg.eigvals(jacobian([C, D, E], h, **kw))
                max_re = eig.real.max()
                kind = "stable" if max_re < -1e-9 else "saddle/unstable"

                print(
                    f"    h={h:.1f}  C={C:8.3f}  D={D:8.3f}  E={E:.3f}"
                    f"  N={N:8.2f}  x={x:.3f}"
                    f"  max Re(eig)={max_re:+.3f}  {kind}"
                )

    print("\nConclusions:")
    print("  (i) Stable interior state, but x depends on N; extinction is unstable.")
    print("  (ii) No bound on N; the interior state is a saddle.")
    print("  (iii) x = x*(h), with stable viable/extinct states and an intervening saddle.")


# =========================================================================================
# C.  Resource-viability threshold Xc and the complexity ceiling k*(h)
# =========================================================================================
def Xc(h):
    """Approximate the minimum pathway integrity X allowing a viable state."""
    xs = xstar(h)
    a = phi0 * (1 + h * s * (1 - xs))

    X = np.logspace(-6, 0, 6000)

    A = a * (theta * X + eta)
    B = a * theta * X + d * (theta * X + eta) - kappa * dE
    C = d * theta * X

    disc = B**2 - 4 * A * C
    valid = disc >= 0

    sqrt_disc = np.sqrt(np.maximum(disc, 0))
    E1 = (B + sqrt_disc) / (2 * A)
    E2 = (B - sqrt_disc) / (2 * A)

    Emax = theta * X / (theta * X + eta)

    viable = valid & (
        ((a * E1 > d) & (E1 > 0) & (E1 < Emax))
        | ((a * E2 > d) & (E2 > 0) & (E2 < Emax))
    )

    return X[np.argmax(viable)] if np.any(viable) else np.nan


def persists(h, k, T=5000):
    def ode(t, y):
        x, N, E = y
        growth = phi(E)

        return [
            growth * ((1 - h) * (gamma - x) - h * x * (mu + s * (1 - x))),
            N * (growth * (1 + h * s * (1 - x)) - d - kappa * N),
            theta * N * max(x, 0)**k * (1 - E) - eta * N * E - dE * E,
        ]

    sol = solve_ivp(
        ode, (0, T), [gamma, 20.0, 0.9],
        method="LSODA", rtol=1e-8, atol=1e-10
    )

    return sol.success and sol.y[1, -1] > 1.0


def complexity_ceiling(h):
    xs, xc = xstar(h), Xc(h)

    k_formula = np.log(1 / xc) / (-np.log(xs))

    lo, hi = 1, int(2 * k_formula) + 5

    while hi - lo > 1:
        mid = (lo + hi) // 2

        if persists(h, mid):
            lo = mid
        else:
            hi = mid

    return xs, xc, k_formula, lo

if print_results:
    print("\nC. Complexity ceiling")
    print(f"{'h':>6} {'x*':>8} {'Xc':>10} {'k* formula':>12} {'k* ODE':>9}")

hs_plot = [0.0, 0.5, 0.8, 0.9, 0.95]
kf_plot, kode_plot = [], []

for h in hs_plot:
    xs, xc, kf, kode = complexity_ceiling(h)

    print(f"{h:6.2f} {xs:8.4f} {xc:10.5f} {kf:12.2f} {kode:9d}")

    kf_plot.append(kf)
    kode_plot.append(kode)

# ---------------- figure ----------------
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1))

# A. Functional fraction and heritability
h = np.linspace(0, 0.995, 300)

ax[0].plot(h, [xstar(v) for v in h],
           color="#0072B2", label=r"$x^*(h)$")
ax[0].plot(h, h**10, "--",
           color="#D55E00", label=r"$H \approx h^{10}$")

ax[0].set(
    xlabel="templating fraction $h$",
    ylabel="functional fraction / heritability",
    title="A"
)
ax[0].legend(frameon=False, fontsize=7)


# B. Complexity ceiling
h = np.linspace(0, 0.97, 60)
k = [np.log(1 / Xc(v)) / (-np.log(xstar(v))) for v in h]

ax[1].semilogy(h, k, color="#0072B2", label=r"$k^*$ (formula)")
ax[1].semilogy(hs_plot, kode_plot, "ko", ms=4, label="ODE")

ax[1].set(
    xlabel="templating fraction $h$",
    ylabel="complexity ceiling $k^*$",
    title="B"
)
ax[1].legend(frameon=False, fontsize=7)

fig.tight_layout()
fig.savefig("fig_h_fixed_points.pdf")
print("\nSaved fig_h_fixed_points.pdf")

import numpy as np
from scipy.optimize import brentq


class Par:
    def __init__(self, M=100, G=500, mu=0.005, s=0.02, tau=10, n=10, U=0.005, mu_t=0.005, delta=0.01):
        self.M, self.G, self.mu, self.s, self.tau, self.n = M, G, mu, s, tau, n
        self.U, self.mu_t, self.delta = U, mu_t, delta


def templated_round(x, mu, s):
    """original paper's per-round update for COPIED catalysts: mutation (functional->dud), then cheater advantage s."""
    x = x * (1 - mu)
    return x / (x + (1 - x) * (1 + s))


def within_cell(x, g, hv, p, rng):
    """tau rounds of  x <- h*templated_round(x) + (1-h)*gamma ;  then finite-copy sampling of the templated share."""
    H = hv[:, None]
    gamma = g * (1 - p.mu_t) + (1 - g) * p.delta
    for _ in range(p.tau):
        x = H * templated_round(x, p.mu, p.s) + (1 - H) * gamma
    if hv.max() > 0:                        # no RNG draw when nothing is templated (keeps stream identical at h=0)
        xs = rng.binomial(p.n, np.clip(x, 0, 1)).astype(float) / p.n
        x = (1 - H) * x + H * xs
    return x


def fitness(x, hv, c=0.0):
    """Protocell fitness = prod_i x_i  (original paper's multiplicative form) times a fixed cost c*(1-h) of germ-line machinery."""
    return np.prod(x, axis=1) * (1 - c * (1 - hv))


def reproduce(x, g, hv, W, p, rng):
    """Between-cell selection (resample proportional to W), then germ-line mutation (intact -> mutated, prob U)."""
    tot = W.sum()
    if tot > 0:
        par = rng.choice(len(W), len(W), p=W / tot, replace=True)
        x, g, hv = x[par].copy(), g[par].copy(), hv[par].copy()
    if hv.min() < 1:                        # genome irrelevant (and not drawn) when h=1 for every cell
        g[rng.random(g.shape) < p.U] = 0
    return x, g, hv


def simulate(h, k, p, seed, c=0.0, p_up=0.0, h_up=1.0, return_h=False):
    """
    Population of p.M protocells over p.G generations. Returns mean fitness per generation.
    h may be a scalar or an (M,) vector.  p_up>0 adds heritable UPWARD mutation of h (a transcript becomes
    replicable), per-gene prob p_up -> per-cell prob 1-(1-p_up)^k, h jumps to h_up. p_up reproduces original paper's Fig. S2.
    """
    rng = np.random.default_rng(seed)
    M = p.M
    hv = np.broadcast_to(np.asarray(h, float), (M,)).copy()
    x = np.ones((M, k)); g = np.ones((M, k), np.int8)
    meanW = np.zeros(p.G); meanh = np.zeros(p.G)
    p_cell = 1 - (1 - p_up) ** k
    for t in range(p.G):
        if p_up > 0:
            flip = (hv < h_up) & (rng.random(M) < p_cell)
            hv[flip] = h_up
        x = within_cell(x, g, hv, p, rng)
        W = fitness(x, hv, c)
        meanW[t], meanh[t] = W.mean(), hv.mean()
        x, g, hv = reproduce(x, g, hv, W, p, rng)
    return (meanW, meanh) if return_h else meanW


# ------------------------------------------------------------------ closed-form (quasi-steady-state) results
def xstar(h, gamma, p):
    """Within-cell fixed point of the per-round map  x = h*X(x) + (1-h)*gamma  (unique, globally stable on (0,gamma])."""
    if h <= 0: return gamma
    if h >= 1: return 0.0
    f = lambda x: h * templated_round(x, p.mu, p.s) + (1 - h) * gamma - x
    return brentq(f, 1e-14, 1.0)


def xstar_closed(h, gamma, mu, s):
    """Continuous-time closed form: smaller root of  h s x^2 - (h mu + h s + 1 - h) x + (1-h) gamma = 0.
       For s=0:  x* = gamma (1-h) / (1-h+h mu)."""
    if h <= 0: return gamma
    if h >= 1: return 0.0
    a, b, c = h * s, -(h * mu + h * s + 1 - h), (1 - h) * gamma
    if a == 0: return -c / b
    return (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)


def load(h, p):
    """Per-type fitness load  l(h) = -ln[(1-U) x*(h)]  (Haldane-Muller germ-line load U  +  workforce load at x*)."""
    return -np.log((1 - p.U) * xstar(h, 1 - p.mu_t, p))


def Wstar(h, k, p, c0=0.0):
    """Equilibrium mean fitness  W*(h,k) = (1 - c0(1-h)) exp(-k l(h)).  Conservative for h >~ 0.9 (ignores group selection
       acting on the heritable remainder, which is ~h^tau)."""
    return (1 - c0 * (1 - h)) * np.exp(-k * load(h, p))


def kstar(h, Wc, p):
    """Complexity ceiling: largest k with W*(h,k) >= Wc  ->  ln(1/Wc)/l(h)."""
    return np.log(1 / Wc) / load(h, p)


def k_hat(h_res, h_inv, c0, p):
    """Critical complexity above which a rare lineage with h_inv<h_res invades (s_inv>0)."""
    num = np.log((1 - c0 * (1 - h_res)) / (1 - c0 * (1 - h_inv)))
    den = load(h_res, p) - load(h_inv, p)
    return num / den

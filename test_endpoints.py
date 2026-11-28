"""
Check that the generalized h-model exactly reproduces the original
pre-genome (h=1) and post-genome (h=0) simulations using identical seeds.

Expected result: zero fitness difference across all generations.

Run:
    python test_endpoints.py
"""

import numpy as np
import hmodel as H
import make_figures as MF

p = H.Par()
ok = True

print("Endpoint validation: original vs generalized h-model")
print(f"{'k':>5} {'h=1 (max diff)':>16} {'h=0 (max diff)':>16}")

for k in [5, 20, 50]:
    # Pre-genome (h = 1)
    pre_old = MF.simulate_pregenome(
        p.M, k, p.n, p.mu, p.s, p.tau, p.G, seed=123 + k
    )
    pre_new = H.simulate(1.0, k, p, seed=123 + k)

    # Post-genome (h = 0)
    post_old = MF.simulate_postgenome(
        p.M, k, p.U, p.mu_t, p.G, seed=77 + k
    )
    post_new = H.simulate(0.0, k, p, seed=77 + k)

    d1 = np.max(np.abs(pre_old - pre_new))
    d0 = np.max(np.abs(post_old - post_new))

    ok &= (d1 == d0 == 0.0)

    print(f"{k:5d} {d1:16.1e} {d0:16.1e}")

print("\nPASS" if ok else "\nFAIL")
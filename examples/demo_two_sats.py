"""
demo_two_sats.py
-----------------
End-to-end demo: build two satellites in slightly different, crossing
orbital planes, time them so they pass close to each other, then run the
full conjunction-assessment pipeline: coarse screening -> TCA refinement
-> probability of collision.

Run:  python3 examples/demo_two_sats.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from orbits import (
    kepler_to_state, propagate, R_EARTH, MU_EARTH,
    plane_crossing_direction, true_anomaly_for_direction,
)
import conjunction as ca


def build_scenario():
    """
    Two ~700 km altitude, near-polar orbits whose planes are tilted 0.6 deg
    apart (a realistic separation for two objects in a crowded LEO shell).
    We compute exactly where their planes cross, then phase each object's
    starting position so both arrive near that crossing point at nearly
    the same time -- with a tiny (0.05 s) timing offset so it's a close
    call rather than an exact head-on hit.
    """
    i = np.radians(98.2)
    raan_a = np.radians(45.0)
    raan_b = np.radians(45.6)
    a1 = R_EARTH + 700.0
    a2 = R_EARTH + 700.2
    argp = 0.0

    crossing_dir = plane_crossing_direction(raan_a, i, raan_b, i)
    nu_a_cross = true_anomaly_for_direction(raan_a, i, argp, crossing_dir)
    nu_b_cross = true_anomaly_for_direction(raan_b, i, argp, crossing_dir)

    n_a = np.sqrt(MU_EARTH / a1**3)   # mean motion (rad/s)
    n_b = np.sqrt(MU_EARTH / a2**3)

    t_cross = 3000.0      # aim for the crossing near t = 3000 s
    timing_offset = 0.005  # seconds -- shifts B's arrival to create a near-miss

    nu_a0 = nu_a_cross - n_a * t_cross
    nu_b0 = nu_b_cross - n_b * (t_cross + timing_offset)

    r0_a, v0_a = kepler_to_state(a=a1, e=0.001, i=i, raan=raan_a, argp=argp, nu=nu_a0)
    r0_b, v0_b = kepler_to_state(a=a2, e=0.001, i=i, raan=raan_b, argp=argp, nu=nu_b0)

    return (r0_a, v0_a), (r0_b, v0_b), a1


def main():
    (r0_a, v0_a), (r0_b, v0_b), a1 = build_scenario()

    period = 2 * np.pi * np.sqrt(a1**3 / MU_EARTH)
    screen_t0, screen_t1 = 2500.0, 3500.0   # window to search for the crossing
    print(f"Orbital period ~ {period/60:.1f} min. Screening t in [{screen_t0}, {screen_t1}] s.\n")

    # IMPORTANT: initial states (r0_a, v0_a) etc. are defined at t=0 (the
    # epoch), so propagation must always start at t=0 -- solve_ivp's
    # t_span=(t0, t1) means "the initial state is y(t0)", so passing the
    # screening window's start here would silently mean "the epoch state
    # occurs at t=2500s", shifting every downstream time by 2500s. dense
    # output (sol.sol(t)) can still be sampled at any t in [0, t1].
    sol_a = propagate(r0_a, v0_a, (0.0, screen_t1))
    sol_b = propagate(r0_b, v0_b, (0.0, screen_t1))

    # --- Stage 1: coarse screening ---
    windows, times, ranges = ca.coarse_screen(sol_a, sol_b, screen_t0, screen_t1, dt=2.0, threshold_km=10.0)
    print(f"Coarse screen found {len(windows)} candidate window(s) under 10 km:")
    for w in windows:
        print(f"  t in [{w[0]:.1f}, {w[1]:.1f}] s")

    if not windows:
        print("No close approach found in this window -- adjust the scenario.")
        return

    # --- Stage 2: refine TCA for each candidate window ---
    print("\nRefined closest approaches:")
    events = []
    for (t_lo, t_hi) in windows:
        tca, miss = ca.refine_tca(sol_a, sol_b, t_lo, t_hi)
        events.append((tca, miss))
        print(f"  TCA = {tca:8.3f} s   miss distance = {miss*1000:8.2f} m")

    tca, miss = min(events, key=lambda e: e[1])

    # --- Stage 3: probability of collision at TCA ---
    dr, dv = ca.relative_state(sol_a, sol_b, tca)
    u, w = ca.encounter_plane_basis(dv)
    miss_2d = ca.project_to_plane(dr, u, w)

    # Example combined position covariance (1-sigma), projected into the
    # encounter plane, in km^2. In a real system this comes from each
    # object's tracked orbit-determination uncertainty.
    sigma_u_km = 0.050   # 50 m
    sigma_w_km = 0.020   # 20 m
    cov_2d = np.diag([sigma_u_km**2, sigma_w_km**2])

    hbr_km = 0.010  # combined hard-body radius, e.g. 5 m + 5 m = 10 m

    pc = ca.probability_of_collision(miss_2d, cov_2d, hbr_km)

    print(f"\nAt TCA (t = {tca:.3f} s):")
    print(f"  3D miss distance   : {miss*1000:.2f} m")
    print(f"  In-plane miss (u,w): ({miss_2d[0]*1000:.2f}, {miss_2d[1]*1000:.2f}) m")
    print(f"  Combined HBR       : {hbr_km*1000:.1f} m")
    print(f"  Probability of collision (Pc): {pc:.3e}")

    # Sensitivity check: Pc is extremely sensitive to how well the orbits
    # are known. Widen the uncertainty (e.g. an old/stale tracking update)
    # and watch Pc grow by orders of magnitude even though nothing about
    # the actual trajectories changed:
    cov_2d_wide = np.diag([(0.30)**2, (0.15)**2])  # 300 m / 150 m 1-sigma
    pc_wide = ca.probability_of_collision(miss_2d, cov_2d_wide, hbr_km)
    print(f"  ...with 6x larger position uncertainty, Pc becomes: {pc_wide:.3e}")
    print("  (this is why operators re-run Pc as fresh tracking data comes in)")

    # --- Plot range vs time ---
    fine_t = np.linspace(screen_t0, screen_t1, 3000)
    fine_r = [ca.relative_range(sol_a, sol_b, t) for t in fine_t]

    plt.figure(figsize=(9, 5))
    plt.plot(fine_t, fine_r, label="Range (dense)")
    plt.scatter(times, ranges, s=10, color="gray", alpha=0.5, label="Coarse screen samples")
    plt.axvline(tca, color="red", linestyle="--", label=f"TCA = {tca:.1f} s")
    plt.xlabel("Time since epoch (s)")
    plt.ylabel("Range (km)")
    plt.title("Relative range between Object A and Object B")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "..", "output", "range_plot.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()

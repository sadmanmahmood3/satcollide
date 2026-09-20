"""
Tests for the conjunction-assessment pipeline.

Run:  python3 -m pytest tests/ -v      (from the project root)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from orbits import (
    kepler_to_state, propagate, R_EARTH, MU_EARTH,
    plane_crossing_direction, true_anomaly_for_direction,
)
import conjunction as ca


def make_crossing_pair(timing_offset=0.005, t_cross=3000.0):
    """Same construction as the demo: two orbits timed to nearly collide."""
    i = np.radians(98.2)
    raan_a = np.radians(45.0)
    raan_b = np.radians(45.6)
    a1 = R_EARTH + 700.0
    a2 = R_EARTH + 700.2
    argp = 0.0

    d = plane_crossing_direction(raan_a, i, raan_b, i)
    nu_a_cross = true_anomaly_for_direction(raan_a, i, argp, d)
    nu_b_cross = true_anomaly_for_direction(raan_b, i, argp, d)

    n_a = np.sqrt(MU_EARTH / a1**3)
    n_b = np.sqrt(MU_EARTH / a2**3)

    nu_a0 = nu_a_cross - n_a * t_cross
    nu_b0 = nu_b_cross - n_b * (t_cross + timing_offset)

    r0_a, v0_a = kepler_to_state(a=a1, e=0.001, i=i, raan=raan_a, argp=argp, nu=nu_a0)
    r0_b, v0_b = kepler_to_state(a=a2, e=0.001, i=i, raan=raan_b, argp=argp, nu=nu_b0)
    return r0_a, v0_a, r0_b, v0_b


def test_kepler_round_trip_matches_vis_viva():
    """A circular orbit's speed should match sqrt(mu/r) (vis-viva at e=0)."""
    a = R_EARTH + 500.0
    r, v = kepler_to_state(a=a, e=0.0, i=np.radians(51.6), raan=0.3, argp=0.7, nu=1.1)
    assert np.isclose(np.linalg.norm(r), a, rtol=1e-10)
    assert np.isclose(np.linalg.norm(v), np.sqrt(MU_EARTH / a), rtol=1e-10)


def test_propagation_conserves_energy_without_j2():
    """Two-body (no J2) specific orbital energy must be conserved along the orbit."""
    a = R_EARTH + 550.0
    r0, v0 = kepler_to_state(a=a, e=0.01, i=np.radians(45.0), raan=0.1, argp=0.2, nu=0.0)
    sol = propagate(r0, v0, (0, 6000), j2=0.0)

    def energy(t):
        y = sol.sol(t)
        r, v = y[0:3], y[3:6]
        return 0.5 * np.dot(v, v) - MU_EARTH / np.linalg.norm(r)

    e0 = energy(0)
    for t in [1000, 3000, 5999]:
        assert np.isclose(energy(t), e0, rtol=1e-8)


def test_screening_and_refinement_find_the_true_minimum():
    """
    Cross-check the fast screen+refine pipeline against a brute-force dense
    grid search over the same window. They must agree on TCA (tight
    tolerance) and on miss distance (looser, since brute force is grid-limited).
    """
    r0_a, v0_a, r0_b, v0_b = make_crossing_pair()
    sol_a = propagate(r0_a, v0_a, (0.0, 3500.0))
    sol_b = propagate(r0_b, v0_b, (0.0, 3500.0))

    # Pipeline under test
    windows, _, _ = ca.coarse_screen(sol_a, sol_b, 2500.0, 3500.0, dt=2.0, threshold_km=10.0)
    assert len(windows) == 1
    tca, miss = ca.refine_tca(sol_a, sol_b, *windows[0])

    # Brute force: 1-second-resolution dense scan over the whole window
    brute_times = np.arange(2500.0, 3500.0, 1.0)
    brute_ranges = np.array([ca.relative_range(sol_a, sol_b, t) for t in brute_times])
    brute_tca = brute_times[np.argmin(brute_ranges)]
    brute_miss = brute_ranges.min()

    assert abs(tca - brute_tca) < 1.0          # within the brute-force grid step
    assert abs(miss - brute_miss) < 0.01        # within 10 m of the grid-limited value
    assert miss < brute_miss + 1e-6             # refined answer is at least as good


def test_coarse_screen_rejects_a_non_conjunction():
    """Two objects that never come close should yield zero candidate windows."""
    a = R_EARTH + 700.0
    r0_a, v0_a = kepler_to_state(a=a, e=0.001, i=np.radians(98.2), raan=0.0, argp=0.0, nu=0.0)
    # Same altitude, wildly different plane (90 deg different RAAN, opposite phasing)
    r0_b, v0_b = kepler_to_state(a=a, e=0.001, i=np.radians(53.0), raan=np.radians(200.0),
                                  argp=0.0, nu=np.radians(10.0))
    sol_a = propagate(r0_a, v0_a, (0.0, 1000.0))
    sol_b = propagate(r0_b, v0_b, (0.0, 1000.0))
    windows, _, _ = ca.coarse_screen(sol_a, sol_b, 0.0, 1000.0, dt=5.0, threshold_km=5.0)
    assert windows == []


def test_probability_of_collision_limits():
    """
    Sanity checks on the Pc integral:
    - A huge miss distance relative to sigma gives Pc ~ 0.
    - A zero miss distance with equal isotropic sigmas gives a Pc that
      matches the closed-form result for integrating a 2D Gaussian over
      a centered disk: Pc = 1 - exp(-HBR^2 / (2*sigma^2)).
    """
    cov = np.diag([0.05**2, 0.05**2])  # isotropic, 50 m sigma
    hbr = 0.01  # 10 m

    pc_far = ca.probability_of_collision(np.array([5.0, 0.0]), cov, hbr)
    assert pc_far < 1e-50

    pc_zero = ca.probability_of_collision(np.array([0.0, 0.0]), cov, hbr)
    sigma = 0.05
    expected = 1 - np.exp(-hbr**2 / (2 * sigma**2))
    assert np.isclose(pc_zero, expected, rtol=1e-6)


def test_probability_increases_as_covariance_grows_from_tight_to_moderate():
    """
    While the miss vector sits many sigma outside a *tight* covariance,
    growing that covariance raises Pc (the object's plausible position
    spreads toward the other object). Note this is NOT true everywhere:
    once sigma grows past roughly the miss distance itself, Pc turns over
    and starts falling again (the "Pc paradox" -- a very uncertain orbit
    can appear safer than a moderately uncertain one, because probability
    mass spreads out too thinly to concentrate over the hard-body disk).
    This test stays in the tight-to-moderate regime where growth holds.
    """
    miss = np.array([0.2, -0.03])  # 200 m / 30 m offset, matches the demo scenario
    hbr = 0.01
    pcs = []
    for scale in [0.02, 0.03, 0.04, 0.05, 0.06]:
        cov = np.diag([scale**2, (scale * 0.4)**2])
        pcs.append(ca.probability_of_collision(miss, cov, hbr))
    assert all(pcs[k] < pcs[k + 1] for k in range(len(pcs) - 1))

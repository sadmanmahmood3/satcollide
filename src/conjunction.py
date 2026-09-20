"""
conjunction.py
--------------
Conjunction assessment: given two propagated trajectories (dense ODE
solutions from orbits.propagate), find close approaches, refine the
time of closest approach (TCA), and estimate probability of collision (Pc).
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.integrate import dblquad


def relative_range(sol_a, sol_b, t):
    """Range between two objects (km) at time t, using dense ODE output."""
    ra = sol_a.sol(t)[0:3]
    rb = sol_b.sol(t)[0:3]
    return np.linalg.norm(ra - rb)


def coarse_screen(sol_a, sol_b, t0, t1, dt=10.0, threshold_km=50.0):
    """
    Step 1: cheap coarse scan over [t0, t1] at fixed dt (seconds).
    Returns a list of (t_lo, t_hi) bracket windows where range dropped
    under `threshold_km` -- candidates for fine refinement.

    This is the "screening" stage every real conjunction-assessment
    pipeline uses first, because propagating and refining every pair of
    the ~40,000 tracked objects at high precision is computationally
    infeasible: cheap filters reject the vast majority of pairs first.
    """
    times = np.arange(t0, t1 + dt, dt)
    ranges = np.array([relative_range(sol_a, sol_b, t) for t in times])

    windows = []
    in_window = False
    lo = None
    for i, r in enumerate(ranges):
        if r < threshold_km and not in_window:
            in_window = True
            lo = times[max(i - 1, 0)]
        elif r >= threshold_km and in_window:
            in_window = False
            windows.append((lo, times[i]))
    if in_window:
        windows.append((lo, times[-1]))
    return windows, times, ranges


def refine_tca(sol_a, sol_b, t_lo, t_hi):
    """
    Step 2: within a bracket window, find the precise time of closest
    approach (TCA) and miss distance using continuous optimization
    (Brent's method) on the dense ODE solution -- no fixed-step error.
    """
    result = minimize_scalar(
        lambda t: relative_range(sol_a, sol_b, t),
        bounds=(t_lo, t_hi), method="bounded",
        options={"xatol": 1e-6},
    )
    tca = result.x
    miss_distance = result.fun
    return tca, miss_distance


def relative_state(sol_a, sol_b, t):
    """Relative position (km) and velocity (km/s) of B with respect to A."""
    ya = sol_a.sol(t)
    yb = sol_b.sol(t)
    dr = yb[0:3] - ya[0:3]
    dv = yb[3:6] - ya[3:6]
    return dr, dv


def encounter_plane_basis(dv):
    """
    Build an orthonormal basis (u, w) spanning the plane perpendicular to
    the relative velocity vector at TCA (the "B-plane"). Collision
    probability is computed as a 2D problem in this plane, since to
    first order the miss distance along the relative-velocity direction
    doesn't matter -- only the offset transverse to it does.
    """
    dv_hat = dv / np.linalg.norm(dv)
    arbitrary = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(arbitrary, dv_hat)) > 0.9:
        arbitrary = np.array([0.0, 1.0, 0.0])
    u = np.cross(dv_hat, arbitrary)
    u /= np.linalg.norm(u)
    w = np.cross(dv_hat, u)
    return u, w


def project_to_plane(vec, u, w):
    return np.array([np.dot(vec, u), np.dot(vec, w)])


def probability_of_collision(miss_vec_2d, cov_2d, hbr_km, n_sigma_grid=8):
    """
    2D "Foster/Chan" style probability of collision.

    miss_vec_2d : (2,) offset of B relative to A in the encounter plane (km)
    cov_2d      : (2,2) combined position covariance projected into that
                  plane (km^2) -- i.e. covariance of A plus covariance of B
    hbr_km      : combined hard-body radius (sum of both objects' radii, km)

    Pc is the probability that the (Gaussian-distributed) relative position
    at TCA falls inside a disk of radius hbr_km centered on the origin,
    i.e. the two objects physically overlap. Computed by numerically
    integrating the bivariate Gaussian PDF over that disk.
    """
    # Diagonalize the covariance so the Gaussian is axis-aligned (uncorrelated)
    eigvals, eigvecs = np.linalg.eigh(cov_2d)
    sigma = np.sqrt(eigvals)  # (2,) standard deviations along principal axes
    # Rotate the miss vector into the same principal-axis frame
    miss_rot = eigvecs.T @ miss_vec_2d

    def pdf(y, x):
        return (1.0 / (2 * np.pi * sigma[0] * sigma[1])) * np.exp(
            -0.5 * (((x - miss_rot[0]) / sigma[0])**2 + ((y - miss_rot[1]) / sigma[1])**2)
        )

    # Integrate over the disk x^2 + y^2 <= hbr^2 using polar-style bounds
    def y_lo(x):
        return -np.sqrt(max(hbr_km**2 - x**2, 0.0))

    def y_hi(x):
        return np.sqrt(max(hbr_km**2 - x**2, 0.0))

    pc, _ = dblquad(pdf, -hbr_km, hbr_km, y_lo, y_hi, epsabs=1e-14, epsrel=1e-10)
    return pc

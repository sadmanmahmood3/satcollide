"""
orbits.py
---------
Minimal orbital mechanics core: Keplerian elements -> Cartesian state vectors,
and numerical propagation of the two-body problem with J2 perturbation.

Units: kilometers, seconds, radians (unless noted).
"""

import numpy as np
from scipy.integrate import solve_ivp

MU_EARTH = 398600.4418        # km^3/s^2
R_EARTH = 6378.137            # km (equatorial radius)
J2 = 1.08262668e-3            # Earth's J2 oblateness coefficient


def kepler_to_state(a, e, i, raan, argp, nu, mu=MU_EARTH):
    """
    Convert classical Keplerian elements to a Cartesian state vector (ECI frame).

    a    : semi-major axis (km)
    e    : eccentricity
    i    : inclination (rad)
    raan : right ascension of ascending node (rad)
    argp : argument of perigee (rad)
    nu   : true anomaly (rad)

    Returns (r, v): each a numpy array of shape (3,), km and km/s.
    """
    p = a * (1 - e**2)
    r_mag = p / (1 + e * np.cos(nu))

    # Position & velocity in the perifocal (PQW) frame
    r_pqw = r_mag * np.array([np.cos(nu), np.sin(nu), 0.0])
    v_pqw = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])

    # Rotation matrix PQW -> ECI (3-1-3 Euler sequence: RAAN, i, argp)
    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(i), np.sin(i)
    cw, sw = np.cos(argp), np.sin(argp)

    R = np.array([
        [cO*cw - sO*sw*ci, -cO*sw - sO*cw*ci,  sO*si],
        [sO*cw + cO*sw*ci, -sO*sw + cO*cw*ci, -cO*si],
        [sw*si,             cw*si,             ci   ],
    ])

    r_eci = R @ r_pqw
    v_eci = R @ v_pqw
    return r_eci, v_eci


def orbit_normal(raan, i):
    """Unit vector along the orbital angular-momentum direction (plane normal)."""
    return np.array([np.sin(raan) * np.sin(i), -np.cos(raan) * np.sin(i), np.cos(i)])


def rotation_pqw_to_eci(raan, i, argp):
    """The PQW -> ECI rotation matrix (same one used inside kepler_to_state)."""
    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(i), np.sin(i)
    cw, sw = np.cos(argp), np.sin(argp)
    return np.array([
        [cO*cw - sO*sw*ci, -cO*sw - sO*cw*ci,  sO*si],
        [sO*cw + cO*sw*ci, -sO*sw + cO*cw*ci, -cO*si],
        [sw*si,             cw*si,             ci   ],
    ])


def plane_crossing_direction(raan_a, i_a, raan_b, i_b):
    """
    Unit vector pointing at one of the two (antipodal) points where two
    orbital planes intersect. Useful for constructing conjunction
    scenarios: two objects in different, crossing orbital planes can
    only collide near one of these two points.
    """
    n_a = orbit_normal(raan_a, i_a)
    n_b = orbit_normal(raan_b, i_b)
    d = np.cross(n_a, n_b)
    return d / np.linalg.norm(d)


def true_anomaly_for_direction(raan, i, argp, direction):
    """
    For a circular (or near-circular) orbit, find the true anomaly nu at
    which the satellite's position points along `direction` (a unit
    vector). Used together with plane_crossing_direction to time two
    satellites to arrive at the same point in space.
    """
    R = rotation_pqw_to_eci(raan, i, argp)
    local = R.T @ direction
    return np.arctan2(local[1], local[0])


def _two_body_j2_accel(r, mu=MU_EARTH, j2=J2, r_earth=R_EARTH):
    """Acceleration from point-mass gravity + J2 oblateness term."""
    x, y, z = r
    r_norm = np.linalg.norm(r)
    r3 = r_norm**3

    a_two_body = -mu * r / r3

    # J2 perturbation (standard closed-form expression)
    factor = 1.5 * j2 * mu * r_earth**2 / r_norm**5
    z2_r2 = 5.0 * z**2 / r_norm**2
    a_j2 = factor * np.array([
        x * (z2_r2 - 1.0),
        y * (z2_r2 - 1.0),
        z * (z2_r2 - 3.0),
    ])
    return a_two_body + a_j2


def _rhs(t, y, mu, j2, r_earth):
    r = y[0:3]
    v = y[3:6]
    a = _two_body_j2_accel(r, mu, j2, r_earth)
    return np.hstack([v, a])


def propagate(r0, v0, t_span, t_eval=None, mu=MU_EARTH, j2=J2, r_earth=R_EARTH,
              rtol=1e-10, atol=1e-10):
    """
    Numerically propagate a state vector under two-body + J2 dynamics.

    r0, v0  : initial position (km) and velocity (km/s), shape (3,)
    t_span  : (t0, tf) in seconds
    t_eval  : optional array of times (s) at which to sample the solution

    Returns the scipy OdeSolution-like result object (has .sol for dense
    output, .t and .y for the sampled grid).
    """
    y0 = np.hstack([r0, v0])
    sol = solve_ivp(
        _rhs, t_span, y0, t_eval=t_eval, args=(mu, j2, r_earth),
        method="DOP853", rtol=rtol, atol=atol, dense_output=True,
    )
    if not sol.success:
        raise RuntimeError(f"Propagation failed: {sol.message}")
    return sol

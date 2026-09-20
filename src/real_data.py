"""
real_data.py
------------
Bolt real tracked-object data onto the conjunction pipeline using TLEs
and the SGP4 propagator.

NOT importable in this sandbox (no internet access here to reach PyPI or
CelesTrak), but this is standard, well-documented usage of the `sgp4` and
`requests` libraries. On your own machine:

    pip install sgp4 requests

Then run this file directly, or import `propagate_tle` / `fetch_tles`
into your own scripts.
"""

import numpy as np

try:
    from sgp4.api import Satrec, jday
except ImportError:
    Satrec = None  # only needed once you actually call the functions below


def fetch_tles(group="active", fmt="tle"):
    """
    Download current TLEs for a named CelesTrak group, e.g. "active",
    "stations", "starlink", "cosmos-1408-debris", "iridium-33-debris".
    See https://celestrak.org/NORAD/elements/ for the full list of groups.

    Returns a list of (name, line1, line2) tuples.
    """
    import requests
    url = f"https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT={fmt}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    lines = [l.strip() for l in resp.text.strip().splitlines() if l.strip()]

    sats = []
    for i in range(0, len(lines), 3):
        name, l1, l2 = lines[i], lines[i + 1], lines[i + 2]
        sats.append((name, l1, l2))
    return sats


def make_satrec(line1, line2):
    """Build an sgp4 Satrec object from two TLE lines."""
    if Satrec is None:
        raise ImportError("pip install sgp4")
    return Satrec.twoline2rv(line1, line2)


def propagate_tle(satrec, times_utc):
    """
    Propagate a Satrec across a list/array of Python datetime objects (UTC).
    Returns (positions_km, velocities_km_s), each shape (N, 3), in the
    TEME frame (SGP4's native output frame -- fine for relative-distance
    conjunction screening between two TEME-propagated objects; convert to
    a common frame like ECI/J2000 first if you're mixing with an ephemeris
    that is *not* also SGP4-derived).
    """
    jd = np.zeros(len(times_utc))
    fr = np.zeros(len(times_utc))
    for k, t in enumerate(times_utc):
        jd[k], fr[k] = jday(t.year, t.month, t.day, t.hour, t.minute,
                             t.second + t.microsecond * 1e-6)

    error_codes, positions, velocities = satrec.sgp4_array(jd, fr)
    if np.any(error_codes != 0):
        bad = np.nonzero(error_codes)[0]
        raise RuntimeError(f"SGP4 propagation error at {len(bad)} time(s), "
                            f"first code={error_codes[bad[0]]} (see sgp4 docs)")
    return positions, velocities


def find_close_pairs_over_catalog(sats, start_utc, minutes=1440, step_seconds=60,
                                   distance_threshold_km=25.0):
    """
    Example of catalog-scale coarse screening: for N tracked objects this
    is O(N^2) pairs, so real systems narrow the candidate set first (e.g.
    by orbital-regime / altitude band, or an "orbital element filter" like
    comparing perigee/apogee ranges) before running this. For a few dozen
    to a few hundred objects, brute-force sampling like below is fine.

    sats        : list of (name, line1, line2)
    start_utc   : datetime.datetime (UTC) to start the scan
    minutes     : how many minutes forward to scan
    step_seconds: coarse sampling step

    Returns a list of dicts: {name_a, name_b, t_index, range_km}
    for every sample where two objects were under distance_threshold_km.
    Feed these into the fine TCA-refinement + Pc code in conjunction.py.
    """
    from datetime import timedelta

    n_steps = int(minutes * 60 / step_seconds)
    times = [start_utc + timedelta(seconds=step_seconds * k) for k in range(n_steps)]

    satrecs = [make_satrec(l1, l2) for (_, l1, l2) in sats]
    all_positions = []
    for sr in satrecs:
        pos, _vel = propagate_tle(sr, times)
        all_positions.append(pos)  # shape (n_steps, 3)
    all_positions = np.array(all_positions)  # (n_sats, n_steps, 3)

    hits = []
    n_sats = len(sats)
    for a in range(n_sats):
        for b in range(a + 1, n_sats):
            diff = all_positions[a] - all_positions[b]
            ranges = np.linalg.norm(diff, axis=1)
            idx = np.nonzero(ranges < distance_threshold_km)[0]
            for i in idx:
                hits.append({
                    "name_a": sats[a][0], "name_b": sats[b][0],
                    "t_index": i, "time_utc": times[i],
                    "range_km": ranges[i],
                })
    return hits


if __name__ == "__main__":
    import datetime
    print("Fetching active satellite TLEs from CelesTrak...")
    sats = fetch_tles(group="stations")  # small group for a quick demo
    print(f"Got {len(sats)} objects.")

    hits = find_close_pairs_over_catalog(
        sats, start_utc=datetime.datetime.utcnow(),
        minutes=180, step_seconds=30, distance_threshold_km=100.0,
    )
    print(f"Found {len(hits)} coarse-screen hits under 100 km in the next 3 hours.")
    for h in hits[:10]:
        print(h)

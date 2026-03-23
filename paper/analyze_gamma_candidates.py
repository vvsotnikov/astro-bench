"""Analyze gamma-ray candidates from run_real_gamma_search.py.

Converts (Ze, Az, timestamp) → equatorial → galactic coordinates.
Checks angular distance to known PeVatron candidates.
Generates skymap figure.

Requires: gamma_candidates.npz (from run_real_gamma_search.py)
          + re-scanning real data for timestamps from extra_features.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import astropy.units as u
import datetime
from glob import glob
from pathlib import Path

KASCADE = EarthLocation(lon=8.4 * u.deg, lat=49.1 * u.deg, height=110 * u.m)
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "real_kascade"

# Known PeVatron candidates / bright TeV sources
KNOWN_SOURCES = [
    ('Crab Nebula', 83.633, 22.015),
    ('Cygnus OB2', 308.08, 41.31),
    ('Cas A', 350.85, 58.82),
    ('LHAASO J2108+5157', 317.15, 51.95),
    ('MGRO J1908+06', 287.05, 6.27),
]


def find_event_timestamps(target_events):
    """Scan real data to find timestamps for events identified by (E, Ze, Az).
    target_events: list of dicts with 'E', 'Ze', 'Az' keys.
    Returns list of dicts with added 'time' key.
    """
    results = []
    for ev in target_events:
        results.append({**ev, 'time': None})

    for mpath in sorted(glob(str(DATA_DIR / '*_matrices.npz'))):
        run_name = Path(mpath).name.replace('_matrices.npz', '')
        fpath = DATA_DIR / f'{run_name}_features.npz'
        efpath = DATA_DIR / f'{run_name}_extra_features.npz'

        feat = np.load(fpath)['features']
        if feat.ndim != 2 or len(feat) == 0:
            continue

        for ri, ev in enumerate(target_events):
            if results[ri]['time'] is not None:
                continue
            for i in range(len(feat)):
                e, xc, yc, ze, az, ne, nmu, age = feat[i]
                if (abs(e - ev['E']) < 0.001 and abs(ze - ev['Ze']) < 0.1
                        and abs(az - ev['Az']) < 0.1):
                    ef = np.load(efpath, allow_pickle=True)['extra_features']
                    results[ri]['time'] = ef[i][8]  # datetime column
                    results[ri]['run'] = run_name
                    print(f"  Found {ev.get('name', '?')} in {run_name}: {results[ri]['time']}")
                    break

        if all(r['time'] is not None for r in results):
            break

    return results


def analyze_and_plot(events_with_times):
    """Compute sky coordinates, check sources, generate skymap."""
    ev_coords = []
    for ev in events_with_times:
        coord = SkyCoord(
            alt=(90 - ev['Ze']) * u.deg, az=ev['Az'] * u.deg,
            obstime=Time(ev['time']), frame='altaz', location=KASCADE
        ).transform_to('icrs')
        ev_coords.append(coord)

        gal = coord.galactic
        print(f"\n{ev.get('name', 'Event')}: {ev['time']}")
        print(f"  E = 10^{ev['E']:.3f} eV = {10**ev['E']:.2e} eV")
        print(f"  Ze = {ev['Ze']:.1f} deg, Az = {ev['Az']:.1f} deg")
        print(f"  Ne/Nmu = {10**(ev['Ne']-ev['Nmu']):.0f}x")
        print(f"  Equatorial: RA = {coord.ra.deg:.3f} deg ({coord.ra.to_string(u.hour, precision=1)}), "
              f"Dec = {coord.dec.deg:.3f} deg")
        print(f"  Galactic:   l = {gal.l.deg:.3f} deg, b = {gal.b.deg:.3f} deg")

        print(f"  Angular distances to known sources:")
        for name, ra, dec in KNOWN_SOURCES:
            src = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
            sep = coord.separation(src)
            print(f"    {name:30s}: {sep.deg:.1f} deg")

    # Skymap
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='aitoff')

    gal_l = np.linspace(-180, 180, 360)
    ax.plot(np.radians(gal_l), np.zeros_like(gal_l), 'k-', alpha=0.15, linewidth=8)

    for name, ra, dec in KNOWN_SOURCES:
        src = SkyCoord(ra=ra * u.deg, dec=dec * u.deg).galactic
        l = src.l.wrap_at('180d').radian
        b = src.b.radian
        ax.scatter(l, b, c='gray', s=40, marker='D', zorder=3, alpha=0.7)
        ax.annotate(name, (l, b), fontsize=6, color='gray',
                    xytext=(5, 5), textcoords='offset points')

    colors = ['#E53935', '#1E88E5', '#4CAF50', '#FF9800']
    for i, (ev, coord) in enumerate(zip(events_with_times, ev_coords)):
        gal = coord.galactic
        l = gal.l.wrap_at('180d').radian
        b = gal.b.radian
        ax.scatter(l, b, c=colors[i % len(colors)], s=120, marker='*',
                   zorder=5, edgecolors='black', linewidths=0.5)
        ax.annotate(ev.get('name', f'Event {i+1}'), (l, b), fontsize=7,
                    color=colors[i % len(colors)],
                    xytext=(8, -12), textcoords='offset points', fontweight='bold')

    ax.set_title('Gamma-ray candidates from AI classifier on real KASCADE data', fontsize=11)
    ax.set_xlabel('Galactic longitude')
    ax.set_ylabel('Galactic latitude')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path(__file__).resolve().parent / 'fig_gamma_candidates_skymap.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f"\nSaved {out}")


if __name__ == '__main__':
    # Events from Sonnet v4 classifier (E=16.0-16.5 bin)
    target_events = [
        {'name': 'Event 1 (13 PeV, 1999)', 'E': 16.114, 'Ze': 16.90, 'Az': 2.99,
         'Ne': 6.905, 'Nmu': 4.260},
        {'name': 'Event 2 (10.5 PeV, 2003)', 'E': 16.020, 'Ze': 16.13, 'Az': 280.57,
         'Ne': 6.749, 'Nmu': 4.284},
    ]

    print("Scanning real data for event timestamps...")
    events = find_event_timestamps(target_events)

    missing = [e for e in events if e['time'] is None]
    if missing:
        print(f"WARNING: Could not find timestamps for {len(missing)} events")

    analyze_and_plot([e for e in events if e['time'] is not None])

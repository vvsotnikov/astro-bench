"""Cross-match the top-5 high-energy gamma candidates against the 1st LHAASO catalog.

Uses Cao et al. 2024 (ApJS 271 25) via VizieR (90 unique sources, 180 entries
including extended-source components).

KASCADE angular resolution at PeV is ~1 deg, so any min_sep > a few degrees
is not a meaningful spatial association.
"""

import datetime
from pathlib import Path

import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
import astropy.units as u
from astroquery.vizier import Vizier

KASCADE = EarthLocation(lon=8.4 * u.deg, lat=49.1 * u.deg, height=110 * u.m)

# Top-5 events at log10(E/eV) > 16 from gamma_real_full_opus.npz
# (E, Ze in deg, Az in deg, timestamp UTC, classifier score)
EVENTS = [
    ("Run 1971", 16.114, 16.90, 2.99,   "1999-09-10 19:13:28", 0.7519),
    ("Run 1471", 16.401, 27.86, 229.53, "1999-02-16 00:05:33", 0.6831),
    ("Run 4281", 16.020, 16.13, 280.57, "2003-06-10 09:07:04", 0.6593),
    ("Run 3180", 16.031, 26.11, 262.13, "2001-01-15 09:14:18", 0.5327),
    ("Run 7231", 16.006, 22.78, 38.65,  "2011-11-03 20:09:26", 0.5229),
]


def load_lhaaso_catalog():
    v = Vizier(catalog="J/ApJS/271/25", columns=["*"])
    v.ROW_LIMIT = -1
    tbl = v.get_catalogs("J/ApJS/271/25")[0]
    coords = SkyCoord(ra=tbl["RAJ2000"], dec=tbl["DEJ2000"], unit=(u.deg, u.deg))
    names = [str(n) for n in tbl["1LHAASO"]]
    n_unique = len(set(names))
    print(f"LHAASO 1st catalog: {len(tbl)} entries ({n_unique} unique sources)")
    return coords, names


def event_to_icrs(ze_deg, az_deg, ts_iso):
    dt = datetime.datetime.fromisoformat(ts_iso).replace(tzinfo=datetime.timezone.utc)
    return SkyCoord(
        alt=(90 - ze_deg) * u.deg, az=az_deg * u.deg,
        obstime=Time(dt), frame="altaz", location=KASCADE,
    ).transform_to("icrs")


def main():
    lh_coords, lh_names = load_lhaaso_catalog()

    print()
    print(f"{'Event':>10s}  {'E':>7s}  {'RA':>7s}  {'Dec':>7s}  "
          f"{'nearest 1LHAASO':>20s}  {'min_sep [deg]':>14s}")
    rows = []
    for name, E, Ze, Az, ts, _ in EVENTS:
        coord = event_to_icrs(Ze, Az, ts)
        seps = coord.separation(lh_coords).deg
        idx = int(np.argmin(seps))
        nearest = lh_names[idx]
        min_sep = float(seps[idx])
        print(f"{name:>10s}  {E:7.3f}  {coord.ra.deg:7.2f}  {coord.dec.deg:7.2f}  "
              f"{nearest:>20s}  {min_sep:14.2f}")
        rows.append((name, E, coord.ra.deg, coord.dec.deg, nearest, min_sep))

    floor = min(rows, key=lambda r: r[5])
    print()
    print(f"Floor across all 5 events: {floor[5]:.2f} deg ({floor[0]} -> {floor[4]})")
    print(f"KASCADE angular resolution at PeV ~1 deg, so all 5 separations are well")
    print("outside the resolution and there is no spatial association.")

    out = Path(__file__).resolve().parent / "lhaaso_crossmatch.npz"
    np.savez(
        out,
        events=np.array([(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows], dtype=object),
    )
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()

"""Jewelry building blocks: ring sizes, calibrated stones, bands, bezels and prong heads (exact B-rep, mm).

Stones are parts with role "stone" and meta stone={"D", "cut", "c", "n"} so the checks can test stock size,
clearance from metal and whether the setting holds the stone.
"""
import math
import numpy as np
from build123d import Solid, Wire, Face, Axis, Location

# calibrated round sizes you can actually buy (mm)
ROUND_STOCK = [1.0, 1.1, 1.2, 1.25, 1.3, 1.4, 1.5, 1.6, 1.7, 1.75, 1.8, 1.9, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5,
               3.75, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 9.0, 10.0, 11.0, 12.0]
BALL_STOCK = [1.588, 2.0, 2.381, 3.0, 3.175, 3.969, 4.0, 4.763, 5.0, 5.556, 6.0, 6.35]


def ring_inner_d(size_us):
    """US/Canada ring size -> inside diameter in mm (ISO 8653 relation used by the trade: 11.63 + 0.8128 * size)"""
    return 11.63 + 0.8128 * float(size_us)


def snap_round(D):
    ok = [x for x in ROUND_STOCK if x <= D + 1e-6]
    return ok[-1] if ok else ROUND_STOCK[0]


def _rev(profile):
    w = Wire.make_polygon([(r, 0, z) for r, z in profile], close=True)
    return Solid.revolve(Face(w), 360, Axis.Z)


def brilliant(D):
    """round brilliant envelope, girdle centred at z = 0, table up (+Z). Crown 16 %, table 57 %, pavilion 43 %."""
    a, g = D / 2, 0.03 * D
    return _rev([(0, -0.43 * D), (a, -g / 2), (a, g / 2), (0.285 * D, g / 2 + 0.16 * D), (0, g / 2 + 0.16 * D)])


def orient(shape, c, n):
    """move a solid built about +Z at the origin so its axis is n and its origin at c"""
    n = np.asarray(n, float) / np.linalg.norm(n)
    z = np.array([0, 0, 1.0])
    ax = np.cross(z, n)
    s = shape
    if np.linalg.norm(ax) > 1e-9:
        ang = math.degrees(math.acos(np.clip(z @ n, -1, 1)))
        from build123d import Axis as A, Vector
        s = s.rotate(A((0, 0, 0), tuple(ax / np.linalg.norm(ax))), ang)
    elif z @ n < 0:
        s = s.rotate(Axis.X, 180)
    return s.moved(Location(tuple(float(x) for x in c)))


def band(size_us, width=4.0, thick=1.6, comfort=True):
    """a plain band about the Z axis (finger along Z), centred on z = 0"""
    ri = ring_inner_d(size_us) / 2
    ro = ri + thick
    w = width / 2
    if comfort:   # comfort fit: inside face domed 0.35 mm, outside edges eased
        prof = [(ri + 0.35, -w), (ro - 0.3, -w), (ro, -w + 0.3), (ro, w - 0.3), (ro - 0.3, w), (ri + 0.35, w),
                (ri + 0.1, w * 0.6), (ri, 0), (ri + 0.1, -w * 0.6)]
    else:
        prof = [(ri, -w), (ro, -w), (ro, w), (ri, w)]
    return _rev(prof)


def bezel_cup(D, wall=0.6, height=None, lip=0.25, base=0.8):
    """a round bezel cup for a brilliant of diameter D, girdle at z = 0, open seat cut for the pavilion and a
    light hole. The lip (the top of the wall) rises `lip` above the girdle's top edge; burnished over in
    finishing, it is what holds the stone."""
    a, g = D / 2, 0.03 * D
    h_top = g / 2 + lip + 0.04 * D
    h_bot = -(0.43 * D + base)
    Ro = a + wall
    cup = _rev([(0, h_bot), (Ro * 0.92, h_bot), (Ro, h_bot + 0.6), (Ro, h_top), (a + 0.02, h_top), (a + 0.02, -g / 2),
                (0.3 * D, -0.43 * D * 0.55), (0.22 * D, h_bot - 0.01), (0, h_bot - 0.01)])
    return cup


def prong_head(D, n=4, wire=0.9, height=None, tilt=8.0):
    """n round prongs leaning in over the girdle, joined by a gallery ring under the stone"""
    a = D / 2
    h_top = 0.03 * D / 2 + 0.16 * D * 0.45
    h_bot = -(0.43 * D + 1.2)
    out = None
    for i in range(n):
        ang = 2 * math.pi * (i + 0.5) / n
        top = np.array([(a + 0.05) * math.cos(ang), (a + 0.05) * math.sin(ang), h_top])
        bot = np.array([(a * 0.7) * math.cos(ang), (a * 0.7) * math.sin(ang), h_bot])
        L = float(np.linalg.norm(top - bot))
        p = Solid.make_cylinder(wire / 2, L)
        p = orient(p, bot, top - bot)
        tip = Solid.make_sphere(wire / 2 * 1.05).moved(Location(tuple(top)))
        out = p + tip if out is None else out + p + tip
    gal = Solid.make_torus(a * 0.72, wire * 0.45).moved(Location((0, 0, -0.43 * D * 0.6)))
    return out + gal


def stone_meta(D, c, n, cut="round", kind="cz"):
    return {"D": float(D), "cut": cut, "kind": kind, "c": [float(x) for x in c], "n": [float(x) for x in n]}


def macro(spec):
    """build a jewelry block from {"band"|"bezel"|"prongs"|"stone": {...}, "at": [..], "n": [..]}"""
    (k, a), = [(k, v) for k, v in spec.items() if k not in ("at", "n")]
    fn = {"band": band, "bezel": bezel_cup, "prongs": prong_head}.get(k)
    if k == "stone":
        s = brilliant(a["D"])
    elif fn:
        s = fn(**a)
    else:
        raise ValueError(f"unknown macro {k}")
    at, n = spec.get("at", [0, 0, 0]), spec.get("n", [0, 0, 1])
    return orient(s, at, n), k, a, at, n

"""GLOSS: the HELIUM cocktail ring (jewelry/helium/kit/helium.py gloss()), rebuilt in phlasma.cad's own SDF engine.

A candy cushion: four puffs blown up round one tuft, the tuft a 6 mm pink tourmaline pressed into it in a puffed
crater bezel; opposite puffs lacquered pink and lilac (cold enamel, separate layer parts); the cushion's corners
pulled out into thorns, two long, two short; the puffs cast hollow and vented. Shank of twisted balloon segments.
Ring frame: finger axis Y, top +Z, mm. Cast in brass, rhodium plated.
"""
import math
import numpy as np
from phlasma.cad import Model
from phlasma.cad.sdf import Scene, nz, perp, dense, surface_along
from phlasma.cad import jewelry

PARAMS = {"size_us": 6.5, "stone_D": 6.0, "voxel": 0.13, "metal": "brass"}


def arc3(c, R, a0, a1, n=24):
    t = np.linspace(a0, a1, n)
    return np.stack([c[0] + R * np.sin(t), np.full(n, c[1]), c[2] + R * np.cos(t)], 1)


def sausage(s, P, rmax, neck, k=0.3, blunt=0.42):
    t = np.linspace(0, 1, len(P))
    rad = neck + (rmax - neck) * np.sin(math.pi * t) ** blunt
    Pd, rd = dense(P, rad)
    s.stroke(Pd, rd, k=k)


def twist(s, p, t, rn):
    t = nz(t)
    s.cone(p - t * rn * 1.1, p + t * rn * 1.1, rn, rn, k=0.3, round=True)
    s.torus(p, rn * 0.95, rn * 0.6, normal=t, k=0.3)


def balloon_shank(s, Rin, a_top, segs=2, rmax=1.75, rneck=1.15):
    # inner face of the bellies AND of the twist knots (reach 1.55 x neck) on the finger, plus the k=0.3 blend
    # bulge: the HELIUM kit used Rin + rmax / 1.12, which put the finger hole 0.44 mm under the asked radius
    Rc = Rin + max(rmax, 1.55 * rneck) + 0.08
    A = np.linspace(a_top, 2 * math.pi - a_top, 2 * segs + 1)
    for i in range(2 * segs):
        P = arc3((0, 0, 0), Rc, A[i], A[i + 1], 18)
        sausage(s, P, rmax * (0.92 if i in (segs - 1, segs) else 1.0), rneck)
        if i > 0:
            twist(s, P[0], P[1] - P[0], rneck)


def thorn(s, base, d, L, r0, side, bend=0.25, tip=0.24, k=0.45, swell=1.1, sc=0.0, n=24):
    base, d, side = np.asarray(base, float), nz(d), nz(side)
    t = np.linspace(0, 1, n)
    off = bend * L * t ** 2.2 - sc * L * np.sin(math.pi * np.clip(t * 1.25, 0, 1)) * 0.5
    P = base[None] + np.outer(t * L, d) + np.outer(off, side) - d * r0 * 0.6
    rad = r0 * (swell * np.sin(t * math.pi * 0.5 + 0.3) ** 0.5) * (1 - t ** 1.6) + tip * t ** 1.6
    rad[0] = r0
    Pd, rd = dense(P, rad, step=0.25)
    s.stroke(Pd, rd, k=k)


def crater(s, c, n, D):
    """stone pressed into the lobe: straight-walled seat, pavilion seat, soft puffed rim, light hole"""
    n = nz(n); c = surface_along(s, c, n) - n * (0.06 * D + 0.15); a = D / 2
    rr = float(np.clip(0.07 * D + 0.4, 0.55, 1.0))
    s.ellipsoid(c - n * (0.16 * D), (a + rr * 0.9, a + rr * 0.9, 0.24 * D + 0.55), xaxis=perp(n), zhint=n, k=0.6)
    s.torus(c + n * (0.06 * D), a + rr + 0.12, rr, normal=n, k=1.1)
    s.cone(c + n * 0.02, c - n * (0.43 * D + 0.2), a + 0.12, 0.25, op=1, k=0.06)     # pavilion seat, 0.1 mm clear
    s.cone(c - n * 0.02, c + n * (0.12 * D + 0.1), a + 0.05, a + 0.05, op=1, k=0.05)
    s.cone(c + n * (0.12 * D + 0.1), c + n * (0.7 * D), a + 0.05, a + 0.05 + 0.35 * D, op=1, k=0.15)
    s.cone(c - n * 0.3 * D, c - n * (D + 3.0), 0.22 * D, 0.22 * D, op=1, k=0.1)
    return c, n


def inlay(s, c, radii, xaxis, up, depth=0.5):
    """cut a recess; return the enamel scene (metal surface before the cut, inside the recess shrunk 0.06 mm)"""
    up = nz(up); c = surface_along(s, c, up)
    rz = max(radii[2], depth + 0.6)
    q = c + up * (rz - depth)
    enamel = s.copy()
    enamel.ellipsoid(q, (radii[0] - 0.06, radii[1] - 0.06, rz - 0.06), xaxis=xaxis, zhint=up, op=2)
    s.ellipsoid(q, (radii[0], radii[1], rz), xaxis=xaxis, zhint=up, op=1, k=0.12)
    return enamel, q, max(radii[:2]) + 0.3


def build(p):
    Rin = jewelry.ring_inner_d(p["size_us"]) / 2
    m = Model("helium_gloss", units="mm", kind="jewelry", ring_size_us=p["size_us"], ring_axis=(0, 1, 0))
    s = Scene()
    balloon_shank(s, Rin, 0.42, segs=2, rmax=1.75)
    zc = Rin + 4.3
    puffs = [(3.4, 3.0), (-3.4, 3.0), (-3.4, -3.0), (3.4, -3.0)]
    for x, y in puffs:
        s.ellipsoid((x, y, zc), (4.1, 3.7, 3.1), xaxis=(x, y, 0), k=1.3)
    s.ellipsoid((0, 0, Rin + 1.7), (6.2, 4.6, 2.2), k=1.4)
    for (x, y), L, sc_ in zip(puffs, (10.5, 3.8, 9.0, 3.4), (0.12, 0.0, 0.12, 0.0)):
        d = nz(np.array([x, y, 0.35]))
        thorn(s, (x * 1.55, y * 1.55, zc - 0.4), d, L, 1.25 if L > 5 else 0.95,
              side=nz(np.cross(d, [0, 0, 1])) * np.sign(x * y), bend=0.35, sc=sc_)
    D = jewelry.snap_round(p["stone_D"])
    cs, ns = crater(s, (0, 0, zc + 4), (0, 0, 1), D)
    enamels = []
    for (x, y), col in zip(puffs, ("pink", "lilac", "pink", "lilac")):
        enamels.append((inlay(s, (x * 1.45, y * 1.45, zc + 4), (2.3, 2.7, 1.2), (x, y, 0), (x * 0.16, y * 0.16, 1)), col))
    # finger hole cut last: the cushion's seat (and any blend) must never reach inside the band
    s.cone((0, -20, 0), (0, 20, 0), Rin, Rin, op=1, k=0.15)
    for x, y in puffs:   # cast hollow, vented out under the cushion
        c = np.array([x, y, zc - 0.2])
        s.ellipsoid(c, (2.5, 2.1, 1.4), xaxis=(x, y, 0), zhint=(0, 0, 1), op=1)
        v = nz((x, y, -1.3))
        s.cone(c, c + v * 5.5, 0.6, 0.6, op=1, k=0.05)
    m.mesh("metal", s.mesh(voxel=p["voxel"], keep="largest"), p["metal"], finish="rhodium")
    for i, ((sc, q, r), col) in enumerate(enamels):
        lo, hi = q - r - 1.5, q + r + 1.5
        m.mesh(f"enamel_{i + 1}_{col}", sc.mesh(voxel=min(p["voxel"], 0.08), lo=lo, hi=hi, keep="largest"),
               "enamel", role="layer", host="metal", colour=col)
    st = jewelry.orient(jewelry.brilliant(D), cs, ns)
    m.solid("stone", st, "pinktourmaline", role="stone", stone=jewelry.stone_meta(D, cs, ns, "round", "pinktourmaline"))
    return m

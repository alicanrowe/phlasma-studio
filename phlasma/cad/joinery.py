"""Solid-wood joinery for exact (brep) parts: sticks as axis-aligned boxes, mortise-and-tenon joints between them.

    leg = stick(m, "Leg_FL", (0, 0, 0), (1.75, 1.75, 17), "white_oak", grain="z")
    rail = stick(m, "Apron_F", (1.75, 0.125, 13), (12.25, 0.875, 17), "white_oak", grain="x")
    mortise_tenon(m, "Apron_F", "Leg_FL", end="-x", length=0.75, thick=0.375, shoulders=(0.25, 0.5))

Every joint is recorded in model.meta["joints"] and checked by checks.joinery() against the house rules of the
hardwood-furniture-cad skill (in inches, converted for metric models):
  mortise walls (wood each side of a mortise)      >= 1/4 in
  web between two mortises in one stick            >= 1/4 in
  blind mortise to the stick's far face            >= 1/8 in
  mortise from the end of a leg or stile           >= 1/2 in (fill the gap with a haunch, not a thin end)
  tenon fills its mortise: same height, fit clearance only, bottom clearance <= 1/16 in
"""
import numpy as np
from build123d import Box, Location

IN = 25.4
AX = {"x": 0, "y": 1, "z": 2}


def _box(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c, s = (lo + hi) / 2, hi - lo
    return Box(*s).moved(Location(tuple(c)))


def stick(m, name, lo, hi, material="white_oak", grain=None, **meta):
    """an axis-aligned square-stock part; grain defaults to its longest side"""
    lo, hi = np.minimum(lo, hi).astype(float), np.maximum(lo, hi).astype(float)
    g = grain or "xyz"[int(np.argmax(hi - lo))]
    p = m.solid(name, _box(lo, hi), material, grain=g, stock=[lo.tolist(), hi.tolist()], **meta)
    return p


def _set(p, shape):
    p.shape = shape
    p._mesh = None


def mortise_tenon(m, rail, post, end, length, thick, shoulders=(0.25, 0.25), thick_axis=None, offset=0.0,
                  fit=None, bottom=None, haunch=False):
    """Tenon on `rail`'s `end` ("+x", "-y", ...: the end that meets `post`) running `length` into `post`, and its
    mortise cut from `post`. thick = tenon thickness across `thick_axis` (default: the rail's thin side);
    shoulders = (low, high) shoulders along the remaining axis; offset moves the tenon across its thickness
    (+ toward the axis' positive side; bias tenons toward the outside of a piece). fit = total side clearance
    (default 0.004 in), bottom = extra mortise depth (default 1/32 in). Units are the model's."""
    unit_in = 1.0 if m.units == "in" else IN / m.mm      # one inch in model units
    fit = 0.004 * unit_in if fit is None else fit
    bottom = unit_in / 32 if bottom is None else bottom
    R, P = m.get(rail), m.get(post)
    rlo, rhi = map(np.array, R.meta["stock"])
    plo, phi = map(np.array, P.meta["stock"])
    sign, a = (1 if end[0] == "+" else -1), AX[end[1]]
    others = [i for i in range(3) if i != a]
    ext = (rhi - rlo)
    ta = AX[thick_axis] if thick_axis else min(others, key=lambda i: ext[i])
    wa = [i for i in others if i != ta][0]
    face = rhi[a] if sign > 0 else rlo[a]
    tlo, thi = np.zeros(3), np.zeros(3)
    tc = (rlo[ta] + rhi[ta]) / 2 + offset
    tlo[ta], thi[ta] = tc - thick / 2, tc + thick / 2
    tlo[wa], thi[wa] = rlo[wa] + shoulders[0], rhi[wa] - shoulders[1]
    tlo[a], thi[a] = (face, face + length) if sign > 0 else (face - length, face)
    mlo, mhi = tlo.copy(), thi.copy()
    mlo[ta] -= fit / 2; mhi[ta] += fit / 2
    mlo[wa] -= fit / 2; mhi[wa] += fit / 2
    if sign > 0:
        mhi[a] += bottom
    else:
        mlo[a] -= bottom
    _set(R, R.shape + _box(tlo, thi))
    _set(P, P.shape - _box(mlo, mhi))
    j = {"rail": rail, "post": post, "end": end, "tenon": [tlo.tolist(), thi.tolist()],
         "mortise": [mlo.tolist(), mhi.tolist()], "axis": a, "sign": sign, "haunch": bool(haunch)}
    m.meta.setdefault("joints", []).append(j)
    return j


def box_gap(a, b):
    """distance between two axis-aligned boxes (0 when they touch or overlap)"""
    g = np.maximum(0, np.maximum(np.asarray(b[0]) - a[1], np.asarray(a[0]) - b[1]))
    return float(np.linalg.norm(g))


def check(m):
    unit_in = 1.0 if m.units == "in" else IN / m.mm
    W, WEB, FAR, END = 0.25 * unit_in, 0.25 * unit_in, 0.125 * unit_in, 0.5 * unit_in
    tol = 1e-6
    out, ok = [], True
    joints = m.meta.get("joints", [])
    for j in joints:
        P = m.get(j["post"])
        plo, phi = map(np.array, P.meta["stock"])
        mlo, mhi = map(np.array, j["mortise"])
        a, sgn = j["axis"], j["sign"]
        walls = {}
        for i in range(3):
            if i == a:
                far = (phi[a] - mhi[a]) if sgn > 0 else (mlo[a] - plo[a])
                walls["far_face"] = round(float(far), 4)
            else:
                walls["xyz"[i]] = round(float(min(mlo[i] - plo[i], phi[i] - mhi[i])), 4)
        g = P.meta.get("grain", "z")
        gi = AX[g]
        end_dist = float(min(mlo[gi] - plo[gi], phi[gi] - mhi[gi])) if gi != a else None
        flags = []
        through = walls["far_face"] <= tol
        if not through and walls["far_face"] < FAR - tol:
            flags.append("far_face")
        for i in range(3):
            if i != a and walls["xyz"[i]] < W - tol and walls["xyz"[i]] > tol:
                flags.append(f"wall_{'xyz'[i]}")
            if i != a and walls["xyz"[i]] <= tol:
                flags.append(f"open_{'xyz'[i]}")    # mortise breaks out of the stick
        if end_dist is not None and end_dist < END - tol and not j["haunch"]:
            flags.append("near_end")
        rec = {"joint": f"{j['rail']}->{j['post']}", "walls": walls, "end_dist": None if end_dist is None else round(end_dist, 4),
               "flags": flags}
        ok &= not flags
        out.append(rec)
    # webs between mortises in the same stick
    webs = []
    for i in range(len(joints)):
        for k in range(i + 1, len(joints)):
            if joints[i]["post"] != joints[k]["post"]:
                continue
            d = box_gap(joints[i]["mortise"], joints[k]["mortise"])
            if d < WEB - tol:
                webs.append({"post": joints[i]["post"], "a": joints[i]["rail"], "b": joints[k]["rail"], "web": round(d, 4)})
    ok &= not webs
    return {"pass": bool(ok), "joints": out, "thin_webs": webs,
            "rules": {"wall": W, "web": WEB, "far_face": FAR, "from_end": END, "units": m.units}}

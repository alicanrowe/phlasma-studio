"""2D sketching for plates (shapely), plus flat-pack joinery cuts. All in model units, (u, v) = (along grain, across).

Every joinery helper returns a shapely geometry to union or subtract, so a part reads as plain boolean steps:
    side = rect(-xo, 0, xo, H) - slot(xc, 0, T + FIT, H / 2)
"""
import math
from shapely.geometry import Polygon, Point, box as _box
from shapely.ops import unary_union
from shapely import affinity


def rect(x0, y0, x1, y1):
    return _box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def crect(cx, cy, w, h):
    return rect(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def circle(cx, cy, r, n=64):
    return Point(cx, cy).buffer(r, quad_segs=max(4, n // 4))


def poly(pts):
    return Polygon(pts).buffer(0)


def union(*g):
    return unary_union([x for x in g if x is not None])


def mirror_v(g):
    return affinity.scale(g, 1, -1, origin=(0, 0))


def mirror_u(g):
    return affinity.scale(g, -1, 1, origin=(0, 0))


def rounded(g, r):
    """round every outside corner by r (inside corners too when r > 0: opening then closing)."""
    return g.buffer(-r, join_style=1).buffer(r, join_style=1) if r > 0 else g


# ------------------------------------------------------------------ joinery (flat-pack, plates of thickness T)
def slot(xc, y_edge, width, depth, from_top=False, eps=0.01):
    """an open slot (half-lap, bridle, slot-together) cut in from an edge at y_edge. from_top: slot hangs down
    from y_edge. width already includes the fit clearance."""
    if from_top:
        return rect(xc - width / 2, y_edge - depth, xc + width / 2, y_edge + eps)
    return rect(xc - width / 2, y_edge - eps, xc + width / 2, y_edge + depth)


def mortise(xc, yc, w, h, fit=0.0):
    """a closed through-mortise for a tab, sized tab + fit each way"""
    return crect(xc, yc, w + fit, h + fit)


def tab(x_edge, yc, length, w, direction=+1):
    """a tenon/tab sticking out of an edge at x_edge by `length` (direction +1 = toward +u)"""
    x1 = x_edge + direction * length
    return rect(min(x_edge, x1), yc - w / 2, max(x_edge, x1), yc + w / 2)


def fingers(x0, x1, y_edge, depth, n, width_fit=0.0, start_in=True, from_top=False):
    """box (finger) joint along an edge from x0 to x1: n equal fingers, alternate ones cut away.
    start_in=True cuts the 1st, 3rd ... so the mating part uses start_in=False. Returns the cut geometry."""
    w = (x1 - x0) / n
    cuts = []
    for i in range(n):
        if (i % 2 == 0) == start_in:
            a, b = x0 + i * w - width_fit / 2, x0 + (i + 1) * w + width_fit / 2
            cuts.append(slot((a + b) / 2, y_edge, b - a, depth, from_top=from_top))
    return union(*cuts)


def arch(x0, x1, h, y=0.0, n=48, eps=0.01):
    """a shallow segmental arch rising h from an edge at y, from x0 to x1 (cut it from a rail for feet)"""
    c, hw = (x0 + x1) / 2, (x1 - x0) / 2
    rr = (hw ** 2 + h ** 2) / (2 * h)
    pts = [(x0, y - eps)]
    for i in range(n + 1):
        x = x0 + (x1 - x0) * i / n
        pts.append((x, y + max(0.0, math.sqrt(max(rr ** 2 - (x - c) ** 2, 0)) - (rr - h))))
    pts.append((x1, y - eps))
    return poly(pts)


def kerf(g, k):
    """offset for the laser: outsides grow k/2, holes shrink k/2, so the part comes out at its drawn size"""
    return g.buffer(k / 2, join_style=2, mitre_limit=4) if k else g


def dogbone(g, tool_d, max_angle=179.0):
    """CNC relief: a round bit can't cut a sharp inside corner, so drill a circle of the bit's diameter into every
    inside corner (outline notches and hole corners alike), centred on the corner bisector one radius out from the corner (its edge passes through it), so the
    mating square part still seats fully. Returns the relieved outline."""
    from shapely.geometry.polygon import orient
    import numpy as np
    r = tool_d / 2
    polys = list(getattr(g, "geoms", [g]))
    cuts = []
    for pg in polys:
        pg = orient(pg, 1.0)                      # material on the left of every ring
        for ring in [pg.exterior, *pg.interiors]:
            P = np.array(ring.coords)[:-1]
            n = len(P)
            for i in range(n):
                a, v, b = P[i - 1], P[i], P[(i + 1) % n]
                e1, e2 = v - a, b - v
                l1, l2 = np.linalg.norm(e1), np.linalg.norm(e2)
                if l1 < 1e-9 or l2 < 1e-9:
                    continue
                e1, e2 = e1 / l1, e2 / l2
                if e1[0] * e2[1] - e1[1] * e2[0] >= -1e-9:       # left turn or straight: outside corner
                    continue
                turn = np.degrees(np.arccos(np.clip(-(e1 @ e2), -1, 1)))
                if turn > max_angle:
                    continue
                bis = e1 - e2                     # points into the material at an inside corner
                bis /= np.linalg.norm(bis)
                c = v - bis * r                    # bit centre: on the bisector, its edge through the corner
                cuts.append(Point(*c).buffer(r, quad_segs=8))
    return g.difference(unary_union(cuts)) if cuts else g

"""Signed-distance sculpting for cast jewelry (and any organic form). Own implementation, NumPy only.

A Scene is an ordered list of (primitive, op, k): op 0 = add, 1 = subtract, 2 = intersect, k = blend radius in
model units (polynomial smooth min). Meshing evaluates every primitive only inside its own bounding box (+ k),
straight into a dense grid, then runs marching cubes. Distances follow Inigo Quilez's published formulas.

    s = Scene()
    s.ellipsoid((0, 0, 12), (4, 3.5, 3), k=1.2)
    s.stroke(points, radii, k=0.4)          # a swelling/tapering member along a path
    s.cone(a, b, ra, rb, op=1, k=0.05)      # cut a seat
    mesh = s.mesh(voxel=0.12)
"""
import math
import numpy as np

BIG = 1e3


def nz(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def perp(n):
    n = nz(n)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    return nz(np.cross(n, a))


def frame_from(xaxis, zhint):
    z = nz(zhint) if zhint is not None else np.array([0, 0, 1.0])
    x = np.asarray(xaxis, float) if xaxis is not None else perp(z)
    x = x - z * np.dot(x, z)
    if np.linalg.norm(x) < 1e-9:
        x = perp(z)
    x = nz(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])          # rows: local axes in world


# ------------------------------------------------------------------ distance functions (P: (N,3) float32)
def d_sphere(P, c, r):
    return np.linalg.norm(P - c, axis=1) - r


def d_ellipsoid(P, c, R, Rm):
    q = (P - c) @ Rm.T
    k0 = np.linalg.norm(q / R, axis=1)
    k1 = np.linalg.norm(q / (R * R), axis=1)
    return np.where(k1 > 1e-9, k0 * (k0 - 1.0) / np.maximum(k1, 1e-9), -min(R))


def d_torus(P, c, Rr, r, Rm):
    q = (P - c) @ Rm.T
    a = np.sqrt(q[:, 0] ** 2 + q[:, 1] ** 2) - Rr
    return np.sqrt(a * a + q[:, 2] ** 2) - r


def d_round_cone(P, a, b, r1, r2):
    """sphere-swept cone between a (radius r1) and b (radius r2): exact"""
    ba = b - a
    l2 = float(ba @ ba)
    if l2 < 1e-12:
        return np.linalg.norm(P - a, axis=1) - max(r1, r2)
    rr = r1 - r2
    a2 = l2 - rr * rr
    il2 = 1.0 / l2
    pa = P - a
    y = pa @ ba
    z = y - l2
    xv = pa * l2 - np.outer(y, ba)
    x2 = np.einsum("ij,ij->i", xv, xv)
    y2 = y * y * l2
    z2 = z * z * l2
    k = np.sign(rr) * rr * rr * x2
    out = np.empty(len(P), np.float64)
    m1 = np.sign(z) * a2 * z2 > k
    m2 = (~m1) & (np.sign(y) * a2 * y2 < k)
    m3 = ~(m1 | m2)
    out[m1] = np.sqrt(x2[m1] + z2[m1]) * il2 - r2
    out[m2] = np.sqrt(x2[m2] + y2[m2]) * il2 - r1
    out[m3] = (np.sqrt(x2[m3] * a2 * il2) + y[m3] * rr) * il2 - r1
    return out


def d_capped_cone(P, a, b, ra, rb):
    """flat-ended cone frustum from a (radius ra) to b (radius rb): exact"""
    rba = rb - ra
    ba = b - a
    baba = float(ba @ ba)
    pa = P - a
    papa = np.einsum("ij,ij->i", pa, pa)
    paba = (pa @ ba) / baba
    x = np.sqrt(np.maximum(papa - paba * paba * baba, 0))
    cax = np.maximum(0.0, x - np.where(paba < 0.5, ra, rb))
    cay = np.abs(paba - 0.5) - 0.5
    k = rba * rba + baba
    f = np.clip((rba * (x - ra) + paba * baba) / k, 0.0, 1.0)
    cbx = x - ra - f * rba
    cby = paba - f
    s = np.where((cbx < 0.0) & (cay < 0.0), -1.0, 1.0)
    return s * np.sqrt(np.minimum(cax * cax + cay * cay * baba, cbx * cbx + cby * cby * baba))


def d_box(P, c, half, Rm, rnd=0.0):
    q = np.abs((P - c) @ Rm.T) - (half - rnd)
    return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0) - rnd


def smin(a, b, k):
    if k <= 0:
        return np.minimum(a, b)
    h = np.maximum(k - np.abs(a - b), 0) / k
    return np.minimum(a, b) - h * h * k * 0.25


def smax(a, b, k):
    return -smin(-a, -b, k)


# ------------------------------------------------------------------ scene
class Scene:
    def __init__(self):
        self.rows = []   # dicts: kind, args, op, k, lo, hi

    def _add(self, kind, args, lo, hi, op, k):
        self.rows.append(dict(kind=kind, args=args, op=int(op), k=float(k),
                              lo=np.asarray(lo, float), hi=np.asarray(hi, float)))
        return self

    def copy(self):
        s = Scene()
        s.rows = list(self.rows)
        return s

    # primitives -----------------------------------------------------
    def sphere(self, c, r, op=0, k=0.0):
        c = np.asarray(c, float)
        return self._add("sphere", (c, r), c - r, c + r, op, k)

    def ellipsoid(self, c, radii, xaxis=None, zhint=None, op=0, k=0.0):
        c, R = np.asarray(c, float), np.asarray(radii, float)
        Rm = frame_from(xaxis, zhint)
        e = np.abs(Rm.T) @ R          # exact bbox half-extent of the rotated ellipsoid (upper bound)
        return self._add("ellipsoid", (c, R, Rm), c - e, c + e, op, k)

    def torus(self, c, R, r, normal=(0, 0, 1), op=0, k=0.0):
        c = np.asarray(c, float)
        Rm = frame_from(None, normal)
        e = R + r
        return self._add("torus", (c, R, r, Rm), c - e, c + e, op, k)

    def cone(self, a, b, ra, rb, op=0, k=0.0, round=False):
        a, b = np.asarray(a, float), np.asarray(b, float)
        r = max(ra, rb)
        return self._add("rcone" if round else "ccone", (a, b, ra, rb),
                         np.minimum(a, b) - r, np.maximum(a, b) + r, op, k)

    def box(self, c, size, xaxis=None, zhint=None, rnd=0.0, op=0, k=0.0):
        c = np.asarray(c, float)
        Rm = frame_from(xaxis, zhint)
        half = np.asarray(size, float) / 2
        e = np.abs(Rm.T) @ half
        return self._add("box", (c, half, Rm, rnd), c - e, c + e, op, k)

    def stroke(self, P, radii, op=0, k=0.0):
        """a member along polyline P with radius radii[i] at each point (sphere-swept cones, hard-joined)"""
        P = np.asarray(P, float)
        r = np.broadcast_to(np.asarray(radii, float), (len(P),)).copy()
        rm = r.max()
        return self._add("stroke", (P, r), P.min(0) - rm, P.max(0) + rm, op, k)

    def halfspace(self, p, n, op=2, k=0.0, extent=200.0):
        p, n = np.asarray(p, float), nz(n)
        return self._add("half", (p, n), p - extent, p + extent, op, k)

    # evaluation -------------------------------------------------------
    @staticmethod
    def eval_row(row, P):
        kd, a = row["kind"], row["args"]
        if kd == "sphere":
            return d_sphere(P, *a)
        if kd == "ellipsoid":
            return d_ellipsoid(P, *a)
        if kd == "torus":
            return d_torus(P, *a)
        if kd == "rcone":
            return d_round_cone(P, *a)
        if kd == "ccone":
            return d_capped_cone(P, *a)
        if kd == "box":
            return d_box(P, *a)
        if kd == "half":
            return (P - a[0]) @ a[1]
        if kd == "stroke":
            Pp, r = a
            d = np.full(len(P), BIG)
            for i in range(len(Pp) - 1):
                lo = np.minimum(Pp[i], Pp[i + 1]) - max(r[i], r[i + 1]) - 0.5
                hi = np.maximum(Pp[i], Pp[i + 1]) + max(r[i], r[i + 1]) + 0.5
                m = np.all((P >= lo) & (P <= hi), 1)
                if m.any():
                    d[m] = np.minimum(d[m], d_round_cone(P[m], Pp[i], Pp[i + 1], r[i], r[i + 1]))
            return d
        raise ValueError(kd)

    @staticmethod
    def combine(d, p, op, k):
        if op == 0:
            return smin(d, p, k)
        if op == 1:
            return smax(d, -p, k)
        return smax(d, p, k)

    def eval_pts(self, P):
        P = np.asarray(P, float)
        d = np.full(len(P), BIG)
        for row in self.rows:
            d = self.combine(d, self.eval_row(row, P), row["op"], row["k"])
        return d

    def bounds(self, pad=1.0):
        adds = [r for r in self.rows if r["op"] == 0]
        lo = np.min([r["lo"] for r in adds], 0) - pad
        hi = np.max([r["hi"] for r in adds], 0) + pad
        return lo, hi

    def grid(self, voxel, lo=None, hi=None, log=None):
        if lo is None:
            lo, hi = self.bounds(pad=2 * voxel + max(r["k"] for r in self.rows))
        n = np.ceil((hi - lo) / voxel).astype(int) + 1
        if np.prod(n) > 6e7:
            raise ValueError(f"grid {n.tolist()} too big for voxel {voxel}; raise voxel")
        D = np.full(tuple(n), BIG, np.float32)
        axes = [lo[i] + voxel * np.arange(n[i]) for i in range(3)]
        for j, row in enumerate(self.rows):
            if row["kind"] == "half" or row["op"] == 2:
                sl = (slice(0, n[0]), slice(0, n[1]), slice(0, n[2]))
            else:
                m = row["k"] + 2 * voxel
                i0 = np.clip(np.floor((row["lo"] - m - lo) / voxel).astype(int), 0, n)
                i1 = np.clip(np.ceil((row["hi"] + m - lo) / voxel).astype(int) + 1, 0, n)
                if np.any(i1 <= i0):
                    continue
                sl = tuple(slice(i0[i], i1[i]) for i in range(3))
            sub = D[sl]
            ax = [axes[i][sl[i]] for i in range(3)]
            # chunk along x so no block holds more than ~3M points at once
            step = max(1, int(3e6 // max(1, len(ax[1]) * len(ax[2]))))
            for x0 in range(0, len(ax[0]), step):
                X, Y, Z = np.meshgrid(ax[0][x0:x0 + step], ax[1], ax[2], indexing="ij")
                P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
                p = self.eval_row(row, P).reshape(X.shape)
                blk = sub[x0:x0 + step].astype(np.float64)
                sub[x0:x0 + step] = self.combine(blk, p, row["op"], row["k"]).astype(np.float32)
            if log and j % 10 == 0:
                log("sdf_progress", row=j, rows=len(self.rows))
        return D, lo

    def mesh(self, voxel=0.12, lo=None, hi=None, log=None, keep="all"):
        import trimesh
        from skimage.measure import marching_cubes
        D, lo = self.grid(voxel, lo, hi, log)
        if D.min() > 0:
            raise ValueError("scene is empty")
        v, f, _, _ = marching_cubes(D, 0.0, spacing=(voxel,) * 3)
        m = trimesh.Trimesh(v + lo, f[:, ::-1], process=True)
        m.merge_vertices()
        if m.volume < 0:
            m.invert()
        if keep == "largest":
            parts = m.split(only_watertight=False)
            m = max(parts, key=lambda x: abs(x.volume))
        return m


# ------------------------------------------------------------------ helpers used by jewelry pieces
def dense(P, *cols, step=0.3):
    """resample a polyline (and values that ride along it) to about `step` spacing"""
    P = np.asarray(P, float)
    L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    n = max(2, int(L[-1] / step) + 1)
    t = np.linspace(0, L[-1], n)
    out = [np.stack([np.interp(t, L, P[:, i]) for i in range(3)], 1)]
    for c in cols:
        c = np.asarray(c, float)
        out.append(np.interp(t, L, c) if c.ndim == 1 else np.stack([np.interp(t, L, c[:, i]) for i in range(c.shape[1])], 1))
    return out if cols else out[0]


def surface_along(scene, p, n, span=8.0, step=0.01):
    """first point where the scene's surface is met coming in from outside along -n, starting span out from p"""
    p, n = np.asarray(p, float), nz(n)
    t = np.arange(span, -span, -step)
    pts = p[None] + np.outer(t, n)
    d = scene.eval_pts(pts)
    j = np.nonzero(d <= 0)[0]
    return pts[j[0]] if len(j) else p

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
    # default local x = world x (world y when z is along x), so an unrotated ellipsoid's radii read as (x, y, z)
    x = np.asarray(xaxis, float) if xaxis is not None else (np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0]))
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

    def grid_sdf(self, D, lo, pitch, op=0, k=0.0):
        """a sampled distance grid (an imported or generated sculpt) as a primitive, trilinear"""
        D = np.asarray(D, np.float32)
        lo = np.asarray(lo, float)
        hi = lo + pitch * (np.array(D.shape) - 1)
        return self._add("grid", (D, lo, float(pitch)), lo, hi, op, k)

    def group(self, sub, warps=(), op=0, k=0.0, pad=None):
        """evaluate a sub-scene at warped points. warps, applied in order to the query point:
        ("twist", {"axis": "z", "rate": deg_per_unit, "c": [x, y, z]}), ("bend", {"axis": "x", "up": "z", "radius": R}),
        ("taper", {"axis": "z", "rate": per_unit, "c": [...]}), ("mirror", {"axis": "x"}). Distances after a warp are
        not exact (fine for meshing; keep blend radii modest)."""
        lo, hi = sub.bounds(pad=0.0)
        ext = float(np.max(hi - lo))
        pad = ext * 0.6 if pad is None else pad
        return self._add("group", (sub, list(warps)), lo - pad, hi + pad, op, k)

    def displace(self, kind="noise", amp=0.2, scale=2.0, octaves=3, seed=0, axis=(0, 0, 1), period=1.0, wobble=0.0,
                 lo=None, hi=None):
        """surface relief on everything added so far: d += amp * field(P).
        kind "noise": fractal value noise of feature size `scale`; "bands": growth bands across `axis` every
        `period`, wobbled by noise of size `scale` and strength `wobble` (0..1), never even ribs when wobble > 0."""
        if lo is None:
            lo, hi = self.bounds(pad=abs(amp) + 0.5)
        return self._add("disp", dict(kind=kind, amp=amp, scale=scale, octaves=octaves, seed=seed,
                                      axis=nz(axis), period=period, wobble=wobble), lo, hi, 3, 0.0)

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
        if kd == "grid":
            from scipy.ndimage import map_coordinates
            D, lo, pitch = a
            idx = ((P - lo) / pitch).T
            inside = np.all((idx >= 0) & (idx <= (np.array(D.shape)[:, None] - 1)), 0)
            out = np.full(len(P), BIG)
            if inside.any():
                out[inside] = map_coordinates(D, idx[:, inside], order=1, mode="nearest")
            return out
        if kd == "group":
            sub, warps = a
            return sub.eval_pts(warp_points(P, warps))
        if kd == "disp":
            return field(P, a)
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
        if op == 3:
            return d + p
        if op == 0:
            return smin(d, p, k)
        if op == 1:
            return smax(d, -p, k)
        return smax(d, p, k)

    def eval_pts(self, P):
        P = np.asarray(P, float)
        d = np.full(len(P), BIG)
        for row in self.rows:
            if row["op"] == 2 or row["kind"] == "half":
                d = self.combine(d, self.eval_row(row, P), row["op"], row["k"])
                continue
            m = row["k"] + 1e-6
            sel = np.all((P >= row["lo"] - m) & (P <= row["hi"] + m), 1)
            if sel.any():
                d[sel] = self.combine(d[sel], self.eval_row(row, P[sel]), row["op"], row["k"])
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
        D[np.abs(D) < 1e-5] = 1e-5            # a grid node exactly on the surface makes coincident vertices
        v, f, _, _ = marching_cubes(D, 0.0, spacing=(voxel,) * 3)
        m = trimesh.Trimesh(v + lo, f[:, ::-1], process=True)
        m.merge_vertices()
        m.update_faces(m.nondegenerate_faces())
        m.remove_unreferenced_vertices()
        if m.volume < 0:
            m.invert()
        if keep == "largest":
            parts = m.split(only_watertight=False)
            m = max(parts, key=lambda x: abs(x.volume))
        return m


# ------------------------------------------------------------------ warps and relief fields
_AX = {"x": 0, "y": 1, "z": 2}


def warp_points(P, warps):
    Q = np.array(P, float)
    for kind, w in warps:
        if kind == "mirror":
            i = _AX[w.get("axis", "x")]
            Q[:, i] = np.abs(Q[:, i] - w.get("at", 0.0)) + w.get("at", 0.0)
            continue
        c = np.asarray(w.get("c", (0, 0, 0)), float)
        Q = Q - c
        a = _AX[w.get("axis", "z")]
        u, v = [i for i in range(3) if i != a]
        if kind == "twist":
            t = np.radians(w["rate"]) * Q[:, a]
            cu, sv = np.cos(t), np.sin(t)
            Q[:, u], Q[:, v] = cu * Q[:, u] + sv * Q[:, v], -sv * Q[:, u] + cu * Q[:, v]
        elif kind == "taper":
            sc = 1.0 + w["rate"] * Q[:, a]
            sc = np.maximum(sc, 0.05)
            Q[:, u] /= sc; Q[:, v] /= sc
        elif kind == "bend":
            # bend the along-axis line into an arc of radius R curving toward +up
            R = float(w["radius"]); up = _AX[w.get("up", "z")]
            x, y = Q[:, a], Q[:, up]
            rr = np.hypot(x, R - y)
            th = np.arctan2(x, R - y)
            Q[:, a], Q[:, up] = R * th, R - rr
        Q = Q + c
    return Q


def _hash3(i, j, k, seed):
    h = (i * 73856093) ^ (j * 19349663) ^ (k * 83492791) ^ (seed * 2654435761)
    h = (h ^ (h >> 13)) * 1274126177
    return ((h ^ (h >> 16)) & 0xFFFFFF) / float(0xFFFFFF) * 2 - 1


def value_noise(P, scale, seed=0):
    Q = np.asarray(P, float) / scale
    i0 = np.floor(Q).astype(np.int64)
    f = Q - i0
    f = f * f * (3 - 2 * f)
    out = 0.0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                out = out + w * _hash3(i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz, seed)
    return out


def fbm(P, scale, octaves=3, seed=0):
    out, amp, tot = 0.0, 1.0, 0.0
    for o in range(octaves):
        out = out + amp * value_noise(P, scale / 2 ** o, seed + o)
        tot += amp; amp *= 0.5
    return out / tot


def field(P, a):
    if a["kind"] == "noise":
        return a["amp"] * fbm(P, a["scale"], a["octaves"], a["seed"])
    if a["kind"] == "bands":
        t = P @ a["axis"]
        if a["wobble"]:
            t = t + a["wobble"] * a["period"] * fbm(P, a["scale"], 2, a["seed"])
        return a["amp"] * (0.5 - 0.5 * np.cos(2 * np.pi * t / a["period"])) ** 2
    raise ValueError(a["kind"])


def weld(mesh):
    """weld an imported mesh by position only. AI and game exports (Tripo GLBs among them) are split along UV and
    normal seams; left split, the mesh is not watertight and a voxel inside/outside fill fails at random."""
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    return mesh


def mesh_to_sdf(mesh, pitch, pad=3):
    """sampled signed distance of a (possibly leaky) mesh: voxelize, fill the inside, distance transform both ways.
    Closes pinholes and gaps under ~2 voxels, so a messy scan or AI mesh comes out as one watertight solid."""
    from scipy.ndimage import distance_transform_edt, binary_fill_holes, binary_closing
    vg = mesh.voxelized(pitch)
    occ = vg.matrix
    occ = np.pad(occ, pad)
    occ = binary_closing(occ, iterations=1)
    occ = binary_fill_holes(occ)
    lo = vg.translation - pad * pitch           # centre of voxel [0,0,0]
    din = distance_transform_edt(occ)
    dout = distance_transform_edt(~occ)
    D = ((dout - din) * pitch - 0.0).astype(np.float32)
    D[occ] += 0.5 * pitch                        # put the zero level on voxel boundaries
    D[~occ] -= 0.5 * pitch
    # near the surface, replace the voxel-staircase distance by the true distance to the mesh surface (sampled
    # densely, nearest point), keeping the inside/outside sign from the filled voxels
    from scipy.spatial import cKDTree
    band = np.abs(D) < 3 * pitch
    idx = np.argwhere(band)
    if len(idx):
        n = int(min(4e6, max(2e5, mesh.area / (pitch * pitch) * 8)))
        pts, fi = mesh.sample(n, return_index=True)
        nrm = mesh.face_normals[fi]
        tree = cKDTree(pts)
        q = np.asarray(lo, float) + idx * pitch
        dist, j = tree.query(q, k=4, workers=-1)
        # sign from the nearest samples' normals (voxel occupancy is a voxel too fat at the surface)
        side = np.einsum("ijk,ijk->ij", q[:, None, :] - pts[j], nrm[j]).sum(1)
        D[tuple(idx.T)] = np.where(side < 0, -dist[:, 0], dist[:, 0]).astype(np.float32)
    return D, np.asarray(lo, float), pitch


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

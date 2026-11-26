"""Rigid parts against the leather: seat the cloth round them, then a hard clipping gate.

A tool in a pocket, a buckle on a tab or a magnet under a placket is rigid; the hide is not. After the drape we
treat every rigid part as a collider for the render surface: cloth that should lie under it (the panel a tool rests
on) is pressed in, cloth that should lie over it (the sleeve or strip holding it) is lifted, with a smooth falloff
so it reads as leather giving way, not a spike. Then `clip_report` measures what is left: any cloth inside a part
(or any part inside the hide) deeper than the tolerance fails the gate, and a failed gate means no render.
"""
import os
import numpy as np


def _adjacency(tris, n):
    import scipy.sparse as sp
    e = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e = np.vstack([e, e[:, ::-1]])
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    return A, np.maximum(deg, 1.0)


def _sd(mesh, P):
    """signed distance, positive inside the part (trimesh convention)"""
    import trimesh
    if len(P) == 0:
        return np.zeros(0)
    return trimesh.proximity.signed_distance(mesh, P)


def _near(P, mesh, pad):
    lo, hi = mesh.bounds[0] - pad, mesh.bounds[1] + pad
    return np.nonzero(np.all((P >= lo) & (P <= hi), axis=1))[0]


def _hull(pm):
    try:
        return pm.convex_hull
    except Exception:
        return pm


def sides(X, N, hull, idx):
    """For cloth vertices X[idx] (normals N[idx]) near a part: True where the cloth lies OUTSIDE the part (over it),
    i.e. the part's nearest surface is inward of the vertex along its normal; a vertex already inside the part is
    judged against the part's centre. Lets a skirt panel hanging over a pocket box lift instead of crushing it."""
    import trimesh
    if len(idx) == 0:
        return np.zeros(0, bool)
    P, n = X[idx], N[idx]
    cp, _, _ = trimesh.proximity.closest_point(hull, P)
    h = ((cp - P) * n).sum(1)
    inside = _sd(hull, P) > 0
    h[inside] = ((hull.centroid - P[inside]) * n[inside]).sum(1)
    return h < 0


def _samples(pm, step=0.0025, cap=8000):
    """the part's surface as points: its vertices plus an even sample (a box has 8 vertices; a cloth vertex grid
    12 mm apart can miss a 15 mm knob), so contact is judged from the part's side"""
    import trimesh
    V = np.asarray(pm.vertices)
    k = int(min(cap, max(64, pm.area / step ** 2)))
    try:
        S, _ = trimesh.sample.sample_surface(pm, k, seed=1)
    except TypeError:
        S, _ = trimesh.sample.sample_surface(pm, k)
    P = np.vstack([V, S])
    if len(P) > 2 * cap:
        P = P[np.random.default_rng(0).choice(len(P), 2 * cap, replace=False)]
    return P


def _pairs(tree, P, Xr, N, radius, reach=0.03, chunk=1500):
    """(cloth vertex, height of the part point above it along its normal, lateral offset) for every part point
    within `radius` laterally and `reach` along the normal of a cloth vertex; built in chunks to bound memory"""
    J, H, L = [], [], []
    for c0 in range(0, len(P), chunk):
        Pc = P[c0:c0 + chunk]
        nb = tree.query_ball_point(Pc, r=float(np.hypot(radius, reach)))
        cnt = np.fromiter((len(x) for x in nb), int, len(nb))
        if not cnt.sum():
            continue
        ii = np.repeat(np.arange(len(Pc)), cnt)
        jj = np.fromiter((v for x in nb for v in x), int, int(cnt.sum()))
        rel = Pc[ii] - Xr[jj]
        h = (rel * N[jj]).sum(1)
        lat = np.linalg.norm(rel - N[jj] * h[:, None], axis=1)
        ok = (lat < radius) & (np.abs(h) < reach)
        J.append(jj[ok]); H.append(h[ok]); L.append(lat[ok])
    if not J:
        return np.zeros(0, int), np.zeros(0), np.zeros(0)
    return np.concatenate(J), np.concatenate(H), np.concatenate(L)


def seat(Xr, tris, nf, thick, parts, over, clear=0.0012, iters=24, radius=0.0055, falloff=16, step_max=0.006, bound_r=0.004, log=None):
    """Move render vertices off the rigid parts. parts: list of trimesh (world). over: per part, a boolean mask over
    the vertices that lie OVER that part (lifted); every other vertex near it lies UNDER it (pressed in).
    Each part point is splatted onto the cloth vertices round it: a vertex under the part keeps its outer face a
    clearance below the lowest part point above it; a vertex over the part keeps its inner face above the highest."""
    from scipy.spatial import cKDTree
    n = len(Xr)
    A, deg = _adjacency(tris, n)
    SP = [_samples(pm) for pm in parts]
    HL = [_hull(pm) for pm in parts]
    moved = np.zeros(n)
    # which side of each part every nearby vertex is on is decided once, before anything moves: a vertex pushed
    # past a part's middle must not flip sides and get shoved on through it
    N0 = nf(Xr)
    OV = []
    for ov0, hl in zip(over, HL):
        ov = ov0.copy()
        near = _near(Xr, hl, 0.03)
        ov[near] |= sides(Xr, N0, hl, near)
        OV.append(ov)
    for it in range(iters):
        N = nf(Xr)
        tree = cKDTree(Xr)
        d = np.zeros(n)
        hi = np.full(n, np.inf)   # how far a vertex may move out before it enters a part above it
        lo = np.full(n, -np.inf)  # how far it may move in before a part below it pokes through
        for P, ov in zip(SP, OV):
            jj, h, lat = _pairs(tree, P, Xr, N, radius)
            if not len(jj):
                continue
            o = ov[jj]
            # which side each vertex is on comes from the geometry (sides) plus the job's explicit "over" pieces
            # under: outer face (thick) must sit `clear` below the part point -> push = h - thick - clear (<0 means in)
            pu = h - thick - clear
            tight = lat < bound_r  # the bounds only count parts (nearly) straight above / below a vertex
            m = (~o) & (h > -0.03) & tight
            if m.any():
                np.minimum.at(hi, jj[m], pu[m])
            m = (~o) & (h > -0.03)
            m &= pu < 0
            if m.any():
                np.minimum.at(d, jj[m], pu[m])
            # over: inner face must sit `clear` above the part point -> push = h + clear (>0 means out)
            po = h + clear
            m = o & (h < 0.03) & tight
            if m.any():
                np.maximum.at(lo, jj[m], po[m])
            m = o & (h < 0.03)
            m &= po > 0
            if m.any():
                np.maximum.at(d, jj[m], po[m])
        if not np.any(np.abs(d) > 1e-5):
            break
        d = np.clip(d, -step_max, step_max)
        s = d.copy()
        for _ in range(falloff):  # leather gives way over a few cm, not one vertex
            s = 0.5 * s + 0.5 * (A @ s) / deg
            s = np.where(np.abs(d) > np.abs(s), d, s)
            s = np.minimum(np.maximum(s, np.minimum(lo, 0.0)), np.maximum(hi, 0.0))  # the falloff never pushes into a part
        Xr += N * s[:, None]
        moved += np.abs(s)
    if log:
        log("seat", parts=len(parts), moved_max_mm=round(float(moved.max()) * 1000, 2),
            moved_verts=int((moved > 1e-5).sum()), iters=it + 1)
    return Xr


def clip_report(Xr, tris, nf, thick, parts, names=None, tol=0.0005, samples=3):
    """Penetration of rigid parts against the hide and against each other.
    - cloth in part: points through the hide (inner, mid, outer face; vertices and triangle centres) inside a part
    - part in cloth: part vertices that sit inside the hide slab (thin edges a vertex test would miss)
    - part in part: pairwise overlap depth between parts of different tools (bounding test, then signed distance)
    Returns {"pass", "worst_mm", "hits": [...]}."""
    import trimesh
    names = names or [f"part{i}" for i in range(len(parts))]
    N = nf(Xr)
    C = Xr[tris].mean(1)
    NC = nf(C)
    P0 = np.vstack([Xr, C])
    N0 = np.vstack([N, NC])
    hits, worst = [], 0.0
    hulls = [_hull(pm) for pm in parts]  # watertight, so inside/outside is exact; conservative for recesses
    for pm, hl, nm in zip(parts, hulls, names):
        idx = _near(P0, pm, 0.005)
        if len(idx):
            depth = np.full(len(idx), -1.0)
            for f in np.linspace(0.0, 1.0, samples):
                depth = np.maximum(depth, _sd(hl, P0[idx] + N0[idx] * thick * f))
            k = int(np.argmax(depth))
            if depth[k] > tol:
                hits.append({"kind": "cloth_in_part", "part": nm, "depth_mm": round(float(depth[k]) * 1000, 2),
                             "count": int((depth > tol).sum())})
            worst = max(worst, float(depth[k]))
        # part vertices inside the hide: closest point on the mid surface closer than half a hide
        mid = trimesh.Trimesh(Xr + N * thick / 2, tris, process=False)
        V = _samples(pm)
        sel = _near(V, mid, 0.003)
        if len(sel):
            cp, dist, tid = trimesh.proximity.closest_point(mid, V[sel])
            # through a face only: a point beside a cut edge (lateral to the slab's end) is outside the hide
            fn = np.asarray(mid.face_normals)[tid]
            hgt = np.abs(((V[sel] - cp) * fn).sum(1))
            lat = np.sqrt(np.maximum(dist ** 2 - hgt ** 2, 0.0))
            pen = np.where(lat < 0.00025, thick / 2 - dist, -1.0)
            k = int(np.argmax(pen))
            if pen[k] > tol:
                hits.append({"kind": "part_in_cloth", "part": nm, "depth_mm": round(float(pen[k]) * 1000, 2),
                             "count": int((pen > tol).sum())})
            worst = max(worst, float(pen[k]))
    tool = [nm.split("/")[0] for nm in names]
    for a in range(len(parts)):
        for b in range(a + 1, len(parts)):
            if tool[a] == tool[b]:
                continue
            A, B = hulls[a], hulls[b]
            if np.any(A.bounds[1] < B.bounds[0]) or np.any(B.bounds[1] < A.bounds[0]):
                continue
            # surface samples of each (vertices alone miss a face-through-face crossing), judged on the other's hull
            sa, sb = _samples(parts[a], cap=3000), _samples(parts[b], cap=3000)
            va, vb = sa[_near(sa, B, 0.0)], sb[_near(sb, A, 0.0)]
            dep = max(float(_sd(B, va).max()) if len(va) else -1, float(_sd(A, vb).max()) if len(vb) else -1)
            if dep > tol:
                hits.append({"kind": "part_in_part", "part": f"{names[a]} x {names[b]}", "depth_mm": round(dep * 1000, 2)})
            worst = max(worst, dep)
    return {"pass": not hits, "worst_mm": round(max(worst, 0.0) * 1000, 2), "tol_mm": tol * 1000, "hits": hits}



def _exit_lift(h, P, ax):
    """for points inside a convex hull: how far the hull must move along +ax for each point to leave it
    (through a face that faces -ax)"""
    n = np.asarray(h.face_normals)
    d = (n * np.asarray(h.triangles)[:, 0]).sum(1)
    na = n @ ax
    b = na < -1e-6
    t = (P @ n[b].T - d[b]) / na[b]  # >= 0 inside
    return t.min(1)


def rest(Xr, nf, thick, parts, unders, axis, clear=0.0006, iters=10, cap=0.04, tilt=0.12, margin=0.012):
    """Rest one rigid assembly (a tool, a button) ON the cloth it isn't tucked under: move it along `axis` and, for a
    long part, tip it about its cross axis (a square's 16 mm head on the welt lifts the head end, the blade stays down
    the sleeve), so no supporting cloth is inside it. Solved as a tiny LP per pass: lift(s) = t + theta*s along the
    part's long axis must cover every nearby cloth point's need (its signed depth + clearance), minimising the mean
    lift. Judged on convex hulls (watertight, exact inside/outside). Moves the meshes in place; returns
    (max lift in metres, tilt in radians). A hide pressed into a tool reads as a fault; a tool sitting on it does not."""
    import trimesh
    from scipy.optimize import linprog
    ax = np.asarray(axis, float) / max(np.linalg.norm(axis), 1e-12)
    N = nf(Xr)
    Q = Xr + N * thick  # the outer face, the one a part rests on
    hulls = [_hull(pm) for pm in parts]
    V = np.vstack([np.asarray(h.vertices) for h in hulls])
    o = V.mean(0)
    C = np.cov((V - o).T)
    C = C - np.outer(ax, ax) * (ax @ C @ ax)
    w, U = np.linalg.eigh(C)
    dv = U[:, -1] - ax * (U[:, -1] @ ax)
    dv /= max(np.linalg.norm(dv), 1e-12)
    k_ = np.cross(dv, ax)  # rotating about k_ by theta lifts a point at s along dv by theta*s
    ext = float(np.ptp((V - o) @ dv))
    use_tilt = tilt > 0 and ext > 0.06
    T_tot, th_tot, lift_max = 0.0, 0.0, 0.0
    sv = (V - o) @ dv
    ax_now = ax
    last = 1.0
    for _ in range(iters):
        need, ss = [], []
        for h, um in zip(hulls, unders):
            idx = _near(Q, h, margin)
            idx = idx[um[idx]]
            idx = idx[~sides(Xr, N, h, idx)]  # cloth hanging over the part doesn't hold it down
            if not len(idx):
                continue
            sd = _sd(h, Q[idx])
            m = sd > -margin
            nd = sd.copy()
            ins = sd > 0
            if ins.any():  # inside: the exact lift along the axis that takes the point out through the bottom
                nd[ins] = _exit_lift(h, Q[idx[ins]], ax_now)
            need.append(nd[m] + clear)
            ss.append((Q[idx[m]] - o) @ dv)
        if not need:
            break
        need, ss = np.concatenate(need), np.concatenate(ss)
        if need.max() <= 2e-5 and not use_tilt:
            break
        if need.max() <= 2e-5 and _ > 0 and abs(last) < 1e-5:
            break
        if use_tilt:
            s0, s1 = float(((V - V.mean(0)) @ dv).min()), float(((V - V.mean(0)) @ dv).max())
            r = linprog([2.0, s0 + s1], A_ub=-np.c_[np.ones(len(ss)), ss], b_ub=-need,  # least lift at both ends
                        bounds=[(-T_tot, cap - T_tot), (-tilt - th_tot, tilt - th_tot)], method="highs")  # may settle back down
            dt, dth = (r.x if r.success else (float(need.max()), 0.0))
        else:
            dt, dth = float(need.max()), 0.0
        if os.environ.get("PHL_REST_DEBUG"):
            print("rest it", round(float(need.max()) * 1000, 2), len(need), round(float(dt) * 1000, 2), round(float(np.degrees(dth)), 2))
        if abs(dt) < 1e-6 and abs(dth) < 1e-6:
            break
        M = trimesh.transformations.rotation_matrix(dth, k_, o) if abs(dth) > 1e-9 else np.eye(4)
        M[:3, 3] += ax * dt
        for h in hulls:
            h.apply_transform(M)
        for pm in parts:
            pm.apply_transform(M)
        o = o + ax * dt
        T_tot += dt
        th_tot += dth
        last = abs(dt) + abs(dth) * ext
        sv_now = (V - V.mean(0)) @ dv
        lift_max = T_tot + abs(th_tot) * float(np.abs(sv_now).max())
        if T_tot >= cap:
            break
    return lift_max, th_tot

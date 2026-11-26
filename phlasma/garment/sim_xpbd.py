"""XPBD cloth with sewing, body collision and particle self-collision, in NVIDIA Warp (CUDA, or CPU fallback).

Own implementation of the public XPBD method (Macklin, Mueller, Chentanez 2016) with Jacobi averaging.
"""
import numpy as np
import warp as wp

wp.config.quiet = True


@wp.kernel
def k_predict(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), w: wp.array(dtype=float),
              xp: wp.array(dtype=wp.vec3), g: wp.vec3, dt: float, damp: float):
    i = wp.tid()
    xp[i] = x[i]
    if w[i] > 0.0:
        v[i] = v[i] * damp + g * dt
        x[i] = x[i] + v[i] * dt


@wp.kernel
def k_dist(x: wp.array(dtype=wp.vec3), w: wp.array(dtype=float), e: wp.array(dtype=wp.vec2i),
           rest: wp.array(dtype=float), scale: float, alpha: float, compress: float,
           dx: wp.array(dtype=wp.vec3), cnt: wp.array(dtype=float)):
    k = wp.tid()
    i = e[k][0]
    j = e[k][1]
    wi = w[i]
    wj = w[j]
    ws = wi + wj
    if ws == 0.0:
        return
    d = x[i] - x[j]
    L = wp.length(d)
    if L < 1.0e-9:
        return
    r = rest[k] * scale
    C = L - r
    if C < 0.0:
        C = C * compress  # leather resists stretch far more than it resists buckling
    lam = -C / (ws + alpha)
    n = d / L
    wp.atomic_add(dx, i, n * (lam * wi))
    wp.atomic_add(dx, j, -n * (lam * wj))
    wp.atomic_add(cnt, i, 1.0)
    wp.atomic_add(cnt, j, 1.0)


@wp.kernel
def k_dist_gs(x: wp.array(dtype=wp.vec3), w: wp.array(dtype=float), e: wp.array(dtype=wp.vec2i),
              rest: wp.array(dtype=float), kind: wp.array(dtype=int), lam: wp.array(dtype=float), start: int,
              a_s: float, a_b: float, a_m: float, a_c: float, seam_scale: float, sc: wp.array(dtype=float)):
    k = wp.tid() + start
    i = e[k][0]
    j = e[k][1]
    wi = w[i]
    wj = w[j]
    ws = wi + wj
    if ws == 0.0:
        return
    d = x[i] - x[j]
    L = wp.length(d)
    if L < 1.0e-9:
        return
    c = kind[k]
    r = rest[k]
    alpha = a_s
    if c == 1:
        alpha = a_b * sc[k]  # per-piece bend stiffness (a collar stand, a stiffened panel)
    if c == 2:
        alpha = a_m
        r = r * seam_scale
    if c == 3:
        alpha = a_m
    C = L - r
    if c == 0 and C < 0.0:
        alpha = a_c * sc[k]  # leather buckles easily but barely stretches
    dl = (-C - alpha * lam[k]) / (ws + alpha)
    lam[k] = lam[k] + dl
    n = d / L
    x[i] = x[i] + n * (dl * wi)
    x[j] = x[j] - n * (dl * wj)


@wp.kernel
def k_apply(x: wp.array(dtype=wp.vec3), dx: wp.array(dtype=wp.vec3), cnt: wp.array(dtype=float), relax: float):
    i = wp.tid()
    c = cnt[i]
    if c > 0.0:
        x[i] = x[i] + dx[i] * (relax / c)
    dx[i] = wp.vec3(0.0)
    cnt[i] = 0.0


@wp.kernel
def k_body(x: wp.array(dtype=wp.vec3), xp: wp.array(dtype=wp.vec3), w: wp.array(dtype=float), mesh: wp.uint64,
           thick: float, mu: float, hit: wp.array(dtype=int)):
    i = wp.tid()
    if w[i] == 0.0:
        return
    p = x[i]
    q = wp.mesh_query_point_sign_normal(mesh, p, 0.25)
    if q.result:
        cp = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
        d = p - cp
        dist = wp.length(d)
        n = wp.mesh_eval_face_normal(mesh, q.face)
        if dist > 1.0e-9:
            n = d / dist * q.sign
        pen = thick - dist * q.sign
        if pen > 0.0:
            p = p + n * pen
            # friction: remove part of the tangential motion this step
            mv = p - xp[i]
            tn = mv - n * wp.dot(mv, n)
            p = p - tn * mu
            x[i] = p
            hit[i] = 1


@wp.kernel
def k_self(x: wp.array(dtype=wp.vec3), w: wp.array(dtype=float), grid: wp.uint64, uv: wp.array(dtype=wp.vec2),
           piece: wp.array(dtype=int), root: wp.array(dtype=int), r: float,
           dx: wp.array(dtype=wp.vec3), cnt: wp.array(dtype=float)):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    p = x[i]
    q = wp.hash_grid_query(grid, p, r)
    j = int(0)
    while wp.hash_grid_query_next(q, j):
        if j == i or root[j] == root[i]:
            continue
        if piece[j] == piece[i]:
            if wp.length(uv[i] - uv[j]) < 2.5 * r:
                continue
        d = p - x[j]
        L = wp.length(d)
        if L < r and L > 1.0e-9:
            wi = w[i]
            wj = w[j]
            if wi + wj > 0.0:
                corr = d / L * ((r - L) * wi / (wi + wj)) * 0.5
                wp.atomic_add(dx, i, corr)
                wp.atomic_add(cnt, i, 1.0)


@wp.kernel
def k_vel(x: wp.array(dtype=wp.vec3), xp: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), dt: float,
          vmax: float):
    i = wp.tid()
    vi = (x[i] - xp[i]) / dt
    s = wp.length(vi)
    if s > vmax:
        vi = vi * (vmax / s)
    v[i] = vi


class Cloth:
    def __init__(self, X, uv, tris, piece_id, pairs, body_V, body_F, density=0.9, device=None, log=print, glue=None,
                 glue_gap=0.0045, stiff=None):
        """X (n,3) start positions [m]; uv (n,2) rest pattern coords [m]; tris (m,3); pairs (k,2) seam vertex pairs.
        density kg/m^2 (1.3 mm steerhide ~ 0.9)."""
        self.dev = wp.get_device(device)
        self.log = log
        n = len(X)
        # masses from triangle areas
        A = 0.5 * np.abs(np.cross(np.c_[uv[tris[:, 1]] - uv[tris[:, 0]], np.zeros(len(tris))],
                                  np.c_[uv[tris[:, 2]] - uv[tris[:, 0]], np.zeros(len(tris))])[:, 2])
        m = np.zeros(n)
        for c in range(3):
            np.add.at(m, tris[:, c], A * density / 3)
        winv = np.where(m > 0, 1.0 / np.maximum(m, 1e-9), 0.0)
        # stretch edges + bending (cross) edges, rest lengths from the flat pattern
        E = np.sort(np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
        Eu, inv = np.unique(E, axis=0, return_inverse=True)
        tri_of_edge = {}
        opp = np.vstack([tris[:, 2], tris[:, 0], tris[:, 1]]).T.reshape(-1)  # opposite vertex per (tri, edge)
        opp = np.concatenate([tris[:, 2], tris[:, 0], tris[:, 1]])
        for k, e in enumerate(inv.reshape(-1)):
            tri_of_edge.setdefault(int(e), []).append(int(opp[k]))
        B = np.array([v for v in tri_of_edge.values() if len(v) == 2], int).reshape(-1, 2)
        rest_s = np.linalg.norm(uv[Eu[:, 0]] - uv[Eu[:, 1]], axis=1)
        rest_b = np.linalg.norm(uv[B[:, 0]] - uv[B[:, 1]], axis=1)
        # seams: union-find roots so sewn vertices skip each other in self-collision
        par = np.arange(n)

        def f(a):
            while par[a] != a:
                par[a] = par[par[a]]
                a = par[a]
            return a
        for a, b in pairs:
            ra, rb = f(a), f(b)
            if ra != rb:
                par[ra] = rb
        root = np.array([f(a) for a in range(n)])
        self.pairs = pairs
        seam0 = np.linalg.norm(X[pairs[:, 0]] - X[pairs[:, 1]], axis=1) if len(pairs) else np.zeros(0)
        self.n, self.nE, self.nB, self.nS = n, len(Eu), len(B), len(pairs)
        W = lambda a, dt: wp.array(a, dtype=dt, device=self.dev)
        self.x = W(X.astype(np.float32), wp.vec3)
        self.xp = W(X.astype(np.float32), wp.vec3)
        self.v = wp.zeros(n, dtype=wp.vec3, device=self.dev)
        self.w = W(winv.astype(np.float32), float)
        self.E = W(Eu.astype(np.int32), wp.vec2i)
        self.rs = W(rest_s.astype(np.float32), float)
        self.B = W(B.astype(np.int32), wp.vec2i)
        self.rb = W(rest_b.astype(np.float32), float)
        self.S = W(pairs.astype(np.int32), wp.vec2i)
        self.r0 = W(seam0.astype(np.float32), float)
        self.uv = W(uv.astype(np.float32), wp.vec2)
        self.piece = W(piece_id.astype(np.int32), int)
        self.root = W(root.astype(np.int32), int)
        self.dx = wp.zeros(n, dtype=wp.vec3, device=self.dev)
        self.cnt = wp.zeros(n, dtype=float, device=self.dev)
        self.hit = wp.zeros(n, dtype=int, device=self.dev)
        self.bV = W(body_V.astype(np.float32), wp.vec3)
        self.bF = W(body_F.reshape(-1).astype(np.int32), int)
        self.mesh = wp.Mesh(points=self.bV, indices=self.bF)
        self.grid = wp.HashGrid(64, 64, 64, device=self.dev)
        self.rest_s, self.tris, self.uv_np = rest_s, tris, uv
        # one constraint list (stretch 0, bend 1, seam 2), greedy-coloured so each colour runs in parallel as true
        # Gauss-Seidel (no two constraints in a colour share a particle)
        # glue (stitched-on patches, under-layers): held at one layer's gap, so the two hides don't fight collision
        glue = np.zeros((0, 2), int) if glue is None or len(glue) == 0 else np.asarray(glue, int)
        C = np.vstack([Eu, B, pairs.reshape(-1, 2), glue]).astype(np.int64)
        kind = np.r_[np.zeros(len(Eu)), np.ones(len(B)), np.full(len(pairs), 2), np.full(len(glue), 3)].astype(np.int32)
        rest = np.r_[rest_s, rest_b, seam0, np.full(len(glue), glue_gap)].astype(np.float32)
        # per-piece stiffness factor (1 = the cloth's own): compliance scale 1/factor on bend + compression
        sp = np.ones(n) if stiff is None else np.asarray(stiff, float)
        fe = np.minimum(sp[Eu[:, 0]], sp[Eu[:, 1]])
        fb = np.minimum(sp[B[:, 0]], sp[B[:, 1]]) if len(B) else np.zeros(0)
        scale = np.r_[1.0 / fe, 1.0 / fb, np.ones(len(pairs)), np.ones(len(glue))].astype(np.float32)
        col = _greedy_colour(C, n)
        order = np.lexsort((kind, col))
        C, kind, rest, col, scale = C[order], kind[order], rest[order], col[order], scale[order]
        self.colour_ranges = []
        for c in range(col.max() + 1):
            idx = np.nonzero(col == c)[0]
            self.colour_ranges.append((int(idx[0]), int(len(idx))))
        self.Cc = W(C.astype(np.int32), wp.vec2i)
        self.Ck = W(kind, int)
        self.Cr = W(rest, float)
        self.Cs = W(scale, float)
        self.lam = wp.zeros(len(C), dtype=float, device=self.dev)
        log("colouring", colours=len(self.colour_ranges), constraints=len(C))
        log("cloth", particles=n, stretch=len(Eu), bend=len(B), seams=len(pairs), mass_kg=round(float(m.sum()), 3),
            device=str(self.dev))

    def pin(self, idx):
        """Make particles kinematic (inverse mass 0): a cinched belt or waistband that holds the garment up."""
        w = self.w.numpy()
        w[np.asarray(idx, int)] = 0.0
        self.w = wp.array(w.astype(np.float32), dtype=float, device=self.dev)
        v = self.v.numpy()
        v[np.asarray(idx, int)] = 0.0
        self.v = wp.array(v.astype(np.float32), dtype=wp.vec3, device=self.dev)

    def set_body(self, V):
        self.bV.assign(V.astype(np.float32))
        self.mesh.refit()

    def step(self, dt, iters=20, seam_scale=0.0, seam_alpha=1e-9, g=-9.81, damp=0.99, stretch_alpha=1e-8,
             bend_alpha=5e-5, compress_alpha=2e-6, self_r=0.004, thick=0.004, mu=0.4, self_every=4):
        """compliances (m/N style XPBD alphas, divided by dt^2 inside). Leather: stiff stretch, soft compression."""
        n = self.n
        wp.launch(k_predict, n, [self.x, self.v, self.w, self.xp, wp.vec3(0, 0, g), dt, damp], device=self.dev)
        a_s = stretch_alpha / dt ** 2
        a_b = bend_alpha / dt ** 2
        a_m = seam_alpha / dt ** 2
        a_c = compress_alpha / dt ** 2
        self.lam.zero_()
        for it in range(iters):
            for start, cnt in self.colour_ranges:
                wp.launch(k_dist_gs, cnt, [self.x, self.w, self.Cc, self.Cr, self.Ck, self.lam, start, a_s, a_b, a_m,
                                           a_c, seam_scale, self.Cs], device=self.dev)
            if self_r > 0 and it % self_every == 0:
                self.grid.build(self.x, self_r)
                wp.launch(k_self, n, [self.x, self.w, self.grid.id, self.uv, self.piece, self.root, self_r, self.dx,
                                      self.cnt], device=self.dev)
                wp.launch(k_apply, n, [self.x, self.dx, self.cnt, 1.0], device=self.dev)
            wp.launch(k_body, n, [self.x, self.xp, self.w, self.mesh.id, thick, mu if it == iters - 1 else 0.0,
                                  self.hit], device=self.dev)
        wp.launch(k_vel, n, [self.x, self.xp, self.v, dt, 3.0], device=self.dev)

    def positions(self):
        return self.x.numpy().astype(np.float64)

    def metrics(self, body_V=None, body_F=None):
        X = self.positions()
        E = self.E.numpy()
        L = np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1)
        ok = self.rest_s > 0.002  # sliver edges (< 2 mm rest, e.g. at cut corners) make the ratio meaningless
        strain = (L / np.maximum(self.rest_s, 1e-9) - 1)[ok] if ok.any() else np.zeros(1)
        S = self.pairs
        gap = np.linalg.norm(X[S[:, 0]] - X[S[:, 1]], axis=1) if len(S) else np.zeros(1)
        out = dict(stretch_p95=float(np.percentile(strain, 95)), stretch_max=float(strain.max()),
                   seam_gap_max_mm=float(gap.max() * 1000), seam_gap_p95_mm=float(np.percentile(gap, 95) * 1000))
        # penetration into the body
        d = wp.zeros(self.n, dtype=float, device=self.dev)
        wp.launch(k_depth, self.n, [self.x, self.mesh.id, d], device=self.dev)
        dd = d.numpy()
        out["penetrating_pct"] = float((dd < -0.0005).mean() * 100)
        out["penetration_max_mm"] = float(max(0.0, -dd.min()) * 1000)
        return out


def _greedy_colour(C, n):
    """Greedy edge colouring: constraints sharing a particle get different colours."""
    col = np.full(len(C), -1, np.int32)
    used = [set() for _ in range(n)]
    deg = np.bincount(C.reshape(-1), minlength=n)
    order = np.argsort(-(deg[C[:, 0]] + deg[C[:, 1]]), kind="stable")
    for k in order:
        i, j = C[k]
        u = used[i] | used[j]
        c = 0
        while c in u:
            c += 1
        col[k] = c
        used[i].add(c)
        used[j].add(c)
    return col


@wp.kernel
def k_depth(x: wp.array(dtype=wp.vec3), mesh: wp.uint64, out: wp.array(dtype=float)):
    i = wp.tid()
    q = wp.mesh_query_point_sign_normal(mesh, x[i], 0.5)
    if q.result:
        cp = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
        out[i] = wp.length(x[i] - cp) * q.sign
    else:
        out[i] = 1.0

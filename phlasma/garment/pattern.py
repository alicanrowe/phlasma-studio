"""2D pattern pieces -> seam-aware boundary sampling -> triangle meshes, plus seam vertex pairs.

Job format (same as boots-jackets-v2 clo/jobs/*.json): patterns[i].points = [[x_mm, y_mm, flag], ...] where flag 0
marks a corner (line break); line k runs from corner k to corner k+1 (wrapping). seams[j] = {a, la, a0, a1, b, lb,
b0, b1, kind}: pattern a line la from fraction a0 to a1 is sewn to pattern b line lb from b0 to b1.
"""
import numpy as np
import triangle as tr


def _arclen(P):
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    return d


def _interp(P, d, s):
    return np.c_[np.interp(s, d, P[:, 0]), np.interp(s, d, P[:, 1])]


class Piece:
    def __init__(self, idx, spec):
        self.idx, self.spec = idx, spec
        self.name = spec["name"]
        pts = np.array([[q[0], q[1]] for q in spec["points"]], float) / 1000.0  # metres
        flags = [int(q[2]) for q in spec["points"]]
        self.pts = pts
        corners = [i for i, f in enumerate(flags) if f == 0]
        self.lines = []
        n = len(pts)
        for k, c in enumerate(corners):
            e = corners[(k + 1) % len(corners)]
            ids = list(range(c, (e if e > c else e + n) + 1))
            self.lines.append(pts[[i % n for i in ids]])
        self.area = 0.5 * abs(np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1]))
        self.ccw = np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1]) > 0
        self.breaks = [set([0.0, 1.0]) for _ in self.lines]  # seam endpoints per line (fractions)
        self.seg_n = {}  # (line, f0, f1) -> forced segment count

    def line_len(self, k):
        return _arclen(self.lines[k])[-1]


def build(job, h=0.012, log=print):
    """Returns dict with per-piece meshes (uv 2D, tris, boundary) and seam vertex pairs (global indices)."""
    P = [Piece(i, s) for i, s in enumerate(job["patterns"])]
    seams = job.get("seams", [])
    # 1. register seam endpoints as breaks
    for s in seams:
        for p, l, f0, f1 in ((s["a"], s["la"], s["a0"], s["a1"]), (s["b"], s["lb"], s["b0"], s["b1"])):
            if l >= len(P[p].lines):
                raise ValueError(f"seam refers to line {l} of {P[p].name} which has {len(P[p].lines)} lines")
            P[p].breaks[l].update([round(min(max(f0, 0), 1), 6), round(min(max(f1, 0), 1), 6)])
    # 2. a seam may cover several sub-intervals of a line once other seams add breaks; give each seam one count
    seam_n = []
    report = []
    for s in seams:
        La = abs(s["a1"] - s["a0"]) * P[s["a"]].line_len(s["la"])
        Lb = abs(s["b1"] - s["b0"]) * P[s["b"]].line_len(s["lb"])
        seam_n.append(max(1, int(round(max(La, Lb) / h))))
        report.append(dict(a=P[s["a"]].name, b=P[s["b"]].name, La_mm=round(La * 1000, 1), Lb_mm=round(Lb * 1000, 1),
                           kind=s.get("kind")))
    # 3. sample each line: between consecutive breaks, uniform in arc length. Points at fraction f stored by key.
    samples = []  # per piece: list of (line, frac) for boundary vertices in order
    for p in P:
        seq = []
        for k, line in enumerate(p.lines):
            d = _arclen(line)
            L = d[-1]
            br = sorted(p.breaks[k])
            for i in range(len(br) - 1):
                f0, f1 = br[i], br[i + 1]
                if f1 - f0 < 1e-6:
                    continue
                n = max(1, int(round((f1 - f0) * L / h)))
                # a seam covering this interval sets the density so both sides have the same spacing in parameter
                for j, s in enumerate(seams):
                    for (pp, ll, a0, a1, other) in ((s["a"], s["la"], s["a0"], s["a1"], "b"), (s["b"], s["lb"], s["b0"], s["b1"], "a")):
                        lo, hi = min(a0, a1), max(a0, a1)
                        if pp == p.idx and ll == k and lo - 1e-6 <= f0 and f1 <= hi + 1e-6 and hi - lo > 1e-9:
                            n = max(1, int(round(seam_n[j] * (f1 - f0) / (hi - lo))))
                for t in np.linspace(f0, f1, n + 1)[:-1]:
                    seq.append((k, float(t)))
        samples.append(seq)
    # make the counts symmetric: both sides of a seam must have equal vertex counts over the seam span.
    # (Sub-interval counts are proportional to the parameter span, so they match when both sides share break structure;
    # otherwise we fall back to param-nearest pairing and report the mismatch.)
    meshes, offset = [], 0
    for p, seq in zip(P, samples):
        B = []
        for k, t in seq:
            d = _arclen(p.lines[k])
            B.append(_interp(p.lines[k], d, np.array([t * d[-1]]))[0])
        B = np.array(B)
        nb = len(B)
        segs = np.c_[np.arange(nb), (np.arange(nb) + 1) % nb]
        area = (h ** 2) * 0.5
        T = tr.triangulate(dict(vertices=B, segments=segs), f"pq28a{area:.8f}YQ")
        V, F = T["vertices"], T["triangles"]
        # orient triangles CCW in uv
        a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        cr = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        F[cr < 0] = F[cr < 0][:, ::-1]
        meshes.append(dict(name=p.name, piece=p, uv=V, tris=F, nb=nb, seq=seq, offset=offset))
        offset += len(V)
    # 4. seam vertex pairs
    pairs, stats = [], []
    for j, s in enumerate(seams):
        A, Bm = meshes[s["a"]], meshes[s["b"]]
        ia = _span(A, s["la"], s["a0"], s["a1"])
        ib = _span(Bm, s["lb"], s["b0"], s["b1"])
        ta = np.linspace(0, 1, len(ia)) if len(ia) > 1 else np.zeros(1)
        tb = np.linspace(0, 1, len(ib)) if len(ib) > 1 else np.zeros(1)
        if len(ia) == len(ib):
            pr = list(zip(ia, ib))
        else:  # pair each vertex of the denser side to the nearest param on the other
            big, small, tbig, tsmall, flip = (ia, ib, ta, tb, False) if len(ia) >= len(ib) else (ib, ia, tb, ta, True)
            pr = []
            for v, t in zip(big, tbig):
                w = small[int(np.argmin(np.abs(tsmall - t)))]
                pr.append((w, v) if flip else (v, w))
        pairs += [(A["offset"] + x, Bm["offset"] + y) for x, y in pr]
        stats.append(dict(report[j], na=len(ia), nb=len(ib)))
    return dict(meshes=meshes, pairs=np.array(pairs, int), seam_report=stats)


def _span(M, line, f0, f1):
    """Boundary vertex indices (local) on `line` from fraction f0 to f1, in that direction, endpoints included."""
    seq = M["seq"]
    lo, hi = min(f0, f1), max(f0, f1)
    ids = [i for i, (k, t) in enumerate(seq) if k == line and lo - 1e-6 <= t <= hi + 1e-6]
    # endpoint f=1.0 of this line is the first vertex of the next line
    if hi >= 1 - 1e-6:
        nxt = [i for i, (k, t) in enumerate(seq) if k == (line + 1) % len(M["piece"].lines) and t < 1e-6]
        ids += nxt
    ids.sort(key=lambda i: seq[i][1] if seq[i][0] == line else 1.0)
    if f0 > f1:
        ids = ids[::-1]
    return ids

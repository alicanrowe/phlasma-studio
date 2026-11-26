"""verb `drape`: pattern job (clo/jobs format) -> sewn, simulated garment on the mannequin -> OBJ/GLB + stills + checks."""
import json
import os
import time

import numpy as np

from . import mannequin as mq
from . import pattern, place
from .sim_xpbd import Cloth

DEFAULTS = dict(h=0.012, pose_from=50.0, pose_to=15.0, dt=1 / 480, iters=4, sew_s=2.0, pose_s=0.8, settle_s=0.6,
                density=0.95, render=True, views=["front", "34", "side", "back"], samples=96, size=[1080, 1350],
                device=None, color=None, mannequin_res=0.7)

LEATHER_RGB = {"oxblood": (0.085, 0.011, 0.013), "black": (0.014, 0.013, 0.013), "natural": (0.40, 0.21, 0.10)}


def _colour(job, o):
    if o.get("color"):
        return tuple(o["color"])
    lth = " ".join(p.get("leather", "") for p in job["patterns"]).lower()
    for k in ("oxblood", "black", "natural"):
        if k in lth:
            return LEATHER_RGB[k]
    return LEATHER_RGB["black"]


def _rgb(c):
    """(r,g,b) linear floats, or '#rrggbb' sRGB hex -> linear."""
    if isinstance(c, str):
        h = c.lstrip("#")
        v = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
        return tuple(x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in v)
    return tuple(c)


def normal_field(X, tris, pid, bV):
    """Outward unit normals of the draped cloth surface, queryable anywhere near it (IDW of vertex normals).
    Oriented per piece by majority vote against the nearest-body direction, so arms next to a side seam
    can't flip them."""
    import trimesh
    from scipy.spatial import cKDTree
    VN = np.asarray(trimesh.Trimesh(X, tris, process=False).vertex_normals).copy()
    out = X - bV[cKDTree(bV).query(X)[1]]
    flip = (VN * out).sum(1) < 0
    for i in np.unique(pid):
        m = pid == i
        if flip[m].mean() > 0.5:
            VN[m] *= -1
    gt = cKDTree(X)

    def nf(P):
        P = np.atleast_2d(P)
        d, j = gt.query(P, k=4)
        w = 1.0 / np.maximum(d, 1e-5) ** 2
        n = (VN[j] * w[..., None]).sum(1)
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    return nf


def _at_point(X, tris, M, at, fallback):
    """the draped position of a pattern point: barycentric inside the piece triangle that holds it (a vertex
    snap moves it by up to half the mesh spacing, enough to stack pencils 8 mm apart onto each other)"""
    o, n = M["offset"], len(M["uv"])
    T = tris[np.all((tris >= o) & (tris < o + n), axis=1)]
    if not len(T):
        return fallback
    A, B, C = (M["uv"][T[:, k] - o] for k in range(3))
    v0, v1, v2 = B - A, C - A, at - A
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    ok = np.abs(den) > 1e-12
    b1 = np.where(ok, (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / np.where(ok, den, 1), -1)
    b2 = np.where(ok, (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / np.where(ok, den, 1), -1)
    b0 = 1 - b1 - b2
    inside = (b0 >= -1e-6) & (b1 >= -1e-6) & (b2 >= -1e-6)
    if not inside.any():
        return fallback
    i = int(np.flatnonzero(inside)[0])
    return b0[i] * X[T[i, 0]] + b1[i] * X[T[i, 1]] + b2[i] * X[T[i, 2]]


def topstitch(P, nf, rows=(-0.003, 0.003), lift=0.0021, on=0.0035, off=0.002):
    """Dashed stitch rows either side of a seam/edge polyline, sitting on the cloth surface."""
    P = np.asarray(P, float)
    if len(P) < 2:
        return []
    N = nf(P)
    T = np.gradient(P, axis=0)
    S = np.cross(T, N)
    S /= np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-9)
    out = []
    for r in rows:
        Q = P + N * lift + S * r
        d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))]
        s0 = 0.002
        while s0 + on < d[-1]:
            ss = np.linspace(s0, s0 + on, 3)
            out.append(np.c_[[np.interp(ss, d, Q[:, k]) for k in range(3)]].T)
            s0 += on + off
    return out


def weld(X, F, pairs, return_mask=False):
    """Merge sewn vertex pairs into one vertex (at their mean) so seams render as one continuous hide."""
    n = len(X)
    par = np.arange(n)

    def f(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    for a, b in pairs:
        ra, rb = f(a), f(b)
        if ra != rb:
            par[max(ra, rb)] = min(ra, rb)
    root = np.array([f(a) for a in range(n)])
    uniq, inv = np.unique(root, return_inverse=True)
    V = np.zeros((len(uniq), 3))
    np.add.at(V, inv, X)
    V /= np.bincount(inv)[:, None]
    F2 = inv[F]
    ok = (F2[:, 0] != F2[:, 1]) & (F2[:, 1] != F2[:, 2]) & (F2[:, 0] != F2[:, 2])
    if return_mask:
        return V, F2[ok], ok
    return V, F2[ok]


def seam_lines(X, pairs, seam_report, lift=0.0016):
    """One polyline per seam (mean of the two sides), lifted off the surface a touch; for welt/topstitch geometry."""
    lines, k = [], 0
    for s in seam_report:
        n = max(s["na"], s["nb"])
        pr = pairs[k:k + n]
        k += n
        if n < 2:
            continue
        P = (X[pr[:, 0]] + X[pr[:, 1]]) / 2
        if len(P) > 6:  # light 5-tap smoothing so a seam over a fold doesn't read as a torn edge
            d1, d2 = P[1:-1] - P[:-2], P[2:] - P[1:-1]
            cos = (d1 * d2).sum(1) / np.maximum(np.linalg.norm(d1, axis=1) * np.linalg.norm(d2, axis=1), 1e-12)
            keep = np.r_[True, cos < 0.8, True]  # corners (a dovetail shoulder) stay sharp
            keep = keep | np.r_[False, keep[:-1]] | np.r_[keep[1:], False]
            Q = P.copy()
            for _ in range(2):
                S = Q.copy()
                S[2:-2] = (Q[:-4] + Q[1:-3] + Q[2:-2] + Q[3:-1] + Q[4:]) / 5
                Q = np.where(keep[:, None], Q, S)
            P = Q
        lines.append({"pts": P, "kind": s.get("kind"), "nostitch": "collar" in s["a"] + s["b"], "a": s["a"]})
    return lines


def glue_pairs(G, X, pid, log, every=3, reach=0.07):
    """Pieces with no seam (patches such as elbow cups) or on an under-layer are stitched flat to the nearest piece
    they sit on, so they ride the garment instead of falling off."""
    from scipy.spatial import cKDTree
    seamed = set()
    for s in G["seam_report"]:
        seamed.update([s["a"], s["b"]])
    out = []
    for i, M in enumerate(G["meshes"]):
        lay = int(M["piece"].spec.get("layer", 0) or 0)
        if M["name"] in seamed and lay == 0:
            continue
        own = np.nonzero(pid == i)[0]
        other = np.nonzero(pid != i)[0]
        tree = cKDTree(X[other])
        idx = own[::every] if lay != 0 else own  # outside patches (pockets) are stitched flat all over
        d, j = tree.query(X[idx])
        ok = d < reach
        out += list(zip(idx[ok], other[j[ok]]))
        log("glue", piece=M["name"], layer=lay, pairs=int(ok.sum()))
    return np.array(out, int).reshape(-1, 2)


def _part_texture(gm, key):
    """(uv, png path) for a textured CAD part, so the render shows its real grain/etch; None when untextured"""
    import tempfile
    m = getattr(gm.visual, "material", None)
    uv = getattr(gm.visual, "uv", None)
    img = None if m is None else (getattr(m, "baseColorTexture", None) or getattr(m, "image", None))
    if img is None or uv is None or len(uv) != len(gm.vertices):
        return None
    d = os.path.join(tempfile.gettempdir(), "phlasma_tex")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "".join(c if c.isalnum() else "_" for c in key) + ".png")
    if not os.path.exists(path):
        img.convert("RGB").save(path)
    return np.asarray(uv, float), path


def _cut_keep(m, pnt, nrm):
    """keep the part of m on the +nrm side of a plane; untouched (texture kept) when it doesn't cross the plane"""
    s = (np.asarray(m.vertices) - pnt) @ nrm
    if (s >= 0).all():
        return m, True
    if (s < 0).all():
        return None, False
    for cap in (True, False):  # capping needs a watertight part; open CAD shells are cut without a cap
        try:
            r = m.slice_plane(pnt, nrm, cap=cap)
            if r is not None and len(r.faces):
                return r, False
        except Exception:
            pass
    keep = s >= 0  # last resort: drop the faces whose vertices are all inside
    f = np.asarray(m.faces)
    m = m.copy()
    m.update_faces(keep[f].any(1))
    m.remove_unreferenced_vertices()
    return m, False

def _part_color(gm):
    """a CAD part's base colour (0-1 sRGB): the texture's mean when it has one (wood, etched steel), else its factor"""
    m = getattr(gm.visual, "material", None)
    if m is None:
        return None
    img = getattr(m, "baseColorTexture", None)
    if img is None:
        img = getattr(m, "image", None)
    if img is not None:
        try:
            a = np.asarray(img.convert("RGB").resize((64, 64)), float) / 255.0
            return a.reshape(-1, 3).mean(0)
        except Exception:
            pass
    try:
        return np.asarray(m.main_color[:3], float) / 255.0
    except Exception:
        return None

def drape(job, out, log, **opts):
    """Sew and drape a 2D pattern job on the mannequin; outputs garment.obj, stills and seam/penetration/stretch checks."""
    o = {**DEFAULTS, **(job.get("sim") or {}), **opts}
    import warp as wp
    wp.init()
    dev = o["device"] or ("cuda" if wp.is_cuda_available() else "cpu")
    T = {}
    t = time.time()
    G = pattern.build(job, h=o["h"], log=log)
    for s in G["seam_report"]:
        if abs(s["La_mm"] - s["Lb_mm"]) > max(3.0, 0.04 * max(s["La_mm"], s["Lb_mm"])):
            log("seam_length_mismatch", "warn", **s)
    T["mesh"] = time.time() - t
    t = time.time()
    mq.STANCE_DEG = float(o.get("stance_deg", job.get("stance_deg", 0.0)) or 0.0)
    bV, bF = mq.build_mesh(o["mannequin_res"], o["pose_from"])
    n_body = len(bV)  # mannequin only (colliders added below are not drawn)
    W = mq.arm_weights(bV, o["pose_from"])
    grow = None
    if o.get("colliders"):  # static props the cloth rests on (tools in a pocket...): boxes grown out of the body
        import trimesh as _tm
        nb0 = len(bV)
        CV, CFs, anc = [], [], []
        for c in o["colliders"]:
            bx = _tm.creation.box(extents=c["size"])  # size = (radial, across, along) before rotation
            a = np.radians(c.get("rot_z", 0.0))
            R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
            v = bx.vertices @ R.T + np.asarray(c["at"], float)
            out = R @ np.array([1.0, 0, 0]) * (1 if c.get("outward", "+x") == "+x" else -1)
            A = np.asarray(c["at"], float) - out * c["size"][0] / 2
            CFs.append(bx.faces + nb0 + sum(len(x) for x in CV))
            CV.append(v)
            anc.append(np.repeat(A[None], len(v), 0))
        CV, anc = np.vstack(CV), np.vstack(anc)
        bF = np.vstack([bF] + CFs)
        W = {k: np.r_[w, np.zeros(len(CV))] for k, w in W.items()}
        grow = lambda f: np.vstack([bV[:nb0], anc + (CV - anc) * f])
        bV = grow(1.0)
    if o.get("under"):  # garments already draped underneath (layering a whole fit): static colliders grown out
        # from the body during the sew phase, each one offset outward by its leather plus the patches on it
        from scipy.spatial import cKDTree as _KD
        nb0 = len(bV)
        UV_, UF_ = [], []
        for u in o["under"]:
            d_ = np.load(u["state"])
            Xu, Fu = d_["X"], d_["tris"]
            nfu = normal_field(Xu, Fu, d_["piece"], bV[:nb0])
            Xu = Xu + nfu(Xu) * float(u.get("offset_mm", 6.0)) / 1000.0
            UF_.append(Fu + len(bV) + sum(len(x) for x in UV_))
            UV_.append(Xu)
            for mp in u.get("meshes") or []:  # rigid parts worn with it (tools in its pockets), already in place
                import trimesh as _tm
                mm_ = _tm.load(mp, force="mesh", process=False)
                grow_mm = float(u.get("mesh_offset_mm", 1.0)) / 1000.0
                Vm = np.asarray(mm_.vertices, float) + np.asarray(mm_.vertex_normals, float) * grow_mm
                UF_.append(np.asarray(mm_.faces) + len(bV) + sum(len(x) for x in UV_))
                UV_.append(Vm)
        UV_ = np.vstack(UV_)
        uanc = bV[:nb0][_KD(bV[:nb0]).query(UV_)[1]]
        base_grow = grow
        bF = np.vstack([bF] + UF_)
        W = {k: np.r_[w, np.zeros(len(UV_))] for k, w in W.items()}
        nprev = len(bV)
        def grow(f, _bg=base_grow, _n=nprev):
            head = _bg(f) if _bg else bV0[:_n]
            return np.vstack([head, uanc + (UV_ - uanc) * f])
        bV0 = bV
        bV = grow(1.0)
        log("under", garments=len(o["under"]), verts=int(len(UV_)))
    T["mannequin"] = time.time() - t
    t = time.time()
    Xs = place.place(G["meshes"], o["pose_from"], log=log)
    X = np.vstack(Xs)
    uv = np.vstack([M["uv"] for M in G["meshes"]])
    tris = np.vstack([M["tris"] + M["offset"] for M in G["meshes"]])
    pid = np.concatenate([np.full(len(M["uv"]), i) for i, M in enumerate(G["meshes"])])
    glue = glue_pairs(G, X, pid, log)
    if o.get("from_state"):  # re-render a finished drape (new colourway/lighting) without re-simulating
        X = np.load(o["from_state"])["X"]
    cl = Cloth(X, uv, tris, pid, G["pairs"], grow(0.05) if grow and not o.get("from_state") else bV, bF,
               density=o["density"], device=dev, log=log, glue=glue,
               stiff=np.array([float(G["meshes"][i]["piece"].spec.get("stiff", 1.0)) for i in pid]))
    T["setup"] = time.time() - t
    t = time.time()
    dt = o["dt"]
    n_sew, n_pose, n_settle = (int(o[k] / dt) for k in ("sew_s", "pose_s", "settle_s"))
    if o.get("from_state"):
        n_sew = n_pose = n_settle = 0
        cl.set_body(mq.pose(bV, W, o["pose_from"], o["pose_to"], o["pose_from"]))
    step = 0
    mk = {k: o[k] for k in ("bend_alpha", "stretch_alpha", "compress_alpha", "mu") if k in o}  # cloth stiffness
    pins = np.nonzero(np.isin(pid, [i for i, M in enumerate(G["meshes"]) if M["piece"].spec.get("pin")]))[0]
    gf = float(o.get("gravity_from", 0.9 if len(pins) else 0.5))  # share of the sew phase before gravity ramps in
    for i in range(n_sew):  # 1. sew: seam rest length shrinks to zero, low gravity, A-pose
        f = 1 - min(1.0, (i + 1) / (0.8 * n_sew))
        if len(pins) and i == int(gf * n_sew):  # seams closed: cinch the waistband where it sits
            cl.pin(pins)
            log("pin", particles=int(len(pins)))
        if grow and i % 8 == 0 and i <= 0.7 * n_sew:
            cl.set_body(grow(0.05 + 0.95 * min(1.0, i / (0.6 * n_sew))))
        cl.step(dt, o["iters"], seam_scale=f, g=-9.81 * min(1.0, max(0.0, (i - gf * n_sew) / ((1 - gf) * n_sew))), damp=0.95,
                self_r=0.0 if i < n_sew // 3 else 0.004, **mk)
        step += 1
        if i % 60 == 0:
            log("sew", step=i, of=n_sew)
    for i in range(n_pose):  # 2. arms come down to the photo pose while the cloth settles under full gravity
        a = o["pose_from"] + (o["pose_to"] - o["pose_from"]) * (0.5 - 0.5 * np.cos(np.pi * (i + 1) / n_pose))
        cl.set_body(mq.pose(bV, W, o["pose_from"], a, o["pose_from"]))
        cl.step(dt, o["iters"], seam_scale=0.0, g=-9.81, damp=0.985, **mk)
        if i % 60 == 0:
            log("pose", step=i, angle=round(float(a), 1))
    for i in range(n_settle):  # 3. settle
        cl.step(dt, o.get("settle_iters", 3 * o["iters"]), seam_scale=0.0, g=-9.81, damp=0.96, **mk)
    if dev.startswith("cuda") if isinstance(dev, str) else False:
        wp.synchronize()
    T["simulate"] = time.time() - t
    Xf = cl.positions()
    bVp = mq.pose(bV, W, o["pose_from"], o["pose_to"], o["pose_from"])
    checks_m = cl.metrics()
    checks = {
        "seams_closed": {"pass": checks_m["seam_gap_p95_mm"] < 3.0, "p95_mm": round(checks_m["seam_gap_p95_mm"], 2),
                         "max_mm": round(checks_m["seam_gap_max_mm"], 2)},
        "no_penetration": {"pass": checks_m["penetrating_pct"] < 0.5, "pct": round(checks_m["penetrating_pct"], 3),
                           "max_mm": round(checks_m["penetration_max_mm"], 2)},
        "stretch": {"pass": checks_m["stretch_p95"] < 0.03, "p95_pct": round(checks_m["stretch_p95"] * 100, 2),
                    "max_pct": round(checks_m["stretch_max"] * 100, 2)},
    }
    log("checks", **{k: v for k, v in checks.items()})
    # outputs
    import trimesh
    os.makedirs(out, exist_ok=True)
    gm = trimesh.Trimesh(Xf, tris, process=False)
    gm.export(os.path.join(out, "garment.obj"))
    trimesh.Trimesh(bVp, bF, process=False).export(os.path.join(out, "mannequin.obj"))
    np.savez_compressed(os.path.join(out, "garment_state.npz"), X=Xf, uv=uv, tris=tris, piece=pid, pairs=G["pairs"])
    files = {"garment_obj": "garment.obj", "mannequin_obj": "mannequin.obj", "state": "garment_state.npz"}
    if o["render"]:
        try:
            import bpy  # noqa: F401
        except ImportError:
            log("render_skipped", "warn", reason="bpy not installed (needs Python 3.13); sim outputs are still written")
            o["render"] = False
    if o["render"]:
        t = time.time()
        from phlasma.core.render import render_views
        col = _colour(job, o)
        Xr = Xf.copy()
        mat = o.get("material") or job.get("material") or "leather"
        thick = float(o.get("thickness_mm", 1.3 if mat == "leather" else 0.9)) / 1000.0
        nf = normal_field(Xf, tris, pid, bVp)
        if len(glue):  # stitched-on patches are laid flat on the hide under them, one hide thickness proud per layer
            from scipy.spatial import cKDTree
            btree = cKDTree(bVp)
            gl = set(pid[glue[:, 0]].tolist())
            base = np.nonzero(~np.isin(pid, list(gl)))[0]
            stree = cKDTree(Xf[base])
            for i, M in enumerate(G["meshes"]):
                if i not in gl:
                    continue
                lay = int(M["piece"].spec.get("layer", 0) or 0)
                v = np.nonzero(pid == i)[0]
                if lay >= 0:  # conform: smooth (inverse-distance) projection onto the surface below
                    d, j = stree.query(Xf[v], k=8)
                    w = 1.0 / np.maximum(d, 1e-4) ** 2
                    Xr[v] = (Xf[base][j] * w[..., None]).sum(1) / w.sum(1)[:, None]
                nrm = nf(Xr[v])
                Xr[v] += nrm * ((thick + 0.0004) * (lay + 1) + 0.0015 * lay if lay >= 0 else -0.0015)
        tones, tmat, trim = {}, {}, {}
        for k, v in (o.get("tones") or {}).items():  # "#hex" | rgb | {"color", "mat", "edge"}
            if isinstance(v, dict):
                tmat[k] = v.get("mat", mat)
                if v.get("edge"):
                    trim[k] = _rgb(v["edge"])
                v = v["color"]
            tones[k] = _rgb(v)
        hw_items, rigid = [], []  # rigid parts on/in the leather: (name, mesh, over-mask)
        btn = o.get("button_color")
        for k, bt in enumerate(job.get("buttons") or []):  # shank buttons at pattern points, sat on the cloth normal
            import trimesh as _tm
            names = [M["name"] for M in G["meshes"]]
            if bt["piece"] not in names:
                continue
            i = names.index(bt["piece"])
            M = G["meshes"][i]
            j = int(np.argmin(np.linalg.norm(M["uv"] - np.array(bt["at"]) / 1000.0, axis=1))) + M["offset"]
            pnt = Xr[j]
            nrm = nf(pnt)[0]
            r = bt.get("r", 0.0075)
            cyl = (_tm.creation.box(extents=bt["box"]) if bt.get("box") else
                   _tm.creation.cylinder(radius=r, height=0.003, sections=28))
            # frame on the cloth: z = outward normal, y = world up projected (so a box's 2nd extent is its height)
            up = np.array([0.0, 0.0, 1.0]) - nrm * nrm[2]
            up = up / max(np.linalg.norm(up), 1e-9) if np.linalg.norm(up) > 1e-6 else np.array([0.0, 1.0, 0.0])
            Tb = np.eye(4)
            Tb[:3, :3] = np.c_[np.cross(up, nrm), up, nrm]
            cyl.apply_transform(Tb)
            depth = (bt["box"][2] / 2 if bt.get("box") else 0.0015) + bt.get("gap", 0.0003)
            cyl.apply_translation(pnt + nrm * (thick + depth + bt.get("lift", 0.0)) + up * bt.get("drop", 0.0)
                                  + np.cross(up, nrm) * bt.get("shift", 0.0))
            rigid.append((f"button{k}/box", cyl.copy(), np.zeros(len(Xr), bool), len(hw_items), nrm.copy(), f"button{k}"))
            tn = bt.get("tone")
            if bt.get("color"):  # a fixed-colour part (tag, the top of a cigarette pack)
                hw_items.append({"name": f"hw{k}", "V": cyl.vertices, "F": cyl.faces, "mat": bt.get("mat", "clay"),
                              "color": _rgb(bt["color"]), "smooth": False, "frame": False})
            elif tn is not None:  # a leather piece (holster, tab) in one of the garment's leathers
                hw_items.append({"name": f"hw{k}", "V": cyl.vertices, "F": cyl.faces, "mat": tmat.get(tn, mat),
                              "color": tones.get(tn, col), "smooth": False, "frame": False})
            else:
                hw_items.append({"name": f"button{k}", "V": cyl.vertices, "F": cyl.faces, "mat": "metal",
                              "smooth": False, "color": _rgb(btn) if btn else (0.30, 0.20, 0.09), "frame": False})
        for k, hd in enumerate(job.get("held") or []):
            # a real tool held in a pocket/sleeve/holster: photo-matched CAD (Z up, longest axis +X) placed on the
            # cloth frame, its long axis along "dir", the tool point at_mm on its long axis at the pocket mouth.
            # seat: the named part (e.g. the blade) lies IN the pocket, on the hide under it; the cloth is then seated
            # round the whole tool (pressed in under it, lifted over it) and the clipping gate checks the result.
            import trimesh as _tm
            names = [M["name"] for M in G["meshes"]]
            if hd["piece"] not in names or not os.path.exists(hd["obj"]):
                log("held_skipped", "warn", piece=hd["piece"], obj=hd["obj"])
                continue
            i = names.index(hd["piece"])
            M = G["meshes"][i]
            j = int(np.argmin(np.linalg.norm(M["uv"] - np.array(hd["at"]) / 1000.0, axis=1))) + M["offset"]
            pnt = _at_point(Xr, tris, M, np.array(hd["at"]) / 1000.0, Xr[j])  # exact, not snapped to a vertex
            near = M["offset"] + np.flatnonzero(np.linalg.norm(Xr[M["offset"]:M["offset"] + len(M["uv"])] - pnt, axis=1) < 0.035)
            nrm = nf(Xr[near]).mean(0) if len(near) else nf(pnt)[0]
            nrm = nrm / max(np.linalg.norm(nrm), 1e-9)
            up = np.array([0.0, 0.0, 1.0]) - nrm * nrm[2]
            up = up / max(np.linalg.norm(up), 1e-9)
            sd = np.cross(up, nrm)
            ang = np.radians(float(hd.get("tilt_deg", 0.0)))
            sgn = {"down": 1.0, "up": -1.0}[hd.get("dir", "down")]
            dvec = np.cos(ang) * -up + np.sin(ang) * sd
            if hd.get("align", bool(hd.get("seat"))):  # follow the pocket as draped: its long axis, centred across it
                Pp = Xr[M["offset"]:M["offset"] + len(M["uv"])]
                cpp = Pp.mean(0)
                Cp = np.cov((Pp - cpp).T)
                Cp = Cp - np.outer(nrm, nrm) * (nrm @ Cp @ nrm)
                ax_p = np.linalg.eigh(Cp)[1][:, -1]
                ax_p = ax_p - nrm * (ax_p @ nrm)
                ax_p /= max(np.linalg.norm(ax_p), 1e-9)
                if ax_p @ dvec < 0:
                    ax_p = -ax_p
                dvec = np.cos(ang) * ax_p + np.sin(ang) * np.cross(ax_p, nrm)
                sv_ = np.cross(nrm, dvec)
                if hd.get("center"):  # one tool per pocket: centred across it (not for a strip of pencils)
                    pnt = pnt + sv_ * float((cpp - pnt) @ sv_)
                elif hd.get("center_ref") is not None:  # a bundle: shift every member so the bundle is centred
                    ref = _at_point(Xr, tris, M, np.array(hd["center_ref"]) / 1000.0, pnt)
                    pnt = pnt + sv_ * float((cpp - ref) @ sv_)
            svec = np.cross(nrm, dvec)
            roll = bool(hd.get("flip_side"))
            over_names = hd.get("over") or [hd["piece"]]
            over_ids = [ii for ii, MM in enumerate(G["meshes"]) if MM["name"] in over_names]
            over_mask = np.isin(pid, over_ids)
            seat_on = hd.get("seat")
            base_pt = pnt + nrm * thick
            if seat_on:  # the hide under the pocket: cast from the pocket mouth along -normal through the cloth below
                keepF = ~np.isin(pid[tris[:, 0]], over_ids)
                under = _tm.Trimesh(Xr, tris[keepF], process=False)
                loc, _, _ = under.ray.intersects_location([pnt + nrm * 0.002], [-nrm])
                if len(loc):
                    h = loc[np.argmin(np.linalg.norm(loc - pnt, axis=1))]
                    base_pt = h + nf(h)[0] * thick
            sc = _tm.load(hd["obj"], force="scene")
            parts = []
            for gm in sc.dump(concatenate=False):
                gname = gm.metadata.get("name") or gm.metadata.get("node") or "part"
                m = gm.copy()
                if hd["obj"].endswith(".glb"):
                    m.apply_transform(_tm.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
                Vt = np.asarray(m.vertices, float) * float(hd.get("scale", 0.001))
                parts.append((gname, gm, m, Vt))
            zall = np.vstack([p_[3] for p_ in parts])[:, 2]
            zmax_all = zall.max()
            def lz(Vt):
                return (zmax_all - Vt[:, 2]) if roll else Vt[:, 2]
            if seat_on:
                zs = [lz(Vt) for (gname, gm, m, Vt) in parts if seat_on in gname.lower()]
                z0 = min(z.min() for z in zs) if zs else 0.0
            else:
                z0 = 0.0
            lift = float(hd.get("lift", 0.0006 if seat_on else 0.0005))
            for gname, gm, m, Vt in parts:
                a = np.array([float(hd.get("at_mm", 0.0)) / 1000.0, float(hd.get("y0_mm", 0.0)) / 1000.0, 0.0])
                L = Vt - a
                if roll:
                    L[:, 1] = -L[:, 1]
                L2 = lz(Vt) - z0
                org = base_pt + nrm * lift if seat_on else pnt + nrm * (thick + lift)
                W = org + np.outer(sgn * L[:, 0], dvec) + np.outer(sgn * L[:, 1], svec) \
                    + np.outer(L2 * (-1 if hd.get("face_in") else 1), nrm)
                mm = _tm.Trimesh(W, m.faces, process=False)
                whole = True
                if not seat_on and hd.get("hide_below", True):  # legacy: cut at the mouth plane
                    mm, whole = _cut_keep(mm, pnt, -dvec)
                if mm is None or len(mm.faces) == 0:
                    continue
                # the job's "over" pieces lie over the whole tool wherever they meet it (a pencil's ferrule as much as
                # its cedar); a pad the tool rests on (a welt under a square's head) must not be listed as over
                ov = over_mask
                rigid.append((f"held{k}/{gname}", mm, ov, len(hw_items), nrm.copy(), hd.get("rest_with") or f"held{k}"))
                tx = _part_texture(mm, os.path.basename(hd["obj"]) + gname)
                low = gname.lower()
                kd = _part_color(gm)
                metal = any(t in low for t in ("blade", "brass", "steel", "stainless", "pvd", "bronze", "ferrule", "screw", "iron", "metal"))
                mf = getattr(getattr(gm.visual, "material", None), "metallicFactor", None)
                if mf is not None:  # a PBR part says for itself whether it is metal
                    metal = float(mf) >= 0.5
                if metal and (kd is None or float(np.mean(kd)) < 0.35):
                    kd = np.array([0.74, 0.75, 0.77])
                if "black" in low or "web" in low:
                    kd = np.array([0.06, 0.06, 0.065])
                col = tuple(float(c) for c in (kd if kd is not None else (0.6, 0.6, 0.6)))
                col = tuple(c ** 2.2 for c in col)
                hw_items.append({"name": f"held{k}_{gname}", "V": mm.vertices, "F": mm.faces,
                                 "mat": hd.get("mats", {}).get(gname, "metal" if metal else "clay"), "color": col,
                                 "smooth": False, "frame": False, **({"uv": tx[0], "image": tx[1]} if tx else {})})
        if rigid:  # the cloth gives way round every rigid part, then the hard clipping gate
            from . import contact
            if o.get("contact_dump"):  # inputs for tuning the seat offline
                import pickle
                pickle.dump({"names": [M_["name"] for M_ in G["meshes"]], "Xr": Xr.copy(), "tris": tris, "Xf": Xf, "pid": pid, "bVp": bVp, "thick": thick,
                             "rigid": [(r[0], np.asarray(r[1].vertices), np.asarray(r[1].faces), r[2], r[4], r[5]) for r in rigid]},
                            open(o["contact_dump"], "wb"))
            if o.get("seat", True):
                groups = {}
                for r in rigid:
                    groups.setdefault(r[5], []).append(r)  # a strip of pencils rests as one bundle
                for gk, rs in groups.items():  # each tool / button rests ON the cloth it isn't tucked under
                    up_, tl_ = contact.rest(Xr, nf, thick, [r[1] for r in rs], [~r[2] for r in rs], rs[0][4])
                    for r in rs:
                        hw_items[r[3]]["V"] = np.asarray(r[1].vertices).copy()
                    if up_ > 1e-5:
                        log("rest", part=gk, lift_mm=round(up_ * 1000, 2), tilt_deg=round(float(np.degrees(tl_)), 2))
                contact.seat(Xr, tris, nf, thick, [r[1] for r in rigid], [r[2] for r in rigid], log=log)
            rep = contact.clip_report(Xr, tris, nf, thick, [r[1] for r in rigid], [r[0] for r in rigid],
                                      tol=float(o.get("clip_tol_mm", 0.5)) / 1000.0)
            checks["clipping"] = rep
            log("clipping", **{k_: v_ for k_, v_ in rep.items() if k_ != "hits"}, hits=rep["hits"][:8])
            json.dump(rep, open(os.path.join(out, "clipping.json"), "w"), indent=1)
            if not rep["pass"] and not o.get("allow_clip"):
                raise RuntimeError(f"clipping gate failed (worst {rep['worst_mm']} mm): {rep['hits'][:4]}")
        WV, WF, okf = weld(Xr, tris, G["pairs"], return_mask=True)
        ftone = np.array([M["piece"].spec.get("tone", "") for M in G["meshes"]], object)[pid[tris[okf, 0]]]
        items = [{"name": "mannequin", "V": bVp, "F": bF, "mat": "mannequin", "frame": False}]
        for tn in sorted(set(ftone)):
            Fi = WF[ftone == tn]
            used, Fc = np.unique(Fi, return_inverse=True)
            items.append({"name": f"garment_{tn or 'main'}", "V": WV[used], "F": Fc.reshape(-1, 3), "mat": tmat.get(tn, mat),
                          "color": tones.get(tn, col), "thick": thick, "subsurf": 1, "rim_color": trim.get(tn)})
        ridge = tuple(c * 0.8 for c in tones.get("lower", col))
        stitch = []
        from scipy.spatial import cKDTree as _T
        btree2 = _T(bVp)
        ptone = {M["name"]: M["piece"].spec.get("tone", "") for M in G["meshes"]}
        for i, L in enumerate(seam_lines(Xf, G["pairs"], G["seam_report"])):
            rc = tuple(c * 0.8 for c in tones[ptone.get(L["a"])]) if ptone.get(L["a"]) in tones else ridge
            items.append({"name": f"seam{i}", "curve": L["pts"], "radius": 0.0016 if "welt" in (L["kind"] or "") else 0.0011,
                          "mat": mat, "color": rc, "frame": False})
            if o.get("stitch") and not L.get("nostitch"):
                stitch += topstitch(L["pts"], nf, lift=thick + 0.0006)
        if o.get("stitch"):
            glued = set(pid[glue[:, 0]].tolist()) if len(glue) else set()
            for i in glued:  # patches: stitch round their outline, 3 mm in
                M = G["meshes"][i]
                B = Xr[M["offset"] + np.arange(M["nb"])]
                c = B.mean(0)
                B = B + (c - B) / np.maximum(np.linalg.norm(c - B, axis=1, keepdims=True), 1e-9) * 0.003
                lay = max(0, int(M["piece"].spec.get("layer", 0) or 0))
                stitch += topstitch(np.vstack([B, B[:1]]), nf, rows=(0.0,), lift=thick + 0.0006)
            items.append({"name": "topstitch", "curve": stitch, "radius": 0.00045, "mat": "clay",
                          "color": _rgb(o["stitch"]), "frame": False})
        for bi, bl in enumerate(job.get("belts") or []):  # a strap riding on a band: ribbon between its long edges
            for M in G["meshes"]:
                if M["piece"].spec.get("base") not in bl["pieces"]:
                    continue
                Bi = M["offset"] + np.arange(M["nb"])
                ks = np.array([k for k, _ in M["seq"]])
                nl = int(ks.max()) + 1
                def edge(k):
                    idx = list(np.nonzero(ks == k)[0]) + [int(np.nonzero(ks == (k + 1) % nl)[0][0])]
                    P = Xr[Bi[idx]]
                    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
                    t = np.linspace(0, d[-1], 48)
                    return np.c_[[np.interp(t, d, P[:, c]) for c in range(3)]].T
                ll, ul = bl.get("low_line", 0), bl.get("up_line", 2)
                if M["piece"].spec.get("side") == "R":
                    ll, ul = (nl - 1 - ll) % nl, (nl - 1 - ul) % nl
                lo_e, up_e = edge(ll), edge(ul)[::-1]
                fr = bl.get("inset", 0.125)
                rows = [lo_e + (up_e - lo_e) * f for f in np.linspace(fr, 1 - fr, 4)]
                R = np.vstack(rows)
                nrm = nf(R)
                R = R + nrm * (thick + 0.0004)
                n = 48
                F = [(r * n + c, r * n + c + 1, (r + 1) * n + c + 1) for r in range(3) for c in range(n - 1)]
                F += [(r * n + c, (r + 1) * n + c + 1, (r + 1) * n + c) for r in range(3) for c in range(n - 1)]
                tn = bl.get("tone", "lower")
                items.append({"name": f"belt{bi}_{M['name']}", "V": R, "F": np.array(F), "mat": tmat.get(tn, mat),
                              "color": tones.get(tn, col), "thick": thick, "subsurf": 1, "rim_color": trim.get(tn),
                              "frame": False})
        for ln in job.get("lines") or []:  # stitched details drawn on a piece (pocket openings, fly): pattern mm
            for M in G["meshes"]:
                sp = M["piece"].spec
                if sp.get("base") != ln["piece"]:
                    continue
                P2 = np.asarray(ln["pts"], float) / 1000.0 * ([-1.0, 1.0] if sp.get("side") == "R" else [1.0, 1.0])
                d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P2, axis=0), axis=1))]
                tl = np.linspace(0, d[-1], max(2, int(d[-1] / 0.004)))
                P2 = np.c_[np.interp(tl, d, P2[:, 0]), np.interp(tl, d, P2[:, 1])]
                from scipy.spatial import cKDTree as _T2
                dd, jj = _T2(M["uv"]).query(P2, k=6)
                w = 1.0 / np.maximum(dd, 1e-5) ** 2
                P3 = (Xr[M["offset"] + jj] * w[..., None]).sum(1) / w.sum(1)[:, None]
                nrm = nf(P3)
                tn = sp.get("tone", "")
                if ln.get("ridge", True):
                    items.append({"name": f"line_{M['name']}_{len(items)}", "curve": P3 + nrm * (thick + 0.0003),
                                  "radius": 0.0012, "mat": tmat.get(tn, mat),
                                  "color": tuple(c * 0.55 for c in tones.get(tn, col)), "frame": False})
                if o.get("stitch"):
                    items.append({"name": f"lstitch_{M['name']}_{len(items)}", "curve": topstitch(
                        P3, nf, rows=tuple(ln.get("rows", (0.005,))), lift=thick + 0.0006),
                        "radius": 0.00045, "mat": "clay", "color": _rgb(o["stitch"]), "frame": False})
        for k, ins in enumerate(job.get("inserts") or []):
            # hidden hardware (magnet | stiffener | weight) sewn between two layers: it shows only as a soft dome in
            # the hide over it plus a saddle-stitched ring round it, never as a visible part on top
            names = [M["name"] for M in G["meshes"]]
            if ins["piece"] not in names:
                continue
            i = names.index(ins["piece"])
            M = G["meshes"][i]
            j = int(np.argmin(np.linalg.norm(M["uv"] - np.array(ins["at"]) / 1000.0, axis=1))) + M["offset"]
            pnt = Xr[j]
            # the pocket's plane, not one vertex: average the normals over the piece within 35 mm of the mouth
            near = M["offset"] + np.flatnonzero(np.linalg.norm(Xr[M["offset"]:M["offset"] + len(M["uv"])] - pnt, axis=1) < 0.035)
            nrm = nf(Xr[near]).mean(0) if len(near) else nf(pnt)[0]
            nrm = nrm / max(np.linalg.norm(nrm), 1e-9)
            up = np.array([0.0, 0.0, 1.0]) - nrm * nrm[2]
            up = up / max(np.linalg.norm(up), 1e-9)
            sd = np.cross(up, nrm)
            R = float(ins.get("d", 0.019)) / 2 + 0.002  # dome runs out 2 mm past the insert
            h = float(ins.get("h", 0.0012))
            base = pnt + nrm * thick  # outer face of the hide (render positions already carry the patch layers)
            nr, na = 10, 36
            rr = np.linspace(0, 1, nr)
            aa = np.linspace(0, 2 * np.pi, na, endpoint=False)
            Vd = [base + nrm * (h - 0.0003)]
            for r_ in rr[1:]:
                z = h * (1 - r_ ** 2) ** 2 - 0.0003  # smooth falloff, edge sinks just under the surface
                for a in aa:
                    Vd.append(base + (np.cos(a) * sd + np.sin(a) * up) * R * r_ + nrm * z)
            Fd = [(0, 1 + (q + 1) % na, 1 + q) for q in range(na)]
            for ri in range(nr - 2):
                o0, o1 = 1 + ri * na, 1 + (ri + 1) * na
                for q in range(na):
                    a_, b_ = o0 + q, o0 + (q + 1) % na
                    Fd += [(a_, b_, o1 + (q + 1) % na), (a_, o1 + (q + 1) % na, o1 + q)]
            # conform the dome to the hide under it (IDW onto that piece's render surface), so it is a displacement
            # of the surface, not a disc laid on it
            from scipy.spatial import cKDTree as _K
            pv = np.nonzero(pid == i)[0]
            kt = _K(Xr[pv])
            Vd = np.array(Vd)
            hgt = (Vd - base) @ nrm
            dd, jj = kt.query(Vd - np.outer(hgt, nrm), k=4)
            ww = 1.0 / np.maximum(dd, 1e-4) ** 2
            S = (Xr[pv][jj] * ww[..., None]).sum(1) / ww.sum(1)[:, None]
            Vd = S + nf(S) * (thick + hgt[:, None])
            tn = ins.get("tone", M["piece"].spec.get("tone"))
            items.append({"name": f"insert{k}", "V": Vd, "F": np.array(Fd), "mat": tmat.get(tn, mat),
                          "color": tones.get(tn, col), "smooth": True, "frame": False})
            if o.get("stitch") and ins.get("ring", 0.006):
                rs = float(ins.get("d", 0.019)) / 2 + float(ins.get("ring", 0.006))
                ring = np.array([base + (np.cos(a) * sd + np.sin(a) * up) * rs for a in np.linspace(0, 2 * np.pi, 72)])
                dd, jj = kt.query(ring, k=4)
                ww = 1.0 / np.maximum(dd, 1e-4) ** 2
                ring = (Xr[pv][jj] * ww[..., None]).sum(1) / ww.sum(1)[:, None]
                ring = ring + nf(ring) * thick  # on the hide, conformed like the dome
                items.append({"name": f"insert{k}_ring", "curve": topstitch(ring, nf, rows=(0.0,), lift=0.0004),
                              "radius": 0.00045, "mat": "clay", "color": _rgb(o["stitch"]), "frame": False})
        items += hw_items
        for k, it in enumerate(o.get("props") or []):  # simple extra meshes (buttons, straps, tools), already in place
            items.append({"name": f"prop{k}", **it, "frame": False})
        lo, hi = WV.min(0), WV.max(0)
        if o.get("frame", "garment") == "full":
            tgt, hgt = (0, 0, 1.02), 1.95
        else:  # product framing: the garment fills the frame with ~9% margin (PHOTO_STANDARD)
            tgt, hgt = (lo + hi) / 2, (hi[2] - lo[2]) * 1.2
        if o.get("items_dump"):  # the render scene as data, so several garments can be layered into one fit render
            import pickle
            with open(o["items_dump"], "wb") as fh:
                pickle.dump({"items": items, "target": tgt, "height": hgt, "n_body": n_body}, fh)
        if o.get("items_only"):
            return {"outputs": files, "checks": checks, "metrics": {}, "seams": G["seam_report"]}
        imgs = render_views(items, out, o["views"], size=tuple(o["size"]), samples=o["samples"],
                            target=tgt, height=hgt, device=None if dev != "cpu" else "cpu", log=log,
                            floor=o.get("frame", "garment") == "full")
        files["views"] = [os.path.basename(p) for p in imgs]
        T["render"] = time.time() - t
    return {"outputs": files, "checks": checks, "metrics": {"seconds": {k: round(v, 2) for k, v in T.items()},
            "particles": cl.n, "triangles": int(len(tris)), "seam_pairs": int(len(G["pairs"])), "device": str(dev),
            "steps": n_sew + n_pose + n_settle}, "seams": G["seam_report"]}

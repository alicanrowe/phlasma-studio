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


def weld(X, F, pairs):
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
        lines.append({"pts": P, "kind": s.get("kind")})
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
        idx = own[::every]
        d, j = tree.query(X[idx])
        ok = d < reach
        out += list(zip(idx[ok], other[j[ok]]))
        log("glue", piece=M["name"], layer=lay, pairs=int(ok.sum()))
    return np.array(out, int).reshape(-1, 2)


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
    bV, bF = mq.build_mesh(o["mannequin_res"], o["pose_from"])
    W = mq.arm_weights(bV, o["pose_from"])
    T["mannequin"] = time.time() - t
    t = time.time()
    Xs = place.place(G["meshes"], o["pose_from"], log=log)
    X = np.vstack(Xs)
    uv = np.vstack([M["uv"] for M in G["meshes"]])
    tris = np.vstack([M["tris"] + M["offset"] for M in G["meshes"]])
    pid = np.concatenate([np.full(len(M["uv"]), i) for i, M in enumerate(G["meshes"])])
    glue = glue_pairs(G, X, pid, log)
    cl = Cloth(X, uv, tris, pid, G["pairs"], bV, bF, density=o["density"], device=dev, log=log, glue=glue)
    T["setup"] = time.time() - t
    t = time.time()
    dt = o["dt"]
    n_sew, n_pose, n_settle = (int(o[k] / dt) for k in ("sew_s", "pose_s", "settle_s"))
    step = 0
    for i in range(n_sew):  # 1. sew: seam rest length shrinks to zero, low gravity, A-pose
        f = 1 - min(1.0, (i + 1) / (0.8 * n_sew))
        cl.step(dt, o["iters"], seam_scale=f, g=-9.81 * min(1.0, max(0.0, (i - 0.5 * n_sew) / (0.5 * n_sew))), damp=0.95,
                self_r=0.0 if i < n_sew // 3 else 0.004)
        step += 1
        if i % 60 == 0:
            log("sew", step=i, of=n_sew)
    for i in range(n_pose):  # 2. arms come down to the photo pose while the cloth settles under full gravity
        a = o["pose_from"] + (o["pose_to"] - o["pose_from"]) * (0.5 - 0.5 * np.cos(np.pi * (i + 1) / n_pose))
        cl.set_body(mq.pose(bV, W, o["pose_from"], a, o["pose_from"]))
        cl.step(dt, o["iters"], seam_scale=0.0, g=-9.81, damp=0.985)
        if i % 60 == 0:
            log("pose", step=i, angle=round(float(a), 1))
    for i in range(n_settle):  # 3. settle
        cl.step(dt, o.get("settle_iters", 3 * o["iters"]), seam_scale=0.0, g=-9.81, damp=0.96)
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
        if len(glue):  # stitched-on patches sit 1.5 mm proud, under-layers 1.5 mm in, so the two hides never z-fight
            from scipy.spatial import cKDTree
            btree = cKDTree(bVp)
            for i, M in enumerate(G["meshes"]):
                if not np.isin(i, pid[glue[:, 0]]):
                    continue
                lay = int(M["piece"].spec.get("layer", 0) or 0)
                v = np.nonzero(pid == i)[0]
                cp = bVp[btree.query(Xr[v])[1]]
                nrm = Xr[v] - cp
                nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
                Xr[v] += nrm * (0.0015 if lay >= 0 else -0.0015)
        WV, WF = weld(Xr, tris, G["pairs"])
        items = [{"name": "mannequin", "V": bVp, "F": bF, "mat": "mannequin", "frame": False},
                 {"name": "garment", "V": WV, "F": WF, "mat": "leather", "color": col, "thick": 0.0013,
                  "subsurf": 1}]
        for i, L in enumerate(seam_lines(Xf, G["pairs"], G["seam_report"])):
            items.append({"name": f"seam{i}", "curve": L["pts"], "radius": 0.0016 if "welt" in (L["kind"] or "") else 0.0011,
                          "mat": "leather", "color": tuple(c * 0.8 for c in col), "frame": False})
        lo, hi = WV.min(0), WV.max(0)
        if o.get("frame", "garment") == "full":
            tgt, hgt = (0, 0, 1.02), 1.95
        else:  # product framing: the garment fills the frame with ~9% margin (PHOTO_STANDARD)
            tgt, hgt = (lo + hi) / 2, (hi[2] - lo[2]) * 1.2
        imgs = render_views(items, out, o["views"], size=tuple(o["size"]), samples=o["samples"],
                            target=tgt, height=hgt, device=None if dev != "cpu" else "cpu", log=log,
                            floor=o.get("frame", "garment") == "full")
        files["views"] = [os.path.basename(p) for p in imgs]
        T["render"] = time.time() - t
    return {"outputs": files, "checks": checks, "metrics": {"seconds": {k: round(v, 2) for k, v in T.items()},
            "particles": cl.n, "triangles": int(len(tris)), "seam_pairs": int(len(G["pairs"])), "device": str(dev),
            "steps": n_sew + n_pose + n_settle}, "seams": G["seam_report"]}

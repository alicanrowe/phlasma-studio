"""Sculpting: SDF brushes as JSON, concept images to 3D, and any mesh cleaned into a castable solid.

1. Brush scenes (JSON, no Python): {"sculpt": {"voxel": 0.15, "ops": [ ... ]}} where each op is one of
     {"blob": {"c": [x,y,z], "r": [rx,ry,rz], "x": [axis], "z": [axis]}}         ellipsoid
     {"ball": {"c": [...], "r": 2}}
     {"stroke": {"pts": [[x,y,z], ...], "r": [r0, r1, ...] | r, "smooth": 0.3}}  member along a path, radius per point
     {"loft": {"pts": [...], "w": [half-widths], "h": [half-heights], "up": [axis], "twist": 0}}  one body
                                                                                 swept through elliptical sections
     {"ring": {"c": [...], "R": 8, "r": 1, "n": [axis]}}                         torus
     {"cone": {"a": [...], "b": [...], "ra": 1, "rb": 0.2, "round": true}}
     {"box": {"c": [...], "size": [...], "round": 0.5}}
     {"mesh": {"file": "x.stl", "pitch": 0.2, "scale": 1.0, "at": [...]}}        any mesh, as a distance grid
     {"inflate": {"image": "x.png", "width": 20, "depth": 4, "profile": 0.5, "soften": 0.3}}  silhouette
                                                                                 blown up (Poisson height, smooth)
     {"hull": {"front": "f.png", "side": "s.png", "width": 20, "round": 0.5}}    front + side visual hull
     {"group": {"ops": [...], "warps": [["twist", {...}], ["bend", {...}]]}}     warped sub-sculpt (sdf.warp_points)
     {"relief": {"kind": "noise"|"bands", "amp": 0.2, "scale": 2, ...}}         surface relief on all so far
   and every op may carry "op": "add"|"sub"|"int", "k": blend radius, "mirror": "x" (adds the mirror image).
   A part may instead start from a press-on nail base: {"nail": {...}, "sculpt": {"ops": [...]}} (nails.py); the
   brushes blend into the base and anything below the fitting surface is cut off afterwards.
2. Images: plain-backdrop product shots or drawings; the silhouette is found by backdrop colour + saturation
   (largest blob, holes kept when "holes": true). Units: "width" is the object's real size across the image.
3. Remesh (verb cad remesh): any STL/GLB/OBJ (scan, AI model, old file) -> voxel SDF -> one watertight body at
   true size, with optional offset (grow/shrink), smoothing, and the usual walls/bodies checks.
"""
import os
import numpy as np

from .sdf import Scene, mesh_to_sdf, nz, weld

OPS = {"add": 0, "sub": 1, "int": 2}


# ------------------------------------------------------------------ images -> masks -> distance grids
def silhouette(path, holes=False, thresh=0.12):
    from PIL import Image
    from scipy import ndimage
    im = Image.open(path)
    if im.mode in ("RGBA", "LA"):
        a = np.asarray(im.convert("RGBA"))[..., 3] > 127
        mask = a
    else:
        rgb = np.asarray(im.convert("RGB")).astype(np.float32) / 255
        b = np.concatenate([rgb[:6].reshape(-1, 3), rgb[-6:].reshape(-1, 3), rgb[:, :6].reshape(-1, 3), rgb[:, -6:].reshape(-1, 3)])
        bg = np.median(b, 0)
        mask = (np.linalg.norm(rgb - bg, axis=2) > thresh) | ((rgb.max(2) - rgb.min(2)) > 0.08)
        mask = ndimage.binary_opening(mask, iterations=2)
    lab, n = ndimage.label(mask)
    if n > 1:
        mask = lab == (np.argmax(ndimage.sum(mask, lab, range(1, n + 1))) + 1)
    if not holes:
        mask = ndimage.binary_fill_holes(mask)
    ys, xs = np.nonzero(mask)
    return mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _sdf2(mask, pitch, blur=1.0):
    """smooth signed 2D distance of a mask (negative inside), in model units: EDT both ways, then a light blur
    so the outline is sub-voxel and free of stair steps"""
    from scipy.ndimage import distance_transform_edt, gaussian_filter
    d = (distance_transform_edt(~mask) - distance_transform_edt(mask)).astype(np.float32)
    d += np.where(mask, 0.5, -0.5).astype(np.float32)
    return gaussian_filter(d, blur) * pitch if blur else d * pitch


def _mask_at(mask, px_per_unit, pitch, blur_units=0.0):
    """resample a silhouette to the voxel grid, low-passing first so pixel noise and jaggies don't survive"""
    from scipy import ndimage
    z = px_per_unit * pitch
    m = mask.astype(np.float32)
    sig = max(0.5 * z, blur_units * px_per_unit)
    m = ndimage.gaussian_filter(m, sig) if sig > 0.3 else m
    return ndimage.zoom(m, 1.0 / z, order=1) > 0.5


def _poisson(m):
    """solve lap(u) = -1 inside the mask, u = 0 outside: a smooth inflation height with no medial-axis ridge"""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve
    idx = -np.ones(m.shape, np.int64)
    ii = np.argwhere(m)
    idx[m] = np.arange(len(ii))
    rows, cols, vals = [np.arange(len(ii))], [np.arange(len(ii))], [np.full(len(ii), 4.0)]
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        j = idx[ii[:, 0] + dx, ii[:, 1] + dy]
        ok = j >= 0
        rows.append(np.nonzero(ok)[0]); cols.append(j[ok]); vals.append(-np.ones(ok.sum()))
    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(ii),) * 2).tocsr()
    u = np.zeros(m.shape, np.float32)
    u[m] = spsolve(A, np.ones(len(ii)))
    return u


def inflate_grid(mask, width, depth, pitch, profile=0.5, soften=0.3):
    """silhouette (rows = down, cols = across) blown up symmetrically: half-thickness at a point = depth/2 *
    (u / umax) ** profile, where u solves a Poisson equation on the silhouette (profile 0.5 = round pillow, 1 = cone).
    The field is max(outline distance, |y| - h): smooth, exact on the outline, and its zero set is the pillow.
    soften (model units) rounds the outline first (removes thin spikes narrower than ~2 x soften).
    Returns (D, lo, pitch) with the silhouette in the XZ plane (X across, Z up), thickness along Y."""
    ppu = mask.shape[1] / width
    m = _mask_at(mask, ppu, pitch, soften)[::-1].T          # (x, z), z up
    pad = int(np.ceil(1.0 / pitch)) + 3
    m = np.pad(m, pad)
    sd = _sdf2(m, pitch)                                     # (x, z), negative inside
    u = _poisson(m)
    h = (depth / 2) * (u / max(u.max(), 1e-9)) ** profile
    ny = int(np.ceil((depth + 2.0) / pitch)) + 7
    y = (np.arange(ny) - (ny - 1) / 2) * pitch
    D = np.maximum(sd[:, None, :], np.abs(y)[None, :, None] - h[:, None, :]).astype(np.float32)
    return D, np.array([-pad * pitch, y[0], -pad * pitch]), pitch


def hull_grid(front, side, width, pitch, round_k=0.0, soften=0.0):
    """front silhouette (XZ) and side silhouette (YZ), scaled to the same height: their visual hull, as
    max(front extrusion distance, side extrusion distance). round_k > 0 rounds the hull's ridges (smooth max)."""
    ppu = front.shape[1] / width
    f = _mask_at(front, ppu, pitch, soften)[::-1].T          # (x, z)
    height = front.shape[0] / ppu
    sp = side.shape[0] / height
    sd = _mask_at(side, sp, pitch, soften)[::-1].T           # (y, z)
    nz_ = min(f.shape[1], sd.shape[1])
    pad = int(np.ceil((1.0 + round_k) / pitch)) + 3
    f, sd = np.pad(f[:, :nz_], pad), np.pad(sd[:, :nz_], pad)
    a, b = _sdf2(f, pitch)[:, None, :], _sdf2(sd, pitch)[None, :, :]
    if round_k:   # smooth max (polynomial), rounds the crease where the two extrusions meet
        hh = np.clip(0.5 - 0.5 * (b - a) / round_k, 0, 1)
        D = b * (1 - hh) + a * hh + round_k * hh * (1 - hh)
    else:
        D = np.maximum(a, b)
    return D.astype(np.float32), np.array([-pad * pitch] * 3, float), pitch


def _centre_grid(D, lo, pitch, at=None, centre=True):
    """centre the solid (D < 0) on x/y = 0 with its base on z = 0, then move it to `at`"""
    if centre:
        ii = np.argwhere(D < 0)
        a, b = ii.min(0) * pitch, ii.max(0) * pitch
        lo = -(a + b) / 2
        lo[2] = -a[2]
    if at is not None:
        lo = lo + np.asarray(at, float)
    return D, lo, pitch


# ------------------------------------------------------------------ JSON brush scenes
def load_mesh(path, scale=1.0):
    import trimesh
    m = weld(trimesh.load(path, force="mesh"))
    m.apply_scale(scale)
    return m


def loft(s, pts, w, h, up=(0, 0, 1), step=0.2, op=0, k=None, twist=0.0, k_join=0.0):
    """one body swept along a path through elliptical sections: half-width w and half-height h per point (either
    may be a number), "up" sets which way h points (rotated by twist degrees in total along the path). Sections are
    densified to `step` and blended (radius k, default 2 x step), so the result is one continuous surface. The first
    section joins what is already in the scene with blend k_join (the op's "k")."""
    from scipy.interpolate import CubicSpline
    P = np.asarray(pts, float)
    n = len(P)
    W = np.broadcast_to(np.asarray(w, float), (n,)).astype(float)
    H = np.broadcast_to(np.asarray(h, float), (n,)).astype(float)
    t = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    L = t[-1]
    m = max(int(np.ceil(L / step)) + 1, 2)
    tt = np.linspace(0, L, m)
    if n >= 3:
        Pd = CubicSpline(t, P, bc_type="natural")(tt)
    else:
        Pd = np.stack([np.interp(tt, t, P[:, i]) for i in range(3)], 1)
    Wd, Hd = np.interp(tt, t, W), np.interp(tt, t, H)
    T = np.gradient(Pd, axis=0)
    up = nz(up)
    k = step * 2.0 if k is None else k
    for i in range(m):
        ti = nz(T[i])
        side = np.cross(ti, up)
        side = nz(side) if np.linalg.norm(side) > 1e-6 else nz(np.cross(ti, [1.0, 0, 0]))
        u = np.cross(side, ti)
        a = np.radians(twist) * tt[i] / max(L, 1e-9)
        side, u = side * np.cos(a) + u * np.sin(a), u * np.cos(a) - side * np.sin(a)
        r_along = max(step * 1.6, 0.6 * min(Wd[i], Hd[i]))
        s.ellipsoid(Pd[i], (max(Wd[i], 0.05), r_along, max(Hd[i], 0.05)), xaxis=side, zhint=u, op=op,
                    k=k if i else k_join)
    return s


def build_scene(ops, base_dir=None, s=None):
    s = s or Scene()
    for o in ops:
        (kind, a), = [(k, v) for k, v in o.items() if k not in ("op", "k", "mirror")]
        op, k = OPS[o.get("op", "add")], float(o.get("k", 0.0))
        copies = [1.0]
        if o.get("mirror"):
            copies.append(-1.0)
        for sg in copies:
            i = "xyz".index(o["mirror"]) if sg < 0 else None

            def M(p):
                p = np.array(p, float)
                if i is not None:
                    p = p.copy(); p[..., i] *= -1
                return p
            if kind == "blob":
                x = a.get("x"); x = M(x) if x is not None and i is not None else x
                s.ellipsoid(M(a["c"]), a["r"], xaxis=x, zhint=a.get("z"), op=op, k=k)
            elif kind == "ball":
                s.sphere(M(a["c"]), a["r"], op=op, k=k)
            elif kind == "stroke":
                from .sdf import dense
                P = M(a["pts"])
                r = np.broadcast_to(np.asarray(a["r"], float), (len(P),)) if np.ndim(a["r"]) == 0 else np.asarray(a["r"], float)
                Pd, rd = dense(P, r, step=a.get("step", 0.25))
                s.stroke(Pd, rd, op=op, k=k or a.get("smooth", 0.0))
            elif kind == "ring":
                n = a.get("n", [0, 0, 1]); n = M(n) if i is not None else n
                s.torus(M(a["c"]), a["R"], a["r"], normal=n, op=op, k=k)
            elif kind == "cone":
                s.cone(M(a["a"]), M(a["b"]), a["ra"], a["rb"], op=op, k=k, round=a.get("round", True))
            elif kind == "box":
                s.box(M(a["c"]), a["size"], rnd=a.get("round", 0.0), op=op, k=k)
            elif kind == "mesh":
                f = _path(a["file"], base_dir)
                D, lo, p = mesh_to_sdf(load_mesh(f, a.get("scale", 1.0)), a.get("pitch", 0.2))
                if a.get("at") is not None:
                    lo = lo + np.asarray(a["at"], float)
                s.grid_sdf(D, lo, p, op=op, k=k)
            elif kind == "inflate":
                mask = silhouette(_path(a["image"], base_dir), a.get("holes", False))
                D, lo, p = inflate_grid(mask, a["width"], a["depth"], a.get("pitch", 0.15), a.get("profile", 0.5),
                                       a.get("soften", 0.3))
                s.grid_sdf(*_centre_grid(D, lo, p, a.get("at")), op=op, k=k)
            elif kind == "hull":
                fm = silhouette(_path(a["front"], base_dir))
                sm = silhouette(_path(a["side"], base_dir))
                D, lo, p = hull_grid(fm, sm, a["width"], a.get("pitch", 0.15), round_k=a.get("round", 0.0),
                                    soften=a.get("soften", 0.0))
                s.grid_sdf(*_centre_grid(D, lo, p, a.get("at")), op=op, k=k)
            elif kind == "loft":
                loft(s, M(a["pts"]), a["w"], a["h"], up=a.get("up", (0, 0, 1)), step=a.get("step", 0.2), op=op,
                     k=a.get("blend"), twist=a.get("twist", 0.0), k_join=k)
            elif kind == "group":
                sub = build_scene(a["ops"], base_dir)
                s.group(sub, [tuple(w) for w in a.get("warps", [])], op=op, k=k)
            elif kind == "relief":
                s.displace(**{kk: v for kk, v in a.items()})
            else:
                raise ValueError(f"unknown sculpt op {kind}")
    return s


def _path(p, base):
    if os.path.isabs(p) or base is None:
        return p
    q = os.path.join(base, p)
    return q if os.path.exists(q) else p

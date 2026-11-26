"""Press-on nail bases: a parametric shell (outline x C-curve x arch x thickness) as an SDF grid, so sculpt brushes
can be added on top of it (and blended into it) in the same scene.

Frame (mm): X across the nail, Y along it (0 = cuticle edge, +Y toward the free edge), Z up out of the nail.
The underside is the fitting surface: a cylinder of radius `c_curve` across X (the C-curve) dipped along Y by the
arch (radius `arch`). The shell grows up from it by `thick`, thinner at the cuticle so it blends into the real nail.

  nail_scene({"size": 4, "shape": "almond", "ext": 8})      one base, as a Scene (add brushes to it)
  nail_set(spec)                                            the 10-size set: [(size, width, Scene, meta), ...]

Sizes: the usual press-on scale, 0 (thumb, widest) .. 9 (pinky). Width is the cuticle-end chord across the nail.
Shapes: square, squoval, round, oval, almond, stiletto, coffin (ballerina), edge (pointed flat-top).
"""
import numpy as np

from .sdf import Scene

WIDTHS = {0: 18.0, 1: 17.0, 2: 16.0, 3: 15.0, 4: 14.0, 5: 13.0, 6: 12.0, 7: 11.0, 8: 10.0, 9: 9.0}
FINGERS = {"thumb": (0, 1), "index": (4, 5), "middle": (3, 4), "ring": (5, 6), "pinky": (8, 9)}
SHAPES = ("square", "squoval", "round", "oval", "almond", "stiletto", "coffin", "edge")
DEFAULTS = {"size": 4, "shape": "almond", "ext": 6.0, "bed": None, "c_curve": None, "arch": 60.0, "thick": 0.7,
            "thick_cuticle": 0.45, "taper": 0.94, "cuticle_depth": None, "tip_len": None, "pitch": 0.06}


def _halfwidth(shape, s, hw):
    """half-width at fraction s (0..1) along the free-edge tip zone, as a fraction of the tip's half-width"""
    s = np.clip(s, 0, 1)
    return {
        "square": np.ones_like(s),
        "squoval": np.sqrt(np.clip(1 - s ** 2.5, 0, 1)),
        "round": np.sqrt(np.clip(1 - s ** 2, 0, 1)),
        "oval": np.sqrt(np.clip(1 - s ** 2, 0, 1)),
        "almond": np.clip(1 - s ** 1.7, 0, 1) ** 0.75,
        "stiletto": np.clip(1 - s, 0, 1) ** 1.1,
        "coffin": 1 - 0.55 * s,
        "edge": np.clip(1 - s, 0, 1) ** 0.9,
    }[shape]


def _tip_len(shape, hw, ext):
    return {"square": 0.6, "squoval": 1.2, "round": hw, "oval": 1.5 * hw, "coffin": 0.85 * (ext + hw),
            "almond": 0.9 * (ext + hw), "stiletto": ext + 1.4 * hw, "edge": ext + hw}[shape]


def outline(p):
    """the nail's plan outline (shapely Polygon, XY) and its length"""
    from shapely.geometry import Polygon
    w = WIDTHS[int(p["size"])] if p.get("width") is None else float(p["width"])
    hw0 = w / 2
    bed = p["bed"] or 0.95 * w
    L = bed + float(p["ext"])
    hwt = hw0 * p["taper"]
    tl = min(p["tip_len"] or _tip_len(p["shape"], hwt, p["ext"]), L * 0.9)
    cd = p["cuticle_depth"] if p["cuticle_depth"] is not None else 0.18 * w
    y = np.linspace(0, L, 240)
    hw = hw0 + (hwt - hw0) * np.clip(y / max(L - tl, 1e-6), 0, 1)
    hw = hw * _halfwidth(p["shape"], (y - (L - tl)) / tl, hwt)
    if p["shape"] == "coffin":
        hw[-1] = hw[-2]
    else:
        hw[-1] = 0.0 if p["shape"] in ("stiletto", "edge", "almond") else hw[-1]
    xs = np.linspace(-hw0, hw0, 60)
    cut = cd * (xs / hw0) ** 2                                 # rounded proximal edge (corners reach further back)
    right = np.stack([hw, y], 1)[1:]
    left = np.stack([-hw[::-1], y[::-1]], 1)[:-1]
    base = np.stack([xs, cut], 1)
    pts = np.concatenate([base, right[right[:, 1] > cd], left[left[:, 1] > cd]])
    pg = Polygon(pts).buffer(0)
    if p["shape"] in ("square", "coffin"):
        pg = pg.buffer(-0.5).buffer(0.5)                        # break the tip corners
    else:
        pg = pg.buffer(-0.25).buffer(0.25)                       # a point is never sharper than 0.25 mm
    return pg, w, L


def _sdf_poly(pg, xs, ys):
    """signed 2D distance to a polygon on a grid (negative inside), exact to the edges"""
    from shapely import points, distance, contains_xy
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    d = distance(pg.boundary, points(X.ravel(), Y.ravel())).reshape(X.shape)
    inside = contains_xy(pg, X, Y)
    return np.where(inside, -d, d).astype(np.float32)


def under_z(p, w, x, y, L):
    """height of the fitting (under) surface: C-curve across, arch along, highest at the stress point ~ 0.45 L"""
    Rc = p["c_curve"] or 0.62 * w
    xa = np.clip(np.abs(x), 0, Rc * 0.999)
    zc = np.sqrt(Rc ** 2 - xa ** 2) - Rc
    ya = y - 0.45 * L
    za = -(ya ** 2) / (2 * p["arch"])
    return zc + za


def nail_grid(p):
    p = {**DEFAULTS, **p}
    if p["shape"] not in SHAPES:
        raise ValueError(f"unknown nail shape {p['shape']}; one of {', '.join(SHAPES)}")
    if p.get("width") is None and int(p["size"]) not in WIDTHS:
        raise ValueError(f"nail size {p['size']} not in 0..9 (or give width in mm)")
    pg, w, L = outline(p)
    pitch = float(p["pitch"])
    x0, y0, x1, y1 = pg.bounds
    pad = 4 * pitch + 0.6
    xs = np.arange(x0 - pad, x1 + pad, pitch)
    ys = np.arange(y0 - pad, y1 + pad, pitch)
    zlo = float(under_z(p, w, np.array([x0, x1]), np.array([0.0, L]), L).min()) - 1.5
    zhi = float(p["thick"]) + 1.5
    zs = np.arange(zlo, zhi, pitch)
    d2 = _sdf_poly(pg, xs, ys)                                  # (x, y)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    zu = under_z(p, w, X, Y, L)
    # thickness ramps from thick_cuticle at the cuticle to thick over the first 35% of the bed
    t = p["thick_cuticle"] + (p["thick"] - p["thick_cuticle"]) * np.clip(Y / (0.35 * w), 0, 1) ** 0.7
    gx, gy = np.gradient(zu, pitch)
    slope = np.sqrt(1 + gx ** 2 + gy ** 2)                     # turns height differences into true distances
    zc = zu + t / 2
    dz = (np.abs(zs[None, None, :] - zc[..., None]) - (t / 2)[..., None]) / slope[..., None]
    D = np.maximum(dz, d2[..., None]).astype(np.float32)
    meta = {"size": int(p["size"]) if p.get("width") is None else None, "width_mm": round(w, 3),
            "length_mm": round(L, 3), "shape": p["shape"], "ext_mm": p["ext"], "thick_mm": p["thick"],
            "c_curve_mm": round(p["c_curve"] or 0.62 * w, 3)}
    return D, np.array([xs[0], ys[0], zs[0]]), pitch, meta


def fit_cut(p, s, pad=6.0):
    """cut away everything below the fitting surface (call after brushes), so the nail still sits flush"""
    p = {**DEFAULTS, **p}
    pg, w, L = outline(p)
    lo, hi = s.bounds(pad=0.0)
    pitch = float(p["pitch"])
    xs = np.arange(lo[0] - pitch, hi[0] + pitch, pitch)
    ys = np.arange(lo[1] - pitch, hi[1] + pitch, pitch)
    zs = np.arange(lo[2] - pitch, hi[2] + pitch, pitch)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    zu = under_z(p, w, X, np.clip(Y, -pad, L + pad), L)
    # < 0 below the fitting surface, over the nail only: an extension past the free edge may curl down freely
    D = np.maximum(zs[None, None, :] - zu[..., None], (Y - (L + 0.5))[..., None]).astype(np.float32)
    s.grid_sdf(D, np.array([xs[0], ys[0], zs[0]]), pitch, op=1)
    return s


def extend(p, s, e):
    """one lofted body growing out of the free edge (never a part stuck on). e = {"profile": [[f, dz], ...] side-view
    centreline, f = 0..1 along the reach, dz mm up from the free edge; "reach": length as a multiple of the bed
    (2.5..3 reads as designed, under 1.5 as a plain nail); "w", "h": half-width / half-height per point, mm;
    "dx": sideways drift per point, mm; "twist": degrees in total; "blend": join radius into the shell}. The root half-width is capped at half the shell's half-width."""
    from .sculpt import loft
    p = {**DEFAULTS, **p}
    pg, w, L = outline(p)
    bed = p["bed"] or 0.95 * w
    reach = float(e.get("reach", 2.5)) * bed
    prof = np.asarray(e.get("profile", [[0, 0], [0.5, 0.6], [1, 0.2]]), float)
    n = len(prof)
    dx = np.broadcast_to(np.asarray(e.get("dx", 0.0), float), (n,))
    y0 = L - min(2.0, 0.3 * L)                                  # start inside the shell, so the two are one body
    z0 = float(under_z(p, w, np.array([0.0]), np.array([y0]), L)[0]) + p["thick"] / 2
    pts = np.stack([dx, y0 + prof[:, 0] * (reach + L - y0), z0 + prof[:, 1]], 1)
    hw = float(np.interp(y0, *_halfwidth_profile(pg, y0)))
    W = np.array(e.get("w", np.linspace(max(hw * 0.5, 0.6), 0.3, n)), float)
    H = np.array(e.get("h", np.linspace(p["thick"] * 0.9, 0.3, n)), float)
    # a flat section wider than the C-curve's crown stands proud of the shell as "wings" (a cap with a feature on
    # it): keep the root inside the crown and let the blend carry it into the shell
    W[0] = min(W[0], 0.5 * hw)
    loft(s, pts, W, H, up=(0, 0, 1), step=float(e.get("step", 0.2)), twist=float(e.get("twist", 0.0)),
         k=e.get("smooth"), k_join=float(e.get("blend", 1.2)))
    return s, {"reach_mm": round(reach, 2), "reach_x_bed": round(reach / bed, 2), "bed_mm": round(bed, 2)}


def _halfwidth_profile(pg, y):
    """(ys, half-widths) of the outline near y, for np.interp"""
    from shapely.geometry import LineString
    ys = np.array([y - 0.01, y + 0.01])
    x0, _, x1, _ = pg.bounds
    hws = [pg.intersection(LineString([(x0 - 1, yy), (x1 + 1, yy)])).length / 2 for yy in ys]
    return ys, np.array(hws)


def nail_scene(p, s=None):
    D, lo, pitch, meta = nail_grid(p)
    s = s or Scene()
    s.grid_sdf(D, lo, pitch)
    if p.get("extend"):
        _, em = extend(p, s, p["extend"])
        meta.update(em)
    return s, meta


def nail_set(spec):
    """the 10-size set (or spec["sizes"]), each its own Scene; ext scales with width unless spec["ext_fixed"]"""
    out = []
    sizes = spec.get("sizes", list(range(10)))
    for z in sizes:
        q = {**spec, "size": z}
        q.pop("sizes", None)
        if not spec.get("ext_fixed"):
            q["ext"] = float(spec.get("ext", DEFAULTS["ext"])) * WIDTHS[z] / WIDTHS[4]
        s, meta = nail_scene(q)
        out.append((z, WIDTHS[z], s, meta))
    return out


def check(model):
    """fit and comfort of every part that carries nail meta: cuticle chord within 0.6 mm of the size width,
    shell thickness over the bed (p50 of measured walls) 0.4..1.2 mm; extensions and brushwork are not counted"""
    from .checks import wall_thickness
    out, ok = {}, True
    for p in model.parts:
        nm = p.meta.get("nail")
        if not nm:
            continue
        m = p.mesh()
        V = m.vertices
        band = V[(V[:, 1] > 0.15 * nm["width_mm"]) & (V[:, 1] < 0.25 * nm["width_mm"])]
        chord = float(np.ptp(band[:, 0])) if len(band) else 0.0
        d, at = wall_thickness(m, n=2500, points=True)
        d = d[(at[:, 1] > 0.1 * nm["width_mm"]) & (at[:, 1] < 0.85 * nm["length_mm"])]   # the shell, not add-ons
        p50 = float(np.percentile(d, 50)) if len(d) else 0.0
        good = abs(chord - nm["width_mm"]) <= 0.6 and 0.4 <= p50 <= 1.2
        ok &= good
        out[p.name] = {"width_mm": nm["width_mm"], "measured_width_mm": round(chord, 2),
                       "wall_p50_mm": round(p50, 3), "pass": bool(good)}
    return {"pass": bool(ok), "parts": out}

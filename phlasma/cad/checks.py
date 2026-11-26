"""Checks that run on every build and land in run.json["checks"]. Each returns {"pass": bool, ...numbers}.

bodies       every part is one closed body (a stone, a plate, a casting)
interference no two parts overlap (exact: manifold boolean volume), contact is fine; layers skip their host
walls        cast/printed metal: ray-sampled wall thickness, 5th percentile >= floor (default 0.8 mm)
cnc          plates, when a bit is given: no slot, notch or hole narrower than the bit
laser        plates: fits the bed with grain along the sheet's long side, no bar or web thinner than the sheet
stones       stock calibrated size, metal over the girdle round at least 180 deg (held), seated
ring_size    measured inside diameter matches the asked US size within 0.1 mm
joinery      mortise walls, webs, far face and distance from a leg end (joinery.py rules)
stands       the assembly rests on its lowest contacts with the centre of mass inside them (furniture)
"""
import math
import numpy as np


def _man(m):
    import manifold3d as mf
    return mf.Manifold(mf.Mesh(vert_properties=np.asarray(m.vertices, np.float32),
                               tri_verts=np.asarray(m.faces, np.uint32)))


def bodies(model):
    bad = {}
    for p in model.parts:
        if p.kind == "brep":
            n = len(p.shape.solids())
        else:
            n = len(p.mesh().split(only_watertight=False))
        wt = bool(p.mesh().is_watertight)
        if n != p.meta.get("bodies", 1) or not wt:
            bad[p.name] = {"bodies": n, "watertight": wt}
    return {"pass": not bad, "bad": bad, "parts": len(model.parts)}


def interference(model, tol_mm3=0.001):
    tol = tol_mm3 / model.mm ** 3
    ps = model.parts
    meshes = [p.mesh() for p in ps]
    mans = [_man(m) for m in meshes]
    allow = {tuple(sorted(a)) for a in model.meta.get("allow_overlap", [])}
    bad, tested = [], 0
    for i in range(len(ps)):
        bi = meshes[i].bounds
        for j in range(i + 1, len(ps)):
            a, b = ps[i], ps[j]
            if a.meta.get("host") == b.name or b.meta.get("host") == a.name:
                continue
            if tuple(sorted((a.name, b.name))) in allow:
                continue
            bj = meshes[j].bounds
            if np.any(bi[1] < bj[0]) or np.any(bj[1] < bi[0]):
                continue
            tested += 1
            v = (mans[i] ^ mans[j]).volume()
            if v > tol:
                bad.append({"a": a.name, "b": b.name, "overlap_mm3": round(v * model.mm ** 3, 4)})
    return {"pass": not bad, "overlaps": bad, "pairs_tested": tested}


def ray_mesh(m, faces=40000):
    """trimesh's pure-Python ray caster blows memory on big meshes (every ray against every triangle box it
    crosses): cast against a decimated copy. Deviation is far under the 0.1 mm the checks care about."""
    if len(m.faces) <= faces:
        return m
    try:
        import fast_simplification, trimesh
        v, f = fast_simplification.simplify(m.vertices, m.faces, 1 - faces / len(m.faces))
        return trimesh.Trimesh(v, f, process=True)
    except ImportError:
        return m


def wall_thickness(mesh, n=1500, seed=0):
    mesh = ray_mesh(mesh)
    """ray-sampled local thickness: from points on the surface, straight in along -normal to the far side"""
    rng = np.random.default_rng(seed)
    pts, fi = mesh.sample(n, return_index=True, seed=int(rng.integers(1 << 30)))
    nrm = mesh.face_normals[fi]
    eps = 1e-4 * mesh.scale
    org = pts - nrm * eps
    loc, ri, _ = mesh.ray.intersects_location(org, -nrm, multiple_hits=True)
    d = np.full(n, np.inf)
    if len(ri):
        dist = np.linalg.norm(loc - org[ri], axis=1)
        dist[dist < eps] = np.inf
        np.minimum.at(d, ri, dist)
    return d[np.isfinite(d)]


def walls(model, floor_mm=0.8, metals=("sterling", "silver", "brass", "bronze", "gold14k", "gold18k", "titanium", "steel")):
    out, ok = {}, True
    for p in model.parts:
        if p.role != "part" or p.material not in metals:
            continue
        d = wall_thickness(p.mesh()) * model.mm
        if not len(d):
            continue
        p5, mn = float(np.percentile(d, 5)), float(d.min())
        good = p5 >= floor_mm
        ok &= good
        out[p.name] = {"p5_mm": round(p5, 3), "min_mm": round(mn, 3), "floor_mm": floor_mm, "pass": bool(good)}
    return {"pass": bool(ok), "parts": out}


def laser(model):
    sh = model.sheet
    if not sh:
        return {"pass": True, "skipped": "no sheet"}
    out, ok = {}, True
    for p in model.parts:
        if p.kind != "plate":
            continue
        x0, y0, x1, y1 = p.poly.bounds
        fits = (x1 - x0) <= sh["w"] and (y1 - y0) <= sh["h"]
        r = 0.49 * p.thick
        opened = p.poly.buffer(-r, join_style=1).buffer(r, join_style=1)
        thin = p.poly.difference(opened)
        geoms = list(getattr(thin, "geoms", [thin])) if not thin.is_empty else []
        bars = [g for g in geoms if g.area > 0.25 * p.thick ** 2]
        good = fits and not bars
        ok &= good
        rec = {"pass": bool(good), "fits_bed": bool(fits), "size": [round(x1 - x0, 4), round(y1 - y0, 4)]}
        if bars:
            rec["thin_spots"] = [[round(c, 3) for c in g.centroid.coords[0]] for g in bars[:8]]
        out[p.name] = rec
    return {"pass": bool(ok), "parts": out}


def cnc(model, tool_d):
    """every cut-away gap (slots, notches, holes) must be at least the bit's width, or the bit can't get in"""
    from shapely.geometry import Polygon
    out, ok = {}, True
    r = 0.49 * tool_d
    for p in model.parts:
        if p.kind != "plate":
            continue
        g = p.poly
        voids = g.envelope.difference(g)
        for pg in getattr(g, "geoms", [g]):
            for ring in pg.interiors:
                voids = voids.union(Polygon(ring))
        reach = voids.buffer(-r, join_style=1).buffer(r, join_style=1)
        bad = voids.difference(reach)
        geoms = [x for x in getattr(bad, "geoms", [bad]) if not x.is_empty and x.area > 0.5 * tool_d ** 2]
        # corners the bit rounds are fine (dogboned at export); only gaps narrower than the bit count
        narrow = [x for x in geoms if x.minimum_rotated_rectangle.length > 0 and
                  min(_sides(x.minimum_rotated_rectangle)) < tool_d * 0.98]
        if narrow:
            ok = False
            out[p.name] = {"pass": False, "narrow_gaps": [[round(c, 3) for c in x.centroid.coords[0]] for x in narrow[:8]]}
    return {"pass": bool(ok), "tool_d": tool_d, "parts": out}


def _sides(rect):
    import numpy as np
    P = np.array(rect.exterior.coords)[:4]
    return [float(np.linalg.norm(P[1] - P[0])), float(np.linalg.norm(P[2] - P[1]))]


def stones(model, held_gap_deg=180.0):
    from .jewelry import ROUND_STOCK, BALL_STOCK
    import trimesh
    metal = [p.mesh() for p in model.parts if p.role == "part" and p.kind != "plate"]
    if not metal:
        return {"pass": True, "skipped": "no metal"}
    M = ray_mesh(trimesh.util.concatenate(metal))
    out, ok = {}, True
    for p in model.parts:
        if p.role != "stone":
            continue
        st = p.meta["stone"]
        D, c, n = st["D"] * model.mm, np.array(st["c"]), np.array(st["n"], float)
        n /= np.linalg.norm(n)
        stock = ROUND_STOCK if st.get("cut", "round") == "round" else BALL_STOCK
        is_stock = any(abs(D - s) < 0.01 for s in stock)
        a = st["D"] / 2
        e1 = np.cross(n, [1, 0, 0] if abs(n[0]) < 0.9 else [0, 1, 0]); e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        # held: just above the girdle (crown side), look outward: metal within 0.9 mm past the girdle edge
        angs = np.radians(np.arange(0, 360, 10))
        dirs = np.array([math.cos(t) * e1 + math.sin(t) * e2 for t in angs])
        z_lip = 0.03 * st["D"] / 2 + 0.06 * st["D"]
        org = c + n * z_lip + dirs * (a * 0.6)
        reach = (a * 0.4 + 0.9 / model.mm)
        loc, ri, _ = M.ray.intersects_location(org, dirs, multiple_hits=False)
        hit = np.zeros(len(angs), bool)
        if len(ri):
            dd = np.linalg.norm(loc - org[ri], axis=1)
            hit[ri[dd <= reach]] = True
        if hit.all():
            gap = 0.0
        elif not hit.any():
            gap = 360.0
        else:
            h = np.r_[hit, hit]
            run, best = 0, 0
            for v in h:
                run = 0 if v else run + 1
                best = max(best, run)
            gap = min(best, len(angs)) * 10.0
        # seated: metal under the pavilion (looking down the axis from inside the stone)
        org2 = c - n * (0.2 * st["D"]) + np.array([math.cos(t) * e1 + math.sin(t) * e2 for t in angs[::3]]) * a * 0.35
        loc2, ri2, _ = M.ray.intersects_location(org2, np.tile(-n, (len(org2), 1)), multiple_hits=False)
        seated = len(set(ri2.tolist())) >= 3
        good = is_stock and gap < held_gap_deg and seated
        ok &= good
        out[p.name] = {"pass": bool(good), "D_mm": round(D, 3), "stock": bool(is_stock),
                       "largest_open_arc_deg": gap, "seated": bool(seated)}
    return {"pass": bool(ok), "stones": out}


def ring_size(model, size_us, axis=(0, 0, 1), centre=(0, 0, 0), tol_mm=0.1):
    from .jewelry import ring_inner_d
    import trimesh
    metal = ray_mesh(trimesh.util.concatenate([p.mesh() for p in model.parts if p.role == "part"]))
    ax = np.asarray(axis, float); ax /= np.linalg.norm(ax)
    e1 = np.cross(ax, [1, 0, 0] if abs(ax[0]) < 0.9 else [0, 1, 0]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(ax, e1)
    angs = np.radians(np.arange(0, 360, 5))
    dirs = np.array([math.cos(t) * e1 + math.sin(t) * e2 for t in angs])
    best = np.inf
    for z in np.linspace(-1.5, 1.5, 7):
        org = np.tile(np.asarray(centre, float) + ax * z / model.mm, (len(dirs), 1))
        loc, ri, _ = metal.ray.intersects_location(org, dirs, multiple_hits=False)
        if len(ri):
            best = min(best, float(np.linalg.norm(loc - org[ri], axis=1).min()))
    want = ring_inner_d(size_us)
    got = 2 * best * model.mm
    return {"pass": bool(abs(got - want) <= tol_mm), "inner_d_mm": round(got, 3), "want_mm": round(want, 3),
            "size_us": size_us}


def stands(model, up=2, tol_mm=0.3):
    import trimesh
    ps = [p for p in model.parts if p.role != "layer"]
    ms = [p.mesh() for p in ps]
    V = np.vstack([m.vertices for m in ms])
    zmin = V[:, up].min()
    other = [i for i in range(3) if i != up]
    feet = V[V[:, up] <= zmin + tol_mm / model.mm][:, other]
    vols = np.array([abs(m.volume) for m in ms])
    com = (np.array([m.center_mass for m in ms]) * vols[:, None]).sum(0) / vols.sum()
    from scipy.spatial import ConvexHull, Delaunay
    try:
        hull = ConvexHull(feet)
        inside = bool(Delaunay(feet[hull.vertices]).find_simplex(com[other]) >= 0)
    except Exception:
        inside = False
    return {"pass": inside, "contacts": int(len(feet)), "com": [round(float(x), 4) for x in com]}


def run_all(model, opts):
    chk = model.meta.get("checks", {})
    res = {"bodies": bodies(model), "interference": interference(model)}
    if model.kind == "jewelry" or chk.get("walls"):
        res["walls"] = walls(model, floor_mm=chk.get("wall_floor_mm", 0.8))
    if any(p.kind == "plate" for p in model.parts):
        res["laser"] = laser(model)
    cn = model.meta.get("cnc") or (model.sheet or {}).get("cnc")
    if cn and any(p.kind == "plate" for p in model.parts):
        res["cnc"] = cnc(model, cn["tool_d"])
    if any(p.role == "stone" for p in model.parts):
        res["stones"] = stones(model)
    if model.meta.get("ring_size_us"):
        res["ring_size"] = ring_size(model, model.meta["ring_size_us"], model.meta.get("ring_axis", (0, 0, 1)))
    if model.meta.get("joints"):
        from .joinery import check as joinery_check
        res["joinery"] = joinery_check(model)
    if model.kind == "furniture" and chk.get("stands", True):
        res["stands"] = stands(model)
    return res

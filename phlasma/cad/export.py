"""Exports. Everything leaves in millimetres except the GLB (glTF: metres, Y up) and the laser DXF (model units,
$INSUNITS set). STEP holds only exact parts (brep + plates); sculpted mesh parts go out as STL/GLB."""
import json, os
import numpy as np

from .part import color_of


def _scaled_mesh(p, s):
    m = p.mesh().copy()
    m.apply_scale(s)
    return m


def step(model, path):
    from build123d import Compound, export_step, scale
    shapes = []
    for p in model.parts:
        b = p.brep()
        if b is None:
            continue
        if model.mm != 1.0:
            b = scale(b, model.mm)
        b.label = p.name
        shapes.append(b)
    if not shapes:
        return None
    c = Compound(children=shapes)
    c.label = model.name
    export_step(c, path)
    return path


def stl(model, folder):
    os.makedirs(folder, exist_ok=True)
    out = []
    for p in model.parts:
        f = os.path.join(folder, f"{p.name}.stl")
        _scaled_mesh(p, model.mm).export(f)
        out.append(f)
    return out


def glb(model, path):
    import trimesh
    sc = trimesh.Scene()
    # Z-up mm -> Y-up m
    R = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1.0]])
    for p in model.parts:
        m = _scaled_mesh(p, model.mm / 1000.0)
        m.apply_transform(R)
        col = np.array(list(color_of(p)) + [1.0])
        metal = p.material in ("sterling", "silver", "brass", "bronze", "steel", "stainless", "titanium", "gold14k", "gold18k")
        mat = trimesh.visual.material.PBRMaterial(name=p.material or "default", baseColorFactor=col,
                                                  metallicFactor=0.9 if metal else 0.0,
                                                  roughnessFactor=0.3 if metal else 0.7)
        m.visual = trimesh.visual.TextureVisuals(material=mat)
        sc.add_geometry(m, node_name=p.name, geom_name=p.name)
    sc.export(path)
    return path


def laser_sheets(model, folder, kerf=0.0, gap=None):
    """nest every plate on the model's sheet (grain along the long side) and write DXF + SVG per sheet"""
    from .nest import nest
    from .sketch import kerf as kerf_off
    import ezdxf
    from ezdxf import units as dxu
    sh = model.sheet
    plates = [p for p in model.parts if p.kind == "plate"]
    if not plates or not sh:
        return [], {}
    gap = gap if gap is not None else max(2 * kerf, 0.125 if model.units == "in" else 3.0)
    sheets, report = nest(plates, sh["w"], sh["h"], gap=gap, margin=sh.get("margin", gap))
    os.makedirs(folder, exist_ok=True)
    files = []
    for si, placed in enumerate(sheets):
        doc = ezdxf.new("R2010")
        doc.units = dxu.IN if model.units == "in" else dxu.MM
        doc.layers.add("CUT_INNER", color=5)
        doc.layers.add("CUT_OUTER", color=1)
        doc.layers.add("LABEL", color=7)
        msp = doc.modelspace()
        svg = []
        S = model.mm   # svg in mm
        for name, g in placed:
            g = kerf_off(g, kerf)
            for pg in getattr(g, "geoms", [g]):
                for ring in pg.interiors:
                    pts = list(ring.coords)
                    msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "CUT_INNER"})
                    svg.append(_svg_path(pts, S, sh["h"], "#0000FF"))
                pts = list(pg.exterior.coords)
                msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "CUT_OUTER"})
                svg.append(_svg_path(pts, S, sh["h"], "#FF0000"))
        f = os.path.join(folder, f"{model.name}_sheet{si + 1}.dxf")
        doc.saveas(f)
        fs = f[:-4] + ".svg"
        W, H = sh["w"] * S, sh["h"] * S
        with open(fs, "w") as fh:
            fh.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.3f}mm" height="{H:.3f}mm" '
                     f'viewBox="0 0 {W:.3f} {H:.3f}">\n<!-- blue = inner cuts (run first), red = outer cuts (run last) -->\n')
            fh.write("\n".join(svg))
            fh.write("\n</svg>\n")
        files += [f, fs]
    return files, report


def _svg_path(pts, S, H, col):
    d = "M " + " L ".join(f"{x * S:.4f} {(H - y) * S:.4f}" for x, y in pts) + " Z"
    return f'<path d="{d}" fill="none" stroke="{col}" stroke-width="0.1"/>'


def parts_json(model, path, extra=None):
    d = {"model": model.name, "units": model.units, "kind": model.kind,
         "parts": [p.info(model.mm) for p in model.parts], **(extra or {})}
    d["total_mass_g"] = round(sum(x.get("mass_g", 0) for x in d["parts"] if x["role"] != "layer"), 3)
    with open(path, "w") as f:
        json.dump(d, f, indent=1)
    return d

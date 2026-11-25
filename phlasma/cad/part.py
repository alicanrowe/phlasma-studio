"""Parts and models. A model is a flat list of named parts; each part is one solid made one way.

Part kinds
  plate : a flat 2D outline (shapely, model units) x thickness, placed by a 4x4 frame (u, v, w columns + origin).
          u is the grain / the sheet's long axis. Laser and CNC parts. Exact 3D (B-rep) and exact 2D (DXF).
  brep  : an OpenCascade solid (build123d). Furniture parts, machined jewelry, stones. Exports STEP.
  mesh  : a triangle mesh (SDF-sculpted jewelry, imported scans). Exports STL/GLB, not STEP.

Roles: part (checked for everything), stone (calibrated stone; checked for stock size, clearance, held),
bought (hardware: skipped by wall checks, still clip-checked), layer (a finish zone or enamel on a host part:
clip-checked only against parts other than its host).
Units: a model is built in its own units ("mm" or "in"); every export is converted to mm (GLB: metres, Y up).
"""
import numpy as np

UNIT_MM = {"mm": 1.0, "cm": 10.0, "in": 25.4, "m": 1000.0}

# g/cm3
DENSITY = {
    "sterling": 10.36, "silver": 10.49, "brass": 8.5, "bronze": 8.8, "gold14k": 13.1, "gold18k": 15.6,
    "titanium": 4.5, "steel": 7.85, "stainless": 8.0, "aluminium": 2.7,
    "walnut": 0.61, "white_oak": 0.75, "red_oak": 0.70, "maple": 0.70, "cherry": 0.56, "ash": 0.67,
    "mahogany": 0.55, "cedar": 0.38, "birch_ply": 0.68,
    "acrylic": 1.19, "gfrc": 2.2, "leather": 0.86,
    "diamond": 3.52, "sapphire": 4.0, "ruby": 4.0, "tourmaline": 3.1, "pinktourmaline": 3.1, "garnet": 3.9,
    "amethyst": 2.65, "topaz": 3.55, "cz": 5.8, "enamel": 2.5,
}
COLOR = {  # preview / GLB base colours (sRGB)
    "sterling": (0.80, 0.80, 0.82), "silver": (0.82, 0.82, 0.84), "brass": (0.80, 0.66, 0.36),
    "steel": (0.55, 0.56, 0.58), "stainless": (0.62, 0.63, 0.65), "titanium": (0.55, 0.55, 0.57),
    "walnut": (0.42, 0.28, 0.19), "white_oak": (0.74, 0.60, 0.43), "maple": (0.86, 0.76, 0.60),
    "cherry": (0.62, 0.38, 0.25), "ash": (0.80, 0.70, 0.55), "acrylic": (0.85, 0.92, 0.95),
    "pinktourmaline": (0.93, 0.35, 0.60), "ruby": (0.75, 0.05, 0.15), "diamond": (0.95, 0.97, 1.0),
    "sapphire": (0.15, 0.25, 0.75), "enamel_pink": (0.98, 0.45, 0.70), "enamel_lilac": (0.72, 0.60, 0.95),
}


def color_of(p):
    if p.meta.get("finish") == "rhodium":
        return COLOR["silver"]
    return COLOR.get(p.material) or COLOR.get(f"enamel_{p.meta.get('colour', '')}") or (0.7, 0.7, 0.7)


class Part:
    def __init__(self, name, kind, material="", role="part", **meta):
        self.name, self.kind, self.material, self.role, self.meta = name, kind, material, role, meta
        self.shape = None        # build123d Shape (brep)
        self.poly = None         # shapely Polygon (plate)
        self.thick = None
        self.M = np.eye(4)
        self._mesh = None

    # -------------------------------------------------------------- geometry
    def mesh(self, tol=None):
        """trimesh in model units (cached)."""
        import trimesh
        if self._mesh is not None:
            return self._mesh
        if self.kind == "plate":
            m = trimesh.creation.extrude_polygon(self.poly, self.thick)
            m.apply_transform(self.M)
        elif self.kind == "brep":
            bb = self.shape.bounding_box()
            t = tol or max(min(bb.size.X, bb.size.Y, bb.size.Z) * 0.002, 1e-4)
            v, f = self.shape.tessellate(t, 0.15)
            V = np.round(np.array([[p.X, p.Y, p.Z] for p in v]), 7)   # weld apexes/seams the mesher leaves split
            m = trimesh.Trimesh(V, np.array(f), process=True)
            m.update_faces(m.nondegenerate_faces())
            m.remove_unreferenced_vertices()
            m.merge_vertices()
        else:
            raise ValueError(f"part {self.name} has no geometry")
        if m.volume < 0:
            m.invert()
        self._mesh = m
        return m

    def brep(self):
        """OpenCascade solid (plates are extruded on demand). None for mesh parts."""
        if self.kind == "brep":
            return self.shape
        if self.kind == "plate":
            if self.shape is None:
                self.shape = plate_solid(self.poly, self.thick, self.M)
            return self.shape
        return None

    def volume(self):
        if self.kind == "brep":
            return float(self.shape.volume)
        return float(abs(self.mesh().volume))

    def info(self, scale_mm):
        m = self.mesh()
        v_mm3 = self.volume() * scale_mm ** 3
        d = {"name": self.name, "kind": self.kind, "material": self.material, "role": self.role,
             "volume_mm3": round(v_mm3, 3),
             "bbox_mm": [round(float(x) * scale_mm, 3) for x in (m.bounds[1] - m.bounds[0])]}
        if self.material in DENSITY:
            d["mass_g"] = round(v_mm3 / 1000.0 * DENSITY[self.material], 3)
        if self.kind == "plate":
            d["thick"] = self.thick
            d["outline_bbox"] = [round(x, 4) for x in self.poly.bounds]
        for k in ("grain", "host", "finish", "stone"):
            if k in self.meta:
                d[k] = self.meta[k]
        return d


class Model:
    def __init__(self, name, units="mm", kind="object", **meta):
        assert units in UNIT_MM, units
        self.name, self.units, self.kind, self.meta = name, units, kind, meta
        self.parts = []
        self.sheet = meta.get("sheet")   # laser bed, model units: {"w": 36.02, "h": 18.03, "thick": 0.25}

    @property
    def mm(self):
        return UNIT_MM[self.units]

    def add(self, part):
        if any(p.name == part.name for p in self.parts):
            raise ValueError(f"duplicate part name {part.name}")
        self.parts.append(part)
        return part

    # ---- constructors an agent calls
    def plate(self, name, poly, M, thick, material="walnut", **meta):
        p = Part(name, "plate", material, **meta)
        p.poly, p.thick, p.M = poly, float(thick), np.asarray(M, float)
        p.meta.setdefault("grain", "u")
        return self.add(p)

    def solid(self, name, shape, material="", role="part", **meta):
        p = Part(name, "brep", material, role, **meta)
        p.shape = shape
        return self.add(p)

    def mesh(self, name, mesh, material="", role="part", **meta):
        p = Part(name, "mesh", material, role, **meta)
        p._mesh = mesh
        return self.add(p)

    def get(self, name):
        return next(p for p in self.parts if p.name == name)


def frame(u, v, w, origin):
    """4x4 placing a plate: columns = where its u, v, w axes go, plus where (0, 0, 0) goes."""
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = u, v, w, origin
    return M


def ocp_transform(shape, M):
    from OCP.gp import gp_Trsf
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from build123d import Solid
    T = gp_Trsf()
    T.SetValues(*[float(x) for x in np.asarray(M, float)[:3].ravel()])
    return Solid(BRepBuilderAPI_Transform(shape.wrapped, T, True).Shape())


def plate_solid(poly, thick, M):
    from build123d import Wire, Face, Solid, Vector
    polys = list(poly.geoms) if poly.geom_type == "MultiPolygon" else [poly]
    solids = []
    for pg in polys:
        ow = Wire.make_polygon([(x, y, 0) for x, y in list(pg.exterior.coords)[:-1]], close=True)
        iw = [Wire.make_polygon([(x, y, 0) for x, y in list(r.coords)[:-1]], close=True) for r in pg.interiors]
        f = Face(ow, iw)
        solids.append(Solid.extrude(f, Vector(0, 0, thick)))
    s = solids[0]
    for o in solids[1:]:
        s = s + o
    return ocp_transform(s, M)

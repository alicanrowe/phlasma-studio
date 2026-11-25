"""JSON feature tree -> OpenCascade solid (build123d). Lets an agent build exact parts without writing Python.

A node is a dict with either "prim" (a primitive) or "op" (an operation on child nodes). Every node may carry
"at": [x, y, z] and "rot": [rx, ry, rz] (degrees, applied X then Y then Z, then moved to "at").

prims:  box {size:[x,y,z]}            centred on the origin
        cylinder {r, h}               axis Z, base at z=0 (centre:true to centre)
        cone {r1, r2, h}              axis Z, base at z=0
        sphere {r}
        torus {R, r}                  in the XY plane
        extrude {outline:[[x,y],...], holes:[[[x,y],...]], h}     sketch in XY, up Z
        revolve {profile:[[r,z],...], angle=360}                  profile in the XZ half-plane, about Z
        ref {part: "name"}            the solid of a part built earlier in the same job
        {"macro": {...}}              a jewelry block (see jewelry.macro): band, bezel, prongs
ops:    union / cut / intersect {items:[node, ...]}               cut = first minus the rest
        fillet / chamfer {item: node, r, edges: "all" | ">Z" | "<Z" | "|Z"}
        polar {item: node, n, axis: "z", start=0}                 n copies round the axis, unioned
        linear {item: node, n, step:[dx,dy,dz]}
        mirror {item: node, plane: "yz"|"xz"|"xy", keep: true}
        shell? not in v0.
"""
import math
from build123d import (Box, Cylinder, Cone, Sphere, Torus, Solid, Wire, Face, Vector, Axis, Plane, Location,
                       Rotation, fillet as _fillet, chamfer as _chamfer, mirror as _mirror)


class TreeError(ValueError):
    pass


def _place(s, node):
    rot, at = node.get("rot"), node.get("at")
    if rot:
        s = s.moved(Location((0, 0, 0), tuple(float(a) for a in rot)))
    if at:
        s = s.moved(Location(tuple(float(a) for a in at)))
    return s


def _edges(s, sel):
    E = s.edges()
    if sel in (None, "all"):
        return E
    if sel == ">Z":
        return E.group_by(Axis.Z)[-1]
    if sel == "<Z":
        return E.group_by(Axis.Z)[0]
    if sel == "|Z":
        return E.filter_by(Axis.Z)
    raise TreeError(f"unknown edge selector {sel}")


def build(node, refs=None):
    refs = refs or {}
    if not isinstance(node, dict):
        raise TreeError(f"node must be an object, got {node!r}")
    if "macro" in node:
        from .jewelry import macro
        return _place(macro(node["macro"])[0], node)
    if "prim" in node:
        k = node["prim"]
        if k == "box":
            x, y, z = node["size"]
            s = Box(x, y, z)
        elif k == "cylinder":
            s = Solid.make_cylinder(node["r"], node["h"])
            if node.get("centre") or node.get("center"):
                s = s.moved(Location((0, 0, -node["h"] / 2)))
        elif k == "cone":
            s = Solid.make_cone(node["r1"], node["r2"], node["h"])
        elif k == "sphere":
            s = Solid.make_sphere(node["r"])
        elif k == "torus":
            s = Solid.make_torus(node["R"], node["r"])
        elif k == "extrude":
            ow = Wire.make_polygon([(x, y, 0) for x, y in node["outline"]], close=True)
            iw = [Wire.make_polygon([(x, y, 0) for x, y in h], close=True) for h in node.get("holes", [])]
            s = Solid.extrude(Face(ow, iw), Vector(0, 0, node["h"]))
        elif k == "revolve":
            prof = node["profile"]
            w = Wire.make_polygon([(r, 0, z) for r, z in prof], close=True)
            s = Solid.revolve(Face(w), node.get("angle", 360), Axis.Z)
        elif k == "ref":
            if node["part"] not in refs:
                raise TreeError(f"ref to unknown part {node['part']}")
            s = refs[node["part"]]
        else:
            raise TreeError(f"unknown prim {k}")
        return _place(s, node)
    op = node.get("op")
    if op in ("union", "cut", "intersect"):
        items = [build(c, refs) for c in node["items"]]
        s = items[0]
        for o in items[1:]:
            s = s + o if op == "union" else (s - o if op == "cut" else s & o)
            if s is None:
                raise TreeError(f"{op} left nothing")
    elif op in ("fillet", "chamfer"):
        base = build(node["item"], refs)
        fn = _fillet if op == "fillet" else _chamfer
        try:
            s = fn(_edges(base, node.get("edges")), node["r"])
        except Exception as e:  # OCC fillet failures are common: report, never hang
            raise TreeError(f"{op} r={node['r']} failed on {node.get('edges', 'all')}: {e}")
    elif op == "polar":
        base = build(node["item"], refs)
        n, a0 = int(node["n"]), float(node.get("start", 0))
        s = None
        for i in range(n):
            c = base.moved(Location((0, 0, 0), (0, 0, a0 + 360.0 * i / n)))
            s = c if s is None else s + c
    elif op == "linear":
        base = build(node["item"], refs)
        dx, dy, dz = node["step"]
        s = None
        for i in range(int(node["n"])):
            c = base.moved(Location((dx * i, dy * i, dz * i)))
            s = c if s is None else s + c
    elif op == "mirror":
        base = build(node["item"], refs)
        pl = {"yz": Plane.YZ, "xz": Plane.XZ, "xy": Plane.XY}[node.get("plane", "yz")]
        m = _mirror(base, about=pl)
        s = base + m if node.get("keep", True) else m
    else:
        raise TreeError(f"node needs prim or op: {node}")
    return _place(s, node)


def solid_of(shape):
    """largest solid of a compound result, plus the solid count (a part must be one body)"""
    sol = shape.solids()
    if not sol:
        raise TreeError("no solid produced")
    return shape, len(sol)

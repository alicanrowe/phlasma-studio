"""A 10-size press-on nail set (sizes 0..9), laid out in a row for one resin print plate. Every base shares the
shape and the extension (scaled by width); the nails check measures each fit and shell."""
from phlasma.cad import Model
from phlasma.cad.nails import nail_set

PARAMS = {"shape": "almond", "ext": 6.0, "thick": 0.7, "c_curve": None, "gap": 3.0, "pitch": 0.07}


def build(p):
    m = Model("nail_set", units="mm", kind="object")
    spec = {k: p[k] for k in ("shape", "ext", "thick", "c_curve", "pitch")}
    x = 0.0
    for z, w, s, meta in nail_set(spec):
        mesh = s.mesh(voxel=p["pitch"], keep="largest")
        mesh.apply_translation((x + w / 2, 0, 0))
        x += w + p["gap"]
        m.mesh(f"size_{z}", mesh, "resin", nail=meta)
    return m

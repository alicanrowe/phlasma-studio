"""phlasma.cad tests: the prototypes pass, and planted mistakes are caught. Run: python3 tests/test_cad.py"""
import json, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phlasma.core.jobs import run_verb
from phlasma.cad import VERBS

EX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "cad")


def run(job, verb="check", **o):
    d = tempfile.mkdtemp()
    return run_verb("cad", verb, VERBS[verb], job, d, echo=False, **o)


def load(name):
    return json.load(open(os.path.join(EX, name)))


def t_solitaire_ok():
    r = run(load("solitaire.job.json"), "build")
    assert r["status"] == "ok", r.get("checks")


def t_finger_box_ok_and_one_sheet():
    r = run({"model": "examples/cad/finger_box.py"}, "build")
    assert r["status"] == "ok" and r["metrics"]["nest"]["sheets"] == 1, r


def t_planted_fingers_clash():
    r = run({"model": "examples/cad/finger_box.py", "params": {"FIT": -0.05}})   # fingers too fat: overlap
    assert not r["checks"]["interference"]["pass"], r["checks"]["interference"]


def t_planted_stone_clips():
    j = load("solitaire.job.json"); j["parts"][1]["macro"]["at"] = [0, 0, 12.6]   # stone sunk into the cup floor
    r = run(j)
    assert not r["checks"]["interference"]["pass"]


def t_planted_wrong_size():
    j = load("solitaire.job.json"); j["ring_size_us"] = 9
    assert not run(j)["checks"]["ring_size"]["pass"]


def t_planted_thin_bar():
    j = {"name": "thin", "units": "in", "kind": "furniture", "sheet": {"w": 36, "h": 18},
         "checks": {"stands": False},
         "parts": [{"name": "bar", "thick": 0.25, "outline": [[0, 0], [4, 0], [4, 1], [2.1, 1], [2.1, 3], [2, 3], [2, 1], [0, 1]]}]}
    r = run(j)
    assert not r["checks"]["laser"]["pass"], r["checks"]


def t_planted_unknown_param():
    r = run({"model": "examples/cad/finger_box.py", "params": {"Lenght": 9}})
    assert r["status"] == "bad_job"


def t_planted_stone_not_stock():
    j = load("solitaire.job.json"); j["parts"][1]["macro"]["stone"]["D"] = 6.13
    assert not run(j)["checks"]["stones"]["pass"]


def t_stool_ok():
    r = run({"model": "examples/cad/stool.py"}, "build")
    assert r["status"] == "ok", r["checks"]
    rows = json.load(open(r["outputs"]["cutlist"][0]))["rows"]
    apron = next(x for x in rows if "Apron_F" in x["parts"])
    assert abs(apron["L"] - 10.5) < 1e-6, apron          # shoulder length + both tenons


def t_planted_mortise_near_leg_end():
    r = run({"model": "examples/cad/stool.py", "params": {"shoulder_top": 0.25}})
    assert not r["checks"]["joinery"]["pass"]


def t_planted_corner_mortises_meet():
    r = run({"model": "examples/cad/stool.py", "params": {"tenon_len": 1.2}})   # long tenons: no web left
    j = r["checks"]["joinery"]
    assert not j["pass"] and (j["thin_webs"] or not r["checks"]["interference"]["pass"]), j


def t_planted_bit_too_big():
    r = run({"model": "examples/cad/finger_box.py", "cnc": {"tool_d": 0.3}})
    assert not r["checks"]["cnc"]["pass"]


def t_dogbone_relieves_corners():
    from phlasma.cad.sketch import rect, dogbone
    g = rect(0, 0, 4, 2) - rect(1, 1, 1.5, 1.5)
    d = dogbone(g, 0.125)
    assert len(d.interiors) == 1 and d.area < g.area     # reliefs join the hole, never float as islands


# ---------------------------------------------------------------- sculpting (own image-to-3D, nails, remesh)
def _imgs():
    """synthetic concept images: a disc and a tall rectangle on a light backdrop"""
    import numpy as np
    from PIL import Image
    d = tempfile.mkdtemp()
    yy, xx = np.mgrid[:300, :300]
    for name, m in (("front", (xx - 150) ** 2 + (yy - 150) ** 2 < 110 ** 2),
                    ("side", (abs(xx - 150) < 75) & (abs(yy - 150) < 110))):
        im = np.full((300, 300, 3), 240, np.uint8); im[m] = (70, 60, 60)
        Image.fromarray(im).save(os.path.join(d, name + ".png"))
    return d


def t_hull_matches_true_volume():
    import math
    from phlasma.cad.sculpt import build_scene
    d = _imgs()
    m = build_scene([{"hull": {"front": "front.png", "side": "side.png", "width": 20, "pitch": 0.25}}], d).mesh(0.25)
    want = math.pi * 10 ** 2 * (20 * 150 / 220)          # disc r 10 extruded the side's depth (150 px of 220 px tall)
    assert m.is_watertight and abs(m.volume / want - 1) < 0.03, (m.volume, want)
    assert abs(m.bounds[0][2]) < 0.2, m.bounds              # base on z = 0


def t_inflate_is_round_and_true_size():
    from phlasma.cad.sculpt import build_scene
    d = _imgs()
    m = build_scene([{"inflate": {"image": "front.png", "width": 20, "depth": 6, "pitch": 0.2}}], d).mesh(0.2)
    ext = m.bounds[1] - m.bounds[0]
    assert m.is_watertight and abs(ext[0] - 20) < 0.5 and abs(ext[1] - 6) < 0.3, ext


def t_nail_set_fits():
    r = run({"model": "examples/cad/nail_set.py", "params": {"pitch": 0.09, "shape": "square"}})
    assert r["checks"]["nails"]["pass"] and len(r["checks"]["nails"]["parts"]) == 10, r["checks"]["nails"]


def t_planted_nail_too_thick():
    r = run({"name": "n", "parts": [{"name": "n", "nail": {"size": 5, "shape": "oval", "thick": 1.8, "pitch": 0.09}}]})
    assert not r["checks"]["nails"]["pass"], r["checks"]["nails"]


def t_planted_nail_bad_shape():
    r = run({"name": "n", "parts": [{"name": "n", "nail": {"size": 5, "shape": "heart"}}]})
    assert r["status"] != "ok" and "shape" in json.dumps(r), r.get("error")


def t_nail_brushes_stay_off_the_finger():
    import numpy as np, trimesh
    from phlasma.cad.nails import under_z, outline, DEFAULTS
    nail = {"size": 6, "shape": "almond", "ext": 5, "pitch": 0.09}
    j = {"name": "n", "parts": [{"name": "n", "nail": nail, "sculpt": {"ops": [{"ball": {"c": [0, 6, 0], "r": 2.0}}]}}]}
    r = run(j, "build")
    m = trimesh.load(r["outputs"]["stl"][0])
    p = {**DEFAULTS, **nail}; pg, w, L = outline(p)
    V = m.vertices
    below = (under_z(p, w, V[:, 0], np.clip(V[:, 1], -6, L + 6), L) - V[:, 2]).max()
    assert below < 0.02, below


def t_remesh_closes_a_leaky_mesh():
    import trimesh
    d = tempfile.mkdtemp()
    s = trimesh.creation.icosphere(3, 5.0)
    s.update_faces(list(range(8, len(s.faces))))           # punch a hole
    f = os.path.join(d, "leaky.stl"); s.export(f)
    r = run({"file": f, "fit": {"size": 20}, "pitch": 0.25}, "remesh")
    assert r["checks"]["bodies"]["pass"], r["checks"]


def t_seam_split_mesh_welds_before_remesh():
    """AI exports (Tripo GLBs) are split along UV seams; unwelded, the voxel fill leaves a scrap"""
    import numpy as np, trimesh
    d = tempfile.mkdtemp()
    s = trimesh.creation.icosphere(3, 5.0)
    V = s.vertices[s.faces].reshape(-1, 3)                  # every face its own vertices, as a seam-split export
    F = np.arange(len(V)).reshape(-1, 3)
    uv = np.random.default_rng(0).random((len(V), 2))      # distinct UVs per corner keep a plain merge split
    m = trimesh.Trimesh(V, F, process=False, visual=trimesh.visual.TextureVisuals(uv=uv))
    f = os.path.join(d, "split.glb"); m.export(f)
    assert not trimesh.load(f, force="mesh").is_watertight
    r = run({"file": f, "fit": {"size": 20}, "pitch": 0.25}, "remesh")
    assert r["checks"]["survived"]["pass"] and r["checks"]["bodies"]["pass"], r["checks"]


def t_nail_extension_is_one_body_with_reach():
    j = load("nail_horn.job.json")
    j["parts"][0]["nail"]["pitch"] = 0.1; j["parts"][0]["sculpt"]["voxel"] = 0.1
    r = run(j, "build")
    info = json.load(open(r["outputs"]["parts"]))
    assert r["status"] == "ok" and r["checks"]["bodies"]["pass"], r["checks"]
    assert info["parts"][0]["bbox_mm"][1] > 2.5 * 13.3, info["parts"][0]["bbox_mm"]     # reach past the bed


if __name__ == "__main__":
    bad = 0
    for k, f in list(globals().items()):
        if k.startswith("t_"):
            try:
                print(k, f() or "ok")
            except AssertionError as e:
                bad += 1; print(k, "FAIL", str(e)[:300])
    print("failed", bad); sys.exit(1 if bad else 0)

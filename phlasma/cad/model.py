"""Load a model from a job. Two ways in, same Model out:

1. A Python model file:  {"model": "examples/cad/finger_box.py", "params": {"L": 10}}
   The file defines PARAMS (defaults) and build(p) -> Model. Params merge: PARAMS < job params < --set p.X=..
2. Pure JSON parts:       {"name": "solitaire", "units": "mm", "kind": "jewelry", "parts": [ ... ]}
   Each part: {"name", "material", "role", and one of "tree" (brep feature tree, see brep.py),
   "macro" (jewelry block, below), "outline"+"thick"+"frame" (a plate)}.
   macros: {"band": {size_us, width, thick, comfort}}, {"bezel": {D, wall, lip}}, {"prongs": {D, n, wire}},
           {"stone": {D, cut, kind}}; any macro takes "at" [x,y,z] and "n" [axis] to place it.
"""
import importlib.util, os, sys
import numpy as np

from .part import Model, frame
from . import brep, jewelry

SUITE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ModelError(ValueError):
    pass


def resolve(path, job):
    try:
        cwd = os.getcwd()
    except FileNotFoundError:     # the shell's folder was removed under it
        cwd = None
    for base in (job.get("_job_dir"), cwd, SUITE):
        if base and os.path.exists(os.path.join(base, path)):
            return os.path.join(base, path)
    if os.path.exists(path):
        return path
    raise ModelError(f"model file not found: {path}")


def params_of(job, opts, defaults):
    p = dict(defaults)
    p.update(job.get("params", {}))
    p.update(opts.get("params", {}) or {})
    for k, v in opts.items():
        if k.startswith("p."):
            p[k[2:]] = v
    unknown = sorted(set(p) - set(defaults)) if defaults else []
    if unknown:
        raise ModelError(f"unknown params {unknown}; known: {sorted(defaults)}")
    return p


def load_py(path):
    name = "phlasma_model_" + os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _macro(spec):
    try:
        return jewelry.macro(spec)
    except (ValueError, TypeError) as e:
        raise ModelError(str(e))


def from_json(job):
    m = Model(job.get("name", "model"), job.get("units", "mm"), job.get("kind", "object"),
              **{k: job[k] for k in ("sheet", "ring_size_us", "ring_axis", "checks", "allow_overlap") if k in job})
    refs = {}
    for ps in job["parts"]:
        name, mat, role = ps["name"], ps.get("material", ""), ps.get("role", "part")
        meta = {k: v for k, v in ps.items() if k in ("host", "finish", "bodies", "colour")}
        if "tree" in ps:
            try:
                s = brep.build(ps["tree"], refs)
            except brep.TreeError as e:
                raise ModelError(f"part {name}: {e}")
            refs[name] = s
            m.solid(name, s, mat, role, **meta)
        elif "macro" in ps:
            s, k, a, at, n = _macro(ps["macro"])
            if k == "stone":
                role = "stone"
                meta["stone"] = jewelry.stone_meta(a["D"], at, n, a.get("cut", "round"), a.get("kind", mat))
            refs[name] = s
            m.solid(name, s, mat, role, **meta)
        elif "outline" in ps:
            from shapely.geometry import Polygon
            pg = Polygon(ps["outline"], ps.get("holes", [])).buffer(0)
            f = ps.get("frame", {"u": [1, 0, 0], "v": [0, 1, 0], "w": [0, 0, 1], "origin": [0, 0, 0]})
            m.plate(name, pg, frame(f["u"], f["v"], f["w"], f["origin"]), ps["thick"], mat or "walnut", **meta)
        else:
            raise ModelError(f"part {name} needs tree, macro or outline")
    return m


def load(job, opts):
    m = _load(job, opts)
    if job.get("cnc"):
        m.meta["cnc"] = job["cnc"]           # {"tool_d": bit diameter, model units}
    return m


def _load(job, opts):
    if job.get("model"):
        mod = load_py(resolve(job["model"], job))
        p = params_of(job, opts, getattr(mod, "PARAMS", {}))
        m = mod.build(p)
        m.meta.setdefault("params", p)
        m.meta.setdefault("source", job["model"])
        return m
    if job.get("parts"):
        return from_json(job)
    raise ModelError('job needs "model" (a .py model file) or "parts" (JSON parts)')

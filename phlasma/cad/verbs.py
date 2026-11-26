"""phlasma cad verbs. Signature fn(job, out, log, **opts) -> {"outputs", "checks", "metrics"} (core/jobs.py)."""
import json, os, time

from phlasma.core.jobs import BadJob
from . import model as M, checks as C, export as X, preview as P

DEFAULT_EXPORTS = ["step", "stl", "glb", "dxf", "cnc", "cutlist", "preview"]


def _load(job, opts, log):
    t = time.time()
    try:
        m = M.load(job, opts)
    except (M.ModelError, KeyError, TypeError) as e:
        raise BadJob(str(e))
    log("model_built", parts=len(m.parts), units=m.units, seconds=round(time.time() - t, 2))
    return m


def _checks(m, log):
    t = time.time()
    res = C.run_all(m, {})
    for k, v in res.items():
        log("check", "info" if v.get("pass", True) else "warn", name=k, ok=v.get("pass", True))
    log("checks_done", seconds=round(time.time() - t, 2))
    return res


def build(job, out, log, **opts):
    """build a model, run every check, write the exports and a preview"""
    m = _load(job, opts, log)
    exports = job.get("export", DEFAULT_EXPORTS)
    res = _checks(m, log) if job.get("check", True) else {}
    outs = {}
    if "step" in exports:
        f = X.step(m, os.path.join(out, f"{m.name}.step"))
        if f:
            outs["step"] = f
    if "stl" in exports:
        outs["stl"] = X.stl(m, os.path.join(out, "stl"))
    if "glb" in exports:
        outs["glb"] = X.glb(m, os.path.join(out, f"{m.name}.glb"))
    nest_rep = {}
    if "dxf" in exports and m.sheet:
        files, nest_rep = X.laser_sheets(m, os.path.join(out, "cut"), kerf=job.get("kerf", m.sheet.get("kerf", 0.0)))
        outs["cut"] = files
        if nest_rep.get("too_big"):
            res.setdefault("laser", {"pass": True})
            res["laser"]["pass"] = False
            res["laser"]["too_big"] = nest_rep["too_big"]
    cnc = job.get("cnc") or (m.sheet or {}).get("cnc")
    if "cnc" in exports and m.sheet and cnc:
        files, _ = X.laser_sheets(m, os.path.join(out, "cut_cnc"), kerf=0.0, dogbone_d=cnc["tool_d"], suffix="_cnc")
        outs["cnc"] = files
    if "cutlist" in exports and m.kind == "furniture":
        outs["cutlist"], rows = X.cutlist(m, os.path.join(out, "build"))
    info = X.parts_json(m, os.path.join(out, "parts.json"), {"checks": res, "nest": nest_rep})
    outs["parts"] = os.path.join(out, "parts.json")
    if "preview" in exports:
        outs["preview"] = P.render(m, os.path.join(out, "preview.png"))
    log("exported", files=sum(len(v) if isinstance(v, list) else 1 for v in outs.values()))
    metrics = {"parts": len(m.parts), "units": m.units, "total_mass_g": info["total_mass_g"],
               "params": m.meta.get("params"), "nest": nest_rep or None}
    return {"outputs": outs, "checks": res, "metrics": metrics}


def check(job, out, log, **opts):
    """checks only (fast loop while designing): same job as build, no exports"""
    m = _load(job, opts, log)
    res = _checks(m, log)
    info = X.parts_json(m, os.path.join(out, "parts.json"), {"checks": res})
    return {"outputs": {"parts": os.path.join(out, "parts.json")}, "checks": res,
            "metrics": {"parts": len(m.parts), "total_mass_g": info["total_mass_g"]}}


def check_mesh(job, out, log, **opts):
    """check existing files: {"files": ["a.stl", "b.glb"], "units": "mm", "kind": "jewelry", "stones": [...]}"""
    import trimesh
    from .part import Model
    m = Model(job.get("name", "imported"), job.get("units", "mm"), job.get("kind", "object"),
              checks=job.get("checks", {}))
    for f in job["files"]:
        f = M.resolve(f, job)
        sc = trimesh.load(f, force="scene")
        for nm, g in sc.geometry.items():
            m.mesh(f"{os.path.splitext(os.path.basename(f))[0]}:{nm}", g, job.get("material", ""))
    res = _checks(m, log)
    return {"outputs": {}, "checks": res, "metrics": {"parts": len(m.parts)}}


def schema(job, out, log, **opts):
    """describe what the cad module takes, for an agent discovering it"""
    from . import brep, model, sketch, jewelry, sdf
    d = {"verbs": {k: (v.__doc__ or "").strip() for k, v in VERBS.items()},
         "job": model.__doc__, "brep_tree": brep.__doc__, "sketch": sketch.__doc__, "sdf": sdf.__doc__,
         "jewelry": jewelry.__doc__, "checks": C.__doc__, "exports": DEFAULT_EXPORTS,
         "examples": sorted(os.listdir(os.path.join(M.SUITE, "examples", "cad")))}
    f = os.path.join(out, "schema.json")
    with open(f, "w") as fh:
        json.dump(d, fh, indent=1)
    return {"outputs": {"schema": f}, "checks": {}, "metrics": {}}


VERBS = {"build": build, "check": check, "check_mesh": check_mesh, "schema": schema}

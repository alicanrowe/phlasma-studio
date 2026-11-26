"""phlasma vfx <verb> <job.json> --out DIR

  ops       the operator catalog as JSON (what an agent reads instead of a GUI)
  validate  every problem in a graph at once, shaders compiled, nothing rendered
  render    frames -> out.mp4 (+ audio), preview.png, contact.png, per-frame and per-node timing, checks
  probe     one or more moments -> every node's image in nodes_<t>.png, per-node stats, signal values
  patch     apply small edits to a graph, validate, write the new graph
  live      real-time loop that hot-reloads the graph file on save; writes live/latest.png + status.json

Jobs: {"graph": "path/to/graph.json" | {inline graph}, ...verb options}. --set key=value overrides a job key.
"""
import json, math, os, time

import numpy as np

from phlasma.core.jobs import BadJob

from . import graph as G
from . import media
from .ops import catalog
from .runtime import Engine, GraphError, NodeError

_GPU = {}


def _gpu(device):
    from .gpu import GPU
    if device not in _GPU:
        _GPU[device] = GPU(device)
    return _GPU[device]


def _job(job, opts):
    job = dict(job)
    for k, v in opts.items():
        if k != "device":
            job[k] = v
    return job


def _graph(job, need=True):
    src = job.get("graph")
    if src is None:
        if need:
            raise BadJob('job needs "graph": a path or an inline graph object')
        return None
    base = job.get("_dir") or os.getcwd()
    if isinstance(src, str):
        p = src if os.path.isabs(src) else os.path.join(base, src)
        if not os.path.exists(p):
            raise BadJob(f"graph file not found: {p}")
        try:
            g = G.load(p)
        except json.JSONDecodeError as e:
            raise BadJob(f"graph is not valid JSON: {e}") from None
    else:
        g = G.load(src, base)
    for k in ("res", "fps", "duration"):
        if k in job:
            g[k] = job[k]
    return g


def _engine(g, log, device, only_live=True):
    """Engine or a structured failure. Returns (engine, issues)."""
    try:
        e = Engine(g, _gpu(device), only_live=only_live)
    except GraphError as ge:
        for i in ge.issues:
            log("issue", i["level"], **i)
        raise BadJob(json.dumps([i for i in ge.issues if i["level"] == "error"]))
    except NodeError as ne:
        log("issue", "error", **ne.issue())
        raise BadJob(json.dumps([ne.issue()]))
    for i in e.issues:
        log("issue", i["level"], **i)
    return e, e.issues


# ------------------------------------------------------------------------------------------------ verbs
def v_ops(job, out, log, **opts):
    c = catalog()
    media.dump(c, os.path.join(out, "ops.json"))
    log("ops", count=len(c["ops"]))
    return {"outputs": {"ops": os.path.join(out, "ops.json")}, "ops": c}


def v_validate(job, out, log, device=None, **opts):
    job = _job(job, opts)
    g = _graph(job)
    issues, _ = G.validate(g)
    errs = [i for i in issues if i["level"] == "error"]
    if not errs:  # compile shaders too
        try:
            e = Engine(g, _gpu(device))
            e.close()
        except NodeError as ne:
            errs.append(ne.issue())
            issues.append(ne.issue())
    for i in issues:
        log("issue", i["level"], **i)
    return {"issues": issues, "checks": {"graph": {"pass": not errs, "errors": len(errs),
                                                    "warnings": len(issues) - len(errs)}}}


def v_render(job, out, log, device=None, **opts):
    job = _job(job, opts)
    g = _graph(job)
    eng, issues = _engine(g, log, device)
    fps = eng.fps
    start = int(job.get("start", 0))
    n = int(job.get("frames") or math.ceil(float(job.get("duration", eng.dur)) * fps))
    video = job.get("video", True) and n > 1
    every = int(job.get("png_every", 0))
    ncontact = int(job.get("contact", 12))
    profile = job.get("profile", True)
    log("start", frames=n, start=start, res=[eng.w, eng.h], fps=fps, adapter=eng.gpu.info)
    for f in range(start):  # pre-roll signals so stateful ones (lag, speed, envelope) match a full run
        eng.step_signals(f)
    picks = set(np.linspace(start, start + n - 1, min(ncontact, n)).round().astype(int).tolist()) if ncontact else set()
    vw = media.VideoWriter(os.path.join(out, "out.mp4"), eng.w, eng.h, fps) if video else None
    if every:
        os.makedirs(os.path.join(out, "frames"), exist_ok=True)
    ft, nan_frames, black, means, sheet, labels = [], [], [], [], [], []
    prev, still = None, 0
    try:
        for f in range(start, start + n):
            t0 = time.perf_counter()
            eng.render_frame(f, timing=profile)
            img = eng.read()
            ft.append((time.perf_counter() - t0) * 1000)
            if not np.isfinite(img).all():
                nan_frames.append(f)
            m = float(np.nan_to_num(img[..., :3]).mean())
            means.append(m)
            if m < 0.004:
                black.append(f)
            small = img[::8, ::8, :3]
            if prev is not None and float(np.abs(small - prev).mean()) < 1e-4:
                still += 1
            prev = small
            if vw:
                vw.write(img)
            if every and (f - start) % every == 0:
                media.save_png(img, os.path.join(out, "frames", f"{f:05d}.png"))
            if f in picks:
                sheet.append(img)
                labels.append(f"f{f}  {f / fps:.2f}s")
            if f == start + n // 2 or n == 1:
                media.save_png(img, os.path.join(out, "preview.png"))
            if (f - start) % max(int(fps), 1) == 0:
                log("progress", frame=f, done=f - start + 1, of=n, ms=round(ft[-1], 1))
    finally:
        eng.close()
    outputs = {"preview": os.path.join(out, "preview.png")}
    if vw:
        a, off = eng.audio_source() if job.get("audio", True) else (None, 0)
        vw.close(audio=a, audio_offset=off + start / fps)
        outputs["video"] = os.path.join(out, "out.mp4")
        outputs["audio_muxed"] = bool(a)
    if sheet:
        media.contact_sheet(sheet, labels, os.path.join(out, "contact.png"))
        outputs["contact"] = os.path.join(out, "contact.png")
    if every:
        outputs["frames"] = os.path.join(out, "frames")
    arr = np.array(ft)
    budget = 1000 / fps
    node_ms = {k: round(v / n, 2) for k, v in sorted(eng.timing.items(), key=lambda kv: -kv[1]) if v} if profile else {}
    perf = {"frame_ms_mean": round(float(arr.mean()), 2), "frame_ms_p95": round(float(np.percentile(arr, 95)), 2),
            "frame_ms_max": round(float(arr.max()), 2), "budget_ms": round(budget, 2),
            "realtime_x": round(budget / float(arr.mean()), 2), "node_ms_mean": node_ms,
            "note": "includes readback + encode; profile=true adds a GPU sync per node",
            "adapter": eng.gpu.info}
    expect_motion = job.get("expect_motion", n > 1)
    warm = arr[1:] if len(arr) > 2 else arr
    warm_p95 = float(np.percentile(warm, 95))
    informational = eng.gpu.is_cpu or job.get("realtime_check", True) is False
    checks = {
        "nan_frames": {"pass": not nan_frames, "frames": nan_frames[:20]},
        "black_frames": {"pass": len(black) < n, "count": len(black), "frames": black[:20],
                         "note": "fails only if every frame is black"},
        "motion": {"pass": (still < n - 1) or not expect_motion, "still_pairs": still,
                   "note": "fails if no frame differs from the last (set expect_motion=false for stills)"},
        "frame_time": {"pass": True if informational else warm_p95 <= budget,
                       "p95_ms": round(warm_p95, 2), "budget_ms": round(budget, 2),
                       "informational": informational,
                       "note": "first frame (pipeline compile) excluded; informational on CPU or with realtime_check=false"},
    }
    if any(i["level"] == "error" for i in issues):
        checks["graph"] = {"pass": False}
    log("perf", **{k: v for k, v in perf.items() if k != "node_ms_mean"})
    return {"outputs": outputs, "checks": checks, "metrics": perf, "issues": issues,
            "brightness": {"mean": round(float(np.mean(means)), 4), "min": round(float(np.min(means)), 4),
                           "max": round(float(np.max(means)), 4)}}


def v_probe(job, out, log, device=None, **opts):
    job = _job(job, opts)
    g = _graph(job)
    eng, issues = _engine(g, log, device, only_live=False)
    fps = eng.fps
    if "frames" in job:
        want = [int(x) for x in job["frames"]]
    else:
        ts = job.get("t", [0.0])
        want = [int(round(float(x) * fps)) for x in (ts if isinstance(ts, list) else [ts])]
    pick = job.get("nodes", "all")
    res = {"moments": []}
    outputs = {}
    f = 0
    for target in sorted(want):
        eng.timing = {k: 0.0 for k in eng.order}
        while f <= target:  # run every frame up to the target so feedback and stateful signals are real
            eng.render_frame(f, timing=(f == target))
            f += 1
        items, per = [], {}
        for nid in eng.order:
            if pick != "all" and nid not in pick:
                continue
            if nid in eng.tex:
                a = eng.read(nid)
                s = media.stats(a)
                iss = "NaN pixels" if s["nan_px"] else ("all black" if max(s["mean"][:3]) < 0.002 and s["mean"][3] > 0.5 else None)
                items.append({"id": nid, "op": eng.nodes[nid]["op"], "img": a, "ms": eng.timing[nid], "stats": s,
                              "issue": iss})
                per[nid] = {"op": eng.nodes[nid]["op"], "ms": round(eng.timing[nid], 2), **s}
                if iss:
                    per[nid]["issue"] = iss
        tag = f"{target / fps:.2f}s".replace(".", "_")
        p = os.path.join(out, f"nodes_{tag}.png")
        media.node_sheet(items, p)
        op = os.path.join(out, f"out_{tag}.png")
        media.save_png(eng.read(), op)
        outputs[f"nodes_{tag}"], outputs[f"out_{tag}"] = p, op
        res["moments"].append({"frame": target, "t": round(target / fps, 3), "signals": {
            k: {c: round(v, 4) for c, v in d.items()} for k, d in eng.sig.items()}, "nodes": per})
        log("probe", frame=target, nodes=len(items))
    eng.close()
    res["issues"] = issues
    res["outputs"] = outputs
    nan = [n for m in res["moments"] for n, d in m["nodes"].items() if d["nan_px"]]
    res["checks"] = {"nan_nodes": {"pass": not nan, "nodes": sorted(set(nan))}}
    return res


def v_patch(job, out, log, device=None, **opts):
    job = _job(job, opts)
    g = _graph(job)
    try:
        g2, applied = G.patch(g, job.get("edits", []))
    except (KeyError, ValueError) as e:
        raise BadJob(f"edit failed: {e!r}") from None
    g2["_dir"] = g["_dir"]
    issues, _ = G.validate(g2)
    errs = [i for i in issues if i["level"] == "error"]
    p = os.path.join(out, "graph.json")
    G.save(g2, p)
    outputs = {"graph": p}
    if job.get("write") and not errs:
        w = job["write"] if os.path.isabs(job["write"]) else os.path.join(job.get("_dir") or os.getcwd(), job["write"])
        G.save(g2, w)
        outputs["written"] = w
    for i in issues:
        log("issue", i["level"], **i)
    return {"outputs": outputs, "applied": len(applied), "issues": issues,
            "checks": {"graph": {"pass": not errs, "errors": len(errs)}}}


def v_live(job, out, log, device=None, **opts):
    """Real-time, headless, hot-reloading. An agent edits graph.json; this picks it up on the next frame, keeps the
    last good graph if the new one is broken, and writes what it sees: live/latest.png (every snapshot_every s) and
    live/status.json (fps, frame, errors, signals). Stop with seconds, or by deleting live/RUN."""
    job = _job(job, opts)
    src = job.get("graph")
    if not isinstance(src, str):
        raise BadJob("live needs graph as a file path (it watches the file)")
    path = src if os.path.isabs(src) else os.path.join(job.get("_dir") or os.getcwd(), src)
    seconds = float(job.get("seconds", 30))
    snap = float(job.get("snapshot_every", 1.0))
    pace = job.get("pace", True)
    d = os.path.join(out, "live")
    os.makedirs(d, exist_ok=True)
    runf = os.path.join(d, "RUN")
    open(runf, "w").close()
    eng, mtime, err, reloads = None, 0, None, 0
    t_start = time.perf_counter()
    last_snap, frame, ft, dropped, t_eng = -1e9, 0, [], 0, 0.0
    while time.perf_counter() - t_start < seconds and os.path.exists(runf):
        m = os.path.getmtime(path)
        if m != mtime:
            mtime = m
            try:
                g = _graph({"graph": path})
                new = Engine(g, _gpu(device))
                if eng:
                    eng.close()
                eng, err, frame, t_eng = new, None, 0, time.perf_counter()
                reloads += 1
                log("reload", n=reloads, warnings=[i for i in new.issues])
            except (GraphError, NodeError, BadJob) as e:
                err = e.issues if isinstance(e, GraphError) else [e.issue()] if isinstance(e, NodeError) else str(e)
                log("reload_failed", "error", issues=err, kept_previous=eng is not None)
        if eng is None:
            time.sleep(0.2)
            continue
        t0 = time.perf_counter()
        loop_n = max(int(eng.dur * eng.fps), 1)
        if pace:  # wall-clock locked: if a frame runs long, skip ahead (signals still step every frame)
            want = int((t0 - t_eng) * eng.fps)
            while frame < want:
                if frame % loop_n == 0 and frame:
                    eng.state = {k: ({"feats": v["feats"], "file": v["file"]} if "feats" in v else {})
                                 for k, v in eng.state.items()}
                eng.step_signals(frame % loop_n)
                frame += 1
                dropped += 1
        eng.render_frame(frame % loop_n)
        frame += 1
        now = time.perf_counter()
        ft.append((now - t0) * 1000)
        if now - last_snap >= snap:
            last_snap = now
            media.save_png(eng.read(), os.path.join(d, "latest.png"))
            media.dump({"frame": frame, "fps_actual": round(1000 / (np.mean(ft[-30:]) or 1), 1), "target_fps": eng.fps,
                        "errors": err, "reloads": reloads, "dropped": dropped, "signals": eng.sig,
                        "graph_mtime": mtime}, os.path.join(d, "status.json"))
        if pace:
            time.sleep(max(0.0, 1 / eng.fps - (time.perf_counter() - t0)))
    if eng:
        eng.close()
    if os.path.exists(runf):
        os.remove(runf)
    return {"outputs": {"latest": os.path.join(d, "latest.png"), "status": os.path.join(d, "status.json")},
            "metrics": {"frames": frame, "dropped": dropped, "reloads": reloads, "frame_ms_mean": round(float(np.mean(ft)), 2) if ft else None},
            "checks": {"last_graph_ok": {"pass": err is None, "errors": err}}}


VERBS = {"ops": v_ops, "validate": v_validate, "render": v_render, "probe": v_probe, "patch": v_patch, "live": v_live}

"""Job in, run folder out. Every verb goes through run_verb so run.json/log.jsonl look the same in every module."""
import hashlib, json, os, platform, time, traceback

from .log import RunLog

EXIT = {"ok": 0, "bad_job": 2, "check_failed": 3, "error": 4}


class BadJob(Exception):
    pass


def load_job(src):
    """dict, path, '-' (stdin) or None/'' (empty job, for introspection verbs). Sets job['_dir'] for relative paths."""
    if isinstance(src, dict):
        return src
    if src in (None, "", "{}"):
        return {"_dir": os.getcwd()}
    if src == "-":
        import sys
        txt = sys.stdin.read().strip()
        job = json.loads(txt) if txt else {}
        job.setdefault("_dir", os.getcwd())
        return job
    with open(src) as f:
        job = json.load(f)
    job.setdefault("_path", os.path.abspath(src))
    job.setdefault("_dir", os.path.dirname(os.path.abspath(src)))
    job.setdefault("_job_dir", job["_dir"])
    return job


def job_hash(job):
    return hashlib.sha1(json.dumps({k: v for k, v in job.items() if not k.startswith("_")}, sort_keys=True, default=str).encode()).hexdigest()[:12]


def run_verb(module, verb, fn, job_src, out, **opts):
    """fn(job, out_dir, log, **opts) -> dict(outputs=..., checks=..., metrics=...)."""
    job = load_job(job_src)
    os.makedirs(out, exist_ok=True)
    log = RunLog(os.path.join(out, "log.jsonl"), echo=opts.pop("echo", True))
    run = {"module": module, "verb": verb, "job_hash": job_hash(job), "host": platform.node(),
           "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "opts": opts, "status": "running"}
    t0 = time.time()
    try:
        schema = getattr(fn, "SCHEMA", None)
        if schema is not None:
            import jsonschema
            try:
                jsonschema.validate(job, schema)
            except jsonschema.ValidationError as e:
                raise BadJob(f"schema: {e.message} at {list(e.absolute_path)}") from None
        res = fn(job, out, log, **opts) or {}
        run.update(res)
        failed = [k for k, v in (res.get("checks") or {}).items() if isinstance(v, dict) and not v.get("pass", True)]
        run["status"] = "check_failed" if failed else "ok"
        if failed:
            run["failed_checks"] = failed
    except BadJob as e:
        run.update(status="bad_job", error=str(e))
        log("bad_job", "error", msg=str(e))
    except Exception as e:  # noqa: BLE001 - a run never dies silently
        run.update(status="error", error=repr(e), trace=traceback.format_exc())
        log("error", "error", msg=repr(e))
    run["seconds"] = round(time.time() - t0, 2)
    with open(os.path.join(out, "run.json"), "w") as f:
        json.dump(run, f, indent=1, default=str)
    log("done", status=run["status"], seconds=run["seconds"])
    log.close()
    return run

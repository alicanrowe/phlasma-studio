"""phlasma <module> <verb> <job.json> --out DIR [--device cuda|cpu] [--json] [--set key=value ...]"""
import argparse, importlib, json, sys

from .jobs import EXIT, run_verb

MODULES = ("garment", "vfx", "cad")


def verbs(module):
    m = importlib.import_module(f"phlasma.{module}")
    return getattr(m, "VERBS", {})


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help", "list"):
        out = {}
        for m in MODULES:
            try:
                out[m] = sorted(verbs(m))
            except Exception as e:  # module not built yet
                out[m] = f"unavailable: {e!r}"
        print(json.dumps(out, indent=1))
        return 0
    if argv[0] == "mcp":
        from .mcp import serve
        serve()
        return 0
    if argv[0] == "batch":
        from .queue import main as qmain
        return qmain(argv[1:])
    ap = argparse.ArgumentParser(prog="phlasma")
    ap.add_argument("module", choices=MODULES)
    ap.add_argument("verb")
    ap.add_argument("job", nargs="?", default=None, help="job.json, - for stdin, omit for {}")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--set", action="append", default=[], help="override opts, key=value (JSON value)")
    a = ap.parse_args(argv)
    V = verbs(a.module)
    if a.verb not in V:
        print(json.dumps({"status": "bad_job", "error": f"unknown verb {a.verb}", "verbs": sorted(V)}))
        return EXIT["bad_job"]
    opts = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        try:
            opts[k] = json.loads(v)
        except json.JSONDecodeError:
            opts[k] = v
    if a.device:
        opts["device"] = a.device
    if a.out is None:
        import tempfile
        a.out = tempfile.mkdtemp(prefix=f"phlasma_{a.module}_{a.verb}_")
    run = run_verb(a.module, a.verb, V[a.verb], a.job, a.out, echo=not a.json, **opts)
    if a.json:
        print(json.dumps(run, default=str))
    return EXIT.get(run["status"], 4)


if __name__ == "__main__":
    sys.exit(main())

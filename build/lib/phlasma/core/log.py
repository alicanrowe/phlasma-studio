"""JSONL event log. One JSON object per line; never prompts, never blocks."""
import json, sys, time


class RunLog:
    def __init__(self, path=None, echo=True):
        self.f = open(path, "a", buffering=1) if path else None
        self.echo, self.t0 = echo, time.time()

    def __call__(self, ev, lvl="info", **kw):
        rec = {"t": round(time.time() - self.t0, 3), "lvl": lvl, "ev": ev, **kw}
        line = json.dumps(rec, default=str)
        if self.f:
            self.f.write(line + "\n")
        if self.echo:
            print(line, file=sys.stderr, flush=True)
        return rec

    def close(self):
        if self.f:
            self.f.close()

"""`phlasma mcp`: a dependency-free MCP server over stdio (JSON-RPC 2.0), one tool per module verb.

Tool name: <module>_<verb>. Input: {"job": <object or path>, "out": <dir, optional>, "opts": {...}}.
Result: the run.json of the run (status, checks, metrics, outputs) as JSON text.
"""
import json
import sys
import tempfile

from .cli import MODULES, verbs
from .jobs import run_verb

PROTO = "2024-11-05"


def _tools():
    out = []
    for m in MODULES:
        try:
            V = verbs(m)
        except Exception:
            continue
        for name, fn in V.items():
            out.append({
                "name": f"{m}_{name}",
                "description": (fn.__doc__ or f"{m} {name}").strip().split("\n")[0],
                "inputSchema": {"type": "object", "properties": {
                    "job": {"description": "job object, or path to a job .json", "type": ["object", "string"]},
                    "out": {"type": "string", "description": "run folder (default: a new temp folder)"},
                    "opts": {"type": "object", "description": "option overrides, same as --set"}},
                    "required": []},
                "_fn": fn, "_m": m, "_v": name})
    return out


def serve():
    tools = _tools()
    by = {t["name"]: t for t in tools}
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        rid, meth, p = req.get("id"), req.get("method"), req.get("params") or {}
        res, err = None, None
        if meth == "initialize":
            res = {"protocolVersion": PROTO, "capabilities": {"tools": {}},
                   "serverInfo": {"name": "phlasma", "version": "0.1.0"}}
        elif meth == "tools/list":
            res = {"tools": [{k: v for k, v in t.items() if not k.startswith("_")} for t in tools]}
        elif meth == "tools/call":
            t = by.get(p.get("name"))
            if not t:
                err = {"code": -32602, "message": f"unknown tool {p.get('name')}"}
            else:
                a = p.get("arguments") or {}
                out = a.get("out") or tempfile.mkdtemp(prefix=f"phlasma_{t['name']}_")
                run = run_verb(t["_m"], t["_v"], t["_fn"], a.get("job") or {}, out, echo=False, **(a.get("opts") or {}))
                run["out"] = out
                res = {"content": [{"type": "text", "text": json.dumps(run, default=str)}],
                       "isError": run.get("status") not in ("ok",)}
        elif rid is None:  # notification
            continue
        else:
            err = {"code": -32601, "message": f"no method {meth}"}
        msg = {"jsonrpc": "2.0", "id": rid}
        msg.update({"error": err} if err else {"result": res})
        sys.stdout.write(json.dumps(msg) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    serve()

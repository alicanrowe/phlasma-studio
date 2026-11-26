"""The graph is a plain JSON file an agent can write, diff and patch.

{
  "phlasma_vfx": 1,
  "res": [1280, 720], "fps": 30, "duration": 8,
  "nodes": {
    "aud":   {"op": "audio", "params": {"file": "beat.wav"}},
    "n":     {"op": "noise", "params": {"scale": "3 + aud.low * 4"}},
    "fb":    {"op": "feedback", "params": {"source": "mix"}},
    "warp":  {"op": "transform", "inputs": ["fb"], "params": {"scale": [1.01, 1.01], "rotate": 0.3}},
    "mix":   {"op": "composite", "inputs": ["n", "warp"], "params": {"mode": "max"}}
  },
  "output": "mix"
}

Validation never stops at the first problem: it returns every issue as {"node", "param", "code", "msg", "hint"}.
"""
import copy, difflib, json, os

from .expr import BUILTIN_NAMES, CONSTS, FUNCS, Expr, ExprError
from .ops import OPS, SIGNAL, parse_hex

DEFAULTS = {"res": [1280, 720], "fps": 30, "duration": 4}
NODE_KEYS = {"op", "inputs", "params", "note", "pos", "bypass"}


def load(src, base=None):
    if isinstance(src, dict):
        g = copy.deepcopy(src)
        g.setdefault("_dir", base or os.getcwd())
        return g
    with open(src) as f:
        g = json.load(f)
    g["_dir"] = os.path.dirname(os.path.abspath(src))
    g["_path"] = os.path.abspath(src)
    return g


def save(g, path):
    out = {k: v for k, v in g.items() if not k.startswith("_")}
    with open(path, "w") as f:
        json.dump(out, f, indent=1)


def near(name, options):
    m = difflib.get_close_matches(name, list(options), n=1, cutoff=0.5)
    return f"did you mean '{m[0]}'?" if m else None


class Issue(dict):
    def __init__(self, code, msg, node=None, param=None, hint=None, level="error"):
        super().__init__(level=level, code=code, msg=msg)
        if node:
            self["node"] = node
        if param:
            self["param"] = param
        if hint:
            self["hint"] = hint


def compile_value(prm, v):
    """Turn a JSON param value into ('const', floats) or ('expr', [Expr|float...]) for numeric types."""
    t = prm.type
    if t == "string":
        if not isinstance(v, str):
            raise ExprError("expected a string")
        return v
    if t == "node":
        if not isinstance(v, str):
            raise ExprError("expected a node id")
        return v
    if t == "enum":
        if isinstance(v, str) and v in prm.choices:
            return [float(prm.choices.index(v))]
        if isinstance(v, (int, float)) and 0 <= int(v) < len(prm.choices):
            return [float(int(v))]
        raise ExprError(f"'{v}' is not one of {prm.choices}")
    if t == "bool":
        if isinstance(v, bool):
            return [1.0 if v else 0.0]
    if t == "color" and isinstance(v, str):
        h = parse_hex(v)
        if h:
            return h
    n = {"float": 1, "int": 1, "bool": 1, "vec2": 2, "vec3": 3, "color": 4}[t]
    vals = v if isinstance(v, list) else [v]
    if t == "color" and len(vals) == 3:
        vals = vals + [1.0]
    if t in ("vec2", "vec3") and len(vals) == 1:
        vals = vals * n
    if len(vals) != n:
        raise ExprError(f"expected {n} values, got {len(vals)}")
    out = []
    for x in vals:
        if isinstance(x, bool):
            out.append(1.0 if x else 0.0)
        elif isinstance(x, (int, float)):
            out.append(float(x))
        elif isinstance(x, str):
            out.append(Expr(x))
        else:
            raise ExprError(f"bad value {x!r}")
    return out


def validate(g):
    """Return (issues, compiled) where compiled = {node: {param: compiled_value}} plus order."""
    issues = []
    for k in g:
        if not k.startswith("_") and k not in ("phlasma_vfx", "res", "fps", "duration", "nodes", "output", "seed",
                                                "note", "outputs", "audio"):
            issues.append(Issue("unknown_key", f"unknown top-level key '{k}'", hint=near(k, DEFAULTS), level="warn"))
    for k, v in DEFAULTS.items():
        g.setdefault(k, v)
    nodes = g.get("nodes")
    if not isinstance(nodes, dict) or not nodes:
        return [Issue("no_nodes", "graph has no 'nodes' object")], None
    compiled, deps = {}, {}
    for nid, n in nodes.items():
        if not isinstance(n, dict) or "op" not in n:
            issues.append(Issue("bad_node", "node must be an object with an 'op'", nid))
            continue
        if not nid.replace("_", "").isalnum() or nid[0].isdigit() or nid in FUNCS or nid in CONSTS or nid in BUILTIN_NAMES:
            issues.append(Issue("bad_id", f"node id '{nid}' must be a plain identifier and not a builtin name", nid))
        o = OPS.get(n["op"])
        if not o:
            issues.append(Issue("unknown_op", f"unknown op '{n['op']}'", nid, hint=near(n["op"], OPS)))
            continue
        for k in n:
            if k not in NODE_KEYS:
                h = "params go inside \"params\": {...}" if k in o.pmap else near(k, NODE_KEYS)
                issues.append(Issue("unknown_key", f"unknown node key '{k}'", nid, hint=h))
        ins = n.get("inputs", [])
        if not isinstance(ins, list):
            issues.append(Issue("bad_inputs", "'inputs' must be a list of node ids", nid))
            ins = []
        nmax = o.inputs if o.name != "wgsl" else 4
        if o.family == "signal" and ins:
            issues.append(Issue("bad_inputs", "signal ops take no inputs; reference other signals in expressions", nid))
        elif len(ins) > nmax:
            issues.append(Issue("too_many_inputs", f"'{o.name}' takes {nmax} input(s), got {len(ins)}", nid))
        d = set()
        for i, src in enumerate(ins):
            if src is None:
                continue
            if src not in nodes:
                issues.append(Issue("missing_input", f"input {i} '{src}' is not a node", nid, hint=near(src, nodes)))
            elif nodes[src].get("op") in SIGNAL:
                issues.append(Issue("wrong_family", f"input {i} '{src}' is a signal; image inputs need image nodes", nid))
            else:
                d.add(src)
        params = n.get("params", {}) or {}
        cp = {}
        for pk, pv in params.items():
            prm = o.pmap.get(pk)
            if not prm:
                issues.append(Issue("unknown_param", f"'{o.name}' has no param '{pk}'", nid, pk, near(pk, o.pmap)))
                continue
            try:
                cp[pk] = compile_value(prm, pv)
            except ExprError as e:
                issues.append(Issue("bad_value", str(e), nid, pk))
                continue
            if isinstance(cp[pk], list):
                for x in cp[pk]:
                    if isinstance(x, Expr):
                        for r in x.refs:
                            if r not in nodes:
                                issues.append(Issue("unknown_name", f"'{r}' in {x.src!r} is not a node or builtin", nid,
                                                    pk, near(r, list(nodes) + list(FUNCS) + list(BUILTIN_NAMES))))
                            elif nodes[r].get("op") not in SIGNAL:
                                issues.append(Issue("wrong_family", f"'{r}' is an image node; expressions read signals",
                                                    nid, pk))
                            else:
                                d.add(r)
                        for r, c in x.chans:
                            ch = OPS[nodes[r]["op"]].channels if r in nodes and nodes[r].get("op") in OPS else None
                            if ch and nodes[r].get("op") in SIGNAL and c not in ch:
                                issues.append(Issue("unknown_channel", f"'{r}' ({nodes[r]['op']}) has no channel '{c}'",
                                                    nid, pk, near(c, ch) or f"channels: {', '.join(ch)}"))
            if prm.type == "node":
                if pv not in nodes:
                    issues.append(Issue("missing_source", f"'{pv}' is not a node", nid, pk, near(pv, nodes)))
            if prm.type == "string" and "{" in str(pv) and o.name == "text":
                import re as _re
                for fld in _re.findall(r"\{([^{}:]+)(?::[^{}]*)?\}", pv):
                    try:
                        e = Expr(fld)
                    except ExprError as ex:
                        issues.append(Issue("bad_value", str(ex), nid, pk))
                        continue
                    for r in e.refs:
                        if r in nodes and nodes[r].get("op") in SIGNAL:
                            d.add(r)
                        else:
                            issues.append(Issue("unknown_name", f"'{r}' in text field is not a signal node", nid, pk))
            if prm.type in ("string",) and pk == "file" and pv:
                fp = os.path.join(g["_dir"], pv)
                if not os.path.exists(fp):
                    issues.append(Issue("missing_file", f"file not found: {fp}", nid, pk))
        if o.name == "feedback" and "source" not in params:
            issues.append(Issue("missing_source", "feedback needs params.source (the node it loops back from)", nid))
        compiled[nid] = cp
        deps[nid] = d
    out = g.get("output")
    if not out:
        issues.append(Issue("no_output", "graph needs \"output\": \"<node id>\""))
    elif out not in nodes:
        issues.append(Issue("no_output", f"output '{out}' is not a node", hint=near(out, nodes)))
    elif nodes[out].get("op") in SIGNAL:
        issues.append(Issue("no_output", f"output '{out}' is a signal node; pick an image node"))
    order, cyc = topo(deps)
    if cyc:
        issues.append(Issue("cycle", "graph has a loop: " + " -> ".join(cyc) +
                            ". Break it with a feedback node (params.source = the node to loop from)."))
    # dead nodes: not reachable from output (warn only; they still run in probe, are skipped in render)
    if out in deps and not cyc:
        live, stack = set(), [out]
        for nid, n in nodes.items():
            if n.get("op") == "feedback":
                pass
        while stack:
            x = stack.pop()
            if x in live:
                continue
            live.add(x)
            stack += list(deps.get(x, ()))
            n = nodes.get(x, {})
            if n.get("op") == "feedback" and n.get("params", {}).get("source") in nodes:
                stack.append(n["params"]["source"])
        for nid in nodes:
            if nid not in live:
                issues.append(Issue("unused", f"'{nid}' does not reach the output", nid, level="warn"))
        g["_live"] = live
    return issues, {"params": compiled, "deps": deps, "order": order}


def topo(deps):
    order, state = [], {}
    path = []

    def visit(n):
        s = state.get(n)
        if s == 1:
            i = path.index(n)
            return path[i:] + [n]
        if s == 2:
            return None
        state[n] = 1
        path.append(n)
        for d in sorted(deps.get(n, ())):
            c = visit(d)
            if c:
                return c
        path.pop()
        state[n] = 2
        order.append(n)
        return None
    for n in deps:
        c = visit(n)
        if c:
            return order, c
    return order, None


# ------------------------------------------------------------------------------------------------ patch
def patch(g, edits):
    """Apply a list of small edits. Each edit is one of:
      {"set": "node.param", "value": v}            set a param (value null removes it)
      {"add": "id", "op": "blur", "inputs": [...], "params": {...}}
      {"remove": "id"}                              also unhooks it from every input list
      {"connect": "node.N", "from": "src"}          set input N (null to disconnect)
      {"rename": "old", "to": "new"}                rewires inputs, sources and expressions
      {"output": "id"}  /  {"graph": "res"|"fps"|"duration", "value": v}
    """
    import re
    g = copy.deepcopy(g)
    nodes = g.setdefault("nodes", {})
    log = []
    for e in edits:
        if "set" in e:
            nid, pk = e["set"].split(".", 1)
            n = nodes[nid]
            ps = n.setdefault("params", {})
            if e.get("value") is None:
                ps.pop(pk, None)
            else:
                ps[pk] = e["value"]
        elif "add" in e:
            nodes[e["add"]] = {k: v for k, v in e.items() if k in NODE_KEYS}
        elif "remove" in e:
            nodes.pop(e["remove"], None)
            for n in nodes.values():
                if e["remove"] in n.get("inputs", []):
                    n["inputs"] = [None if x == e["remove"] else x for x in n["inputs"]]
        elif "connect" in e:
            nid, i = e["connect"].split(".")
            ins = nodes[nid].setdefault("inputs", [])
            i = int(i)
            while len(ins) <= i:
                ins.append(None)
            ins[i] = e.get("from")
        elif "rename" in e:
            a, b = e["rename"], e["to"]
            nodes[b] = nodes.pop(a)
            pat = re.compile(r"\b%s\." % re.escape(a))
            for n in nodes.values():
                n["inputs"] = [b if x == a else x for x in n.get("inputs", [])]
                for k, v in list(n.get("params", {}).items()):
                    if v == a and k == "source":
                        n["params"][k] = b
                    elif isinstance(v, str):
                        n["params"][k] = pat.sub(b + ".", v)
                    elif isinstance(v, list):
                        n["params"][k] = [pat.sub(b + ".", x) if isinstance(x, str) else x for x in v]
            if g.get("output") == a:
                g["output"] = b
        elif "output" in e:
            g["output"] = e["output"]
        elif "graph" in e:
            g[e["graph"]] = e["value"]
        else:
            raise ValueError(f"unknown edit {e}")
        log.append(e)
    return g, log

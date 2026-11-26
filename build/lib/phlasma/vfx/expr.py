"""Safe expressions for parameter values.

A numeric parameter may be a number or a string expression, e.g. "0.2 + aud.low * 3" or "sin(t * 2) * 0.5".
Names: t (seconds), frame, fps, dur, aspect, every signal node id (node.channel, e.g. aud.low, clock.v),
and the functions below. No attribute access except node.channel; no builtins; no imports.
"""
import ast, math

FUNCS = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "abs": abs, "min": min, "max": max,
    "pow": pow, "sqrt": lambda x: math.sqrt(max(x, 0.0)), "floor": math.floor, "ceil": math.ceil,
    "exp": math.exp, "log": lambda x: math.log(max(x, 1e-9)), "atan2": math.atan2,
    "fract": lambda x: x - math.floor(x), "clamp": lambda x, a=0.0, b=1.0: min(max(x, a), b),
    "mix": lambda a, b, k: a + (b - a) * k, "step": lambda e, x: 1.0 if x >= e else 0.0,
    "smoothstep": lambda a, b, x: (lambda k: k * k * (3 - 2 * k))(min(max((x - a) / ((b - a) or 1e-9), 0.0), 1.0)),
    "remap": lambda x, a, b, c, d: c + (x - a) * (d - c) / ((b - a) or 1e-9),
    "pulse": lambda x, w=0.1: 1.0 if (x - math.floor(x)) < w else 0.0,
}
CONSTS = {"pi": math.pi, "tau": math.tau, "e": math.e, "true": 1.0, "false": 0.0}
BUILTIN_NAMES = {"t", "frame", "fps", "dur", "aspect"}

_OK = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp, ast.Call, ast.Name,
       ast.Load, ast.Constant, ast.Attribute, ast.operator, ast.unaryop, ast.boolop, ast.cmpop)


class ExprError(ValueError):
    pass


class Expr:
    __slots__ = ("src", "code", "refs", "chans")

    def __init__(self, src):
        self.src = src
        try:
            tree = ast.parse(src.strip(), mode="eval")
        except SyntaxError as e:
            raise ExprError(f"syntax error in expression {src!r}: {e.msg}") from None
        refs, chans = set(), set()
        for n in ast.walk(tree):
            if not isinstance(n, _OK):
                raise ExprError(f"{type(n).__name__} not allowed in expression {src!r}")
            if isinstance(n, ast.Call) and not (isinstance(n.func, ast.Name) and n.func.id in FUNCS):
                raise ExprError(f"unknown function in {src!r}; allowed: {', '.join(sorted(FUNCS))}")
            if isinstance(n, ast.Attribute):
                if not isinstance(n.value, ast.Name):
                    raise ExprError(f"only node.channel references are allowed, in {src!r}")
                refs.add(n.value.id)
                chans.add((n.value.id, n.attr))
            if isinstance(n, ast.Name) and n.id not in FUNCS and n.id not in CONSTS and n.id not in BUILTIN_NAMES:
                refs.add(n.id)
        # names that are used bare (not as node.channel) and aren't builtins also land in refs; graph checks them
        self.refs = refs
        self.chans = chans
        self.code = compile(tree, "<expr>", "eval")

    def __call__(self, env):
        try:
            v = eval(self.code, {"__builtins__": {}}, env)  # noqa: S307 - AST is whitelisted above
        except AttributeError as e:
            raise ExprError(f"{self.src!r}: {e}") from None
        except ZeroDivisionError:
            return 0.0
        except TypeError as e:
            raise ExprError(f"{self.src!r}: {e} (signal values are node.channel, e.g. aud.low)") from None
        return float(v)


class Channels:
    """What a signal node exposes to expressions: aud.low, aud.rms, ..."""
    def __init__(self, node_id, values):
        self._id, self._v = node_id, values

    def __getattr__(self, k):
        try:
            return self._v[k]
        except KeyError:
            raise AttributeError(f"signal node '{self._id}' has no channel '{k}' (has: {', '.join(self._v)})") from None

    def __float__(self):  # bare `lfo1` means its first channel
        return float(next(iter(self._v.values()), 0.0))

    def __repr__(self):
        return f"<{self._id} {self._v}>"


def base_env(t, frame, fps, dur, aspect):
    env = dict(FUNCS)
    env.update(CONSTS)
    env.update(t=t, frame=frame, fps=fps, dur=dur, aspect=aspect)
    return env

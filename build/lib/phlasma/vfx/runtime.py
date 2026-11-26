"""Evaluates a validated graph frame by frame. Deterministic: frame N always looks the same for the same graph.

Signal nodes run on the CPU first (in dependency order), then image nodes on the GPU. Feedback nodes hand out the
previous frame of their source; their copy happens after the frame, which is what makes loops legal.
"""
import math, os, re, time

import numpy as np

from . import media
from .expr import Channels, Expr, ExprError, base_env
from .gpu import GPU, ShaderError
from .graph import validate
from .ops import OPS, SIGNAL


class GraphError(Exception):
    def __init__(self, issues):
        super().__init__("; ".join(f'{i.get("node", "")}: {i["msg"]}' for i in issues if i["level"] == "error"))
        self.issues = issues


class NodeError(Exception):
    def __init__(self, node, msg, param=None, line=None):
        super().__init__(f"{node}: {msg}")
        self.node, self.msg, self.param, self.line = node, msg, param, line

    def issue(self):
        d = {"level": "error", "code": "runtime", "node": self.node, "msg": self.msg}
        if self.param:
            d["param"] = self.param
        if self.line:
            d["line"] = self.line
        return d


def _smooth_noise(x, seed):
    i = math.floor(x)
    f = x - i
    def h(n):
        return math.sin(n * 127.1 + seed * 311.7) * 43758.5453 % 1.0
    a, b = h(i), h(i + 1)
    w = f * f * (3 - 2 * f)
    return (a + (b - a) * w) * 2 - 1


class Engine:
    def __init__(self, g, gpu=None, device=None, only_live=True):
        self.g = g
        issues, comp = validate(g)
        self.issues = issues
        if any(i["level"] == "error" for i in issues):
            raise GraphError(issues)
        self.comp = comp
        self.w, self.h = int(g["res"][0]), int(g["res"][1])
        self.fps = float(g["fps"])
        self.dur = float(g["duration"])
        self.nodes = g["nodes"]
        live = g.get("_live") if only_live else None
        self.order = [n for n in comp["order"] if live is None or n in live]
        self.gpu = gpu or GPU(device)
        self.tex, self.ubufs, self.state, self.cpu_src = {}, {}, {}, {}
        self.sig = {}
        self.timing = {n: 0.0 for n in self.order}
        self._build()

    # ------------------------------------------------------------------------------------------- setup
    def _path(self, p):
        return p if os.path.isabs(p) else os.path.join(self.g["_dir"], p)

    def _build(self):
        G = self.gpu
        self.env = self.env_for(0)
        for nid in self.order:
            n = self.nodes[nid]
            o = OPS[n["op"]]
            if o.family == "signal":
                self.state[nid] = {}
                if o.name == "audio":
                    f = self._p(nid, "file")
                    if not f:
                        raise NodeError(nid, "audio needs params.file", "file")
                    s = media.decode_audio(self._path(f))
                    nf = int(math.ceil(self.dur * self.fps)) + 2
                    pv = lambda k: self._const(nid, k)
                    self.state[nid]["feats"] = media.analyse(
                        s, self.fps, nf, offset=pv("offset")[0], bands=(pv("low"), pv("mid"), pv("high")),
                        normalize=pv("normalize")[0] > 0.5, gain=pv("gain")[0])
                    self.state[nid]["file"] = self._path(f)
                continue
            self.tex[nid] = G.texture(self.w, self.h)
            if o.passes:
                self.tex[nid + "#tmp"] = G.texture(self.w, self.h)
            self.ubufs[nid] = G.ubuf()
            if o.name == "feedback":
                self.tex[nid + "#prev"] = G.texture(self.w, self.h)
            # compile shaders up front, so a broken shader is reported before frame 0
            self._pipes(nid)

    def _pipes(self, nid):
        n, o = self.nodes[nid], OPS[self.nodes[nid]["op"]]
        if o.family != "image" or o.cpu or o.name == "feedback":
            return []
        gp = [p for p in o.params if p.gpu]
        try:
            if o.name == "wgsl":
                code = self._p(nid, "code") or "return input0(uv);"
                nin = int(self._const(nid, "inputs")[0])
                nin = max(nin, len(n.get("inputs", [])))
                return [(self.gpu.pipeline(("wgsl", code, nin), code, gp, nin), nin)]
            bodies = o.passes or [o.wgsl]
            return [(self.gpu.pipeline((o.name, k), b, gp, max(o.inputs, 1 if k else 0) if k else o.inputs),
                     max(o.inputs, 1) if k else o.inputs) for k, b in enumerate(bodies)]
        except ShaderError as e:
            raise NodeError(nid, "shader did not compile:\n" + str(e), "code" if o.name == "wgsl" else None, e.line)

    def _p(self, nid, k):
        n, o = self.nodes[nid], OPS[self.nodes[nid]["op"]]
        if k in self.comp["params"][nid]:
            return self.comp["params"][nid][k]
        from .graph import compile_value
        return compile_value(o.pmap[k], o.pmap[k].default)

    def _const(self, nid, k):
        v = self._p(nid, k)
        return [x if not isinstance(x, Expr) else x(self.env) for x in v] if isinstance(v, list) else v

    # ------------------------------------------------------------------------------------------- frame
    def env_for(self, frame):
        t = frame / self.fps
        e = base_env(t, frame, self.fps, self.dur, self.w / self.h)
        for k, v in self.sig.items():
            e[k] = Channels(k, v)
        return e

    def _val(self, nid, k):
        v = self._p(nid, k)
        if not isinstance(v, list):
            return v
        try:
            return [x(self.env) if isinstance(x, Expr) else x for x in v]
        except ExprError as e:
            raise NodeError(nid, str(e), k) from None

    def step_signals(self, frame):
        dt = 1.0 / self.fps
        self.env = self.env_for(frame)
        t = frame / self.fps
        for nid in self.order:
            o = OPS[self.nodes[nid]["op"]]
            if o.family != "signal":
                continue
            st = self.state[nid]
            V = lambda k: self._val(nid, k)[0]
            if o.name == "lfo":
                ph = t * V("freq") + V("phase")
                f = ph - math.floor(ph)
                w = int(V("wave"))
                x = [math.sin(ph * math.tau), 1 - 4 * abs(f - 0.5), 2 * f - 1, 1.0 if f < 0.5 else -1.0][w]
                out = {"v": x * V("amp") + V("offset")}
            elif o.name == "wobble":
                out = {"v": _smooth_noise(t * V("freq"), V("seed")) * V("amp") + V("offset")}
            elif o.name == "value":
                out = {"v": V("v")}
            elif o.name == "lag":
                x = V("in")
                y = st.get("y", x)
                tau = V("up") if x > y else V("down")
                y += (x - y) * (1 - math.exp(-dt / max(tau, 1e-4)))
                st["y"] = y
                out = {"v": y}
            elif o.name == "speed":
                y = st.get("y", 0.0) + V("rate") * dt if frame > 0 or "y" in st else 0.0
                wr = V("wrap")
                if wr > 0:
                    y = y % wr
                st["y"] = y
                out = {"v": y}
            elif o.name == "envelope":
                x, th = V("in"), V("threshold")
                was = st.get("above", False)
                y, cnt, last = st.get("y", 0.0), st.get("count", 0), st.get("last", -1e9)
                if x >= th and not was and t - last >= V("hold"):
                    st["attack_left"] = V("attack")
                    cnt, last = cnt + 1, t
                if st.get("attack_left", 0) > 0:
                    y = min(1.0, y + dt / max(V("attack"), 1e-4))
                    st["attack_left"] -= dt
                else:
                    y *= math.exp(-dt / max(V("decay"), 1e-4))
                st.update(y=y, count=cnt, last=last, above=x >= th)
                out = {"v": y, "count": float(cnt)}
            elif o.name == "audio":
                F = st["feats"]
                r = F[min(frame, len(F) - 1)]
                out = dict(zip(o.channels, (float(x) for x in r)))
            self.sig[nid] = out
            self.env[nid] = Channels(nid, out)

    def render_frame(self, frame, timing=False):
        """Run every node for this frame. Returns the output texture."""
        self.step_signals(frame)
        G = self.gpu
        t = frame / self.fps
        for nid in self.order:
            n = self.nodes[nid]
            o = OPS[n["op"]]
            if o.family == "signal":
                continue
            t0 = time.perf_counter()
            ins = [self.tex.get(x) if x else None for x in n.get("inputs", [])]
            if n.get("bypass"):
                if ins and ins[0] is not None:
                    G.copy(ins[0], self.tex[nid])
                continue
            if o.name == "feedback":
                reset = frame == 0 or self._val(nid, "reset")[0] > 0.5
                if reset:
                    if ins and ins[0] is not None:
                        G.copy(ins[0], self.tex[nid])
                    else:
                        G.upload(self.tex[nid], np.zeros((self.h, self.w, 4), np.float32))
                else:
                    G.copy(self.tex[nid + "#prev"], self.tex[nid])
            elif o.cpu:
                self._cpu_node(nid, o, frame)
            else:
                gp = [p for p in o.params if p.gpu]
                slots = []
                for p in gp:
                    v = self._val(nid, p.name)
                    if p.type == "int":
                        v = [float(round(v[0]))]
                    slots.append(v)
                G.write_uniforms(self.ubufs[nid], self.w, self.h, t, frame, self.fps, slots)
                pipes = self._pipes(nid)
                if len(pipes) == 1:
                    pipe, nin = pipes[0]
                    G.run(pipe, nin, self.ubufs[nid], ins, self.tex[nid])
                else:
                    (p0, n0), (p1, n1) = pipes
                    G.run(p0, n0, self.ubufs[nid], ins, self.tex[nid + "#tmp"])
                    G.run(p1, n1, self.ubufs[nid], [self.tex[nid + "#tmp"]], self.tex[nid])
            if timing:
                G.sync()
                self.timing[nid] += (time.perf_counter() - t0) * 1000
        # feedback copies after the whole frame
        for nid in self.order:
            n = self.nodes[nid]
            if n["op"] == "feedback":
                src = n["params"]["source"]
                if src in self.tex:
                    G.copy(self.tex[src], self.tex[nid + "#prev"])
        return self.tex[self.g["output"]]

    def _cpu_node(self, nid, o, frame):
        G = self.gpu
        st = self.cpu_src.setdefault(nid, {})
        fit = OPS[o.name].pmap.get("fit")
        if o.name == "image":
            if "done" not in st:
                fitv = fit.choices[int(self._val(nid, "fit")[0])]
                G.upload(self.tex[nid], media.load_image(self._path(self._p(nid, "file")), self.w, self.h, fitv))
                st["done"] = True
        elif o.name == "movie":
            if "r" not in st:
                fitv = fit.choices[int(self._val(nid, "fit")[0])]
                st["r"] = media.MovieReader(self._path(self._p(nid, "file")), self.w, self.h, self.fps, fitv,
                                            self._val(nid, "offset")[0], self._val(nid, "loop")[0] > 0.5)
            G.upload(self.tex[nid], st["r"].frame(frame))
        elif o.name == "text":
            raw = self._p(nid, "text")

            def fld(m):
                ex, _, fmt = m.group(1).partition(":")
                try:
                    return format(Expr(ex)(self.env), fmt)
                except Exception as e:  # noqa: BLE001
                    raise NodeError(nid, f"text field {{{m.group(1)}}}: {e}", "text") from None
            s = re.sub(r"\{([^{}]+)\}", fld, raw)
            if self._val(nid, "upper")[0] > 0.5:
                s = s.upper()
            kind = OPS["text"].pmap["font"].choices[int(self._val(nid, "font")[0])]
            al = OPS["text"].pmap["align"].choices[int(self._val(nid, "align")[0])]
            key = (s, kind, tuple(self._val(nid, "size")), tuple(self._val(nid, "pos")), al, tuple(self._val(nid, "color")))
            if st.get("key") != key:
                G.upload(self.tex[nid], media.render_text(s, kind, key[2][0], key[3], al, key[5], self.w, self.h))
                st["key"] = key

    def read(self, nid=None):
        return self.gpu.read(self.tex[nid or self.g["output"]])

    def close(self):
        for st in self.cpu_src.values():
            if "r" in st:
                st["r"].close()

    def audio_source(self):
        """(file, offset) of the audio node to mux, if any."""
        want = self.g.get("audio")
        for nid in self.order:
            if self.nodes[nid]["op"] == "audio" and (want is None or want == nid):
                return self.state[nid]["file"], self._const(nid, "offset")[0]
        return None, 0.0

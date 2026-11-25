"""Operator catalog. Two families:

  image  (TOP-like): GPU, one RGBA float texture out, 0..N image inputs. WGSL fragment bodies.
  signal (CHOP-like): CPU, a few named float channels per frame, read by expressions as node.channel.

Every op is described as data (params with type, default, range, doc) so `phlasma vfx ops` can hand the whole
catalog to an agent as JSON. Nothing here is copied from any other tool; the names are plain English.

WGSL body conventions (inside the generated fragment function):
  uv      vec2f  0..1, origin top-left (matches image rows)
  p       vec2f  centred, y up, x in [-aspect/2, aspect/2], y in [-0.5, 0.5]
  px      vec2f  pixel coords;  res vec2f width/height;  texel vec2f 1/res
  time    f32 seconds; frame f32; aspect f32
  input0(uv) .. inputN(uv)  sample the image inputs (black if unconnected)
  each param is a local `let` with its own name (enum -> i32, bool -> f32 0/1, color -> vec4f)
  the body must `return` a vec4f.
"""
import math, re

TYPES = ("float", "int", "bool", "vec2", "vec3", "color", "enum", "string", "node")
SLOT_SWZ = {"float": ".x", "int": ".x", "bool": ".x", "enum": ".x", "vec2": ".xy", "vec3": ".xyz", "color": ""}


class P:
    def __init__(self, name, type, default, doc="", min=None, max=None, choices=None):
        assert type in TYPES, type
        self.name, self.type, self.default, self.doc = name, type, default, doc
        self.min, self.max, self.choices = min, max, choices

    @property
    def gpu(self):
        return self.type in SLOT_SWZ

    def spec(self):
        d = {"type": self.type, "default": self.default, "doc": self.doc}
        if self.min is not None or self.max is not None:
            d["range"] = [self.min, self.max]
        if self.choices:
            d["choices"] = self.choices
        if self.type in ("float", "int", "vec2", "vec3", "color"):
            d["animatable"] = True
        return d


class Op:
    def __init__(self, name, family, params, doc, inputs=0, wgsl=None, passes=None, cpu=None, channels=None,
                 tags=()):
        self.name, self.family, self.params, self.doc = name, family, params, doc
        self.inputs, self.wgsl, self.passes, self.cpu = inputs, wgsl, passes, cpu
        self.channels, self.tags = channels or [], tags
        self.pmap = {p.name: p for p in params}

    def spec(self):
        d = {"family": self.family, "doc": self.doc, "params": {p.name: p.spec() for p in self.params}}
        if self.family == "image":
            d["inputs"] = self.inputs
        else:
            d["channels"] = self.channels
        if self.tags:
            d["tags"] = list(self.tags)
        return d


OPS = {}


def op(*a, **k):
    o = Op(*a, **k)
    OPS[o.name] = o
    return o


EXTEND = ["hold", "repeat", "mirror", "zero"]
BLEND = ["over", "add", "multiply", "screen", "difference", "max", "min", "subtract", "mix"]

# ----------------------------------------------------------------------------------------------- WGSL library
LIB = r"""
fn hash3(p: vec3f) -> f32 {
  var q = fract(p * 0.1031);
  q += dot(q, q.zyx + 31.32);
  return fract((q.x + q.y) * q.z);
}
fn vnoise3(p: vec3f) -> f32 {
  let i = floor(p); let f = fract(p);
  let w = f * f * (3.0 - 2.0 * f);
  let a = mix(mix(hash3(i + vec3f(0,0,0)), hash3(i + vec3f(1,0,0)), w.x), mix(hash3(i + vec3f(0,1,0)), hash3(i + vec3f(1,1,0)), w.x), w.y);
  let b = mix(mix(hash3(i + vec3f(0,0,1)), hash3(i + vec3f(1,0,1)), w.x), mix(hash3(i + vec3f(0,1,1)), hash3(i + vec3f(1,1,1)), w.x), w.y);
  return mix(a, b, w.z) * 2.0 - 1.0;
}
fn fbm3(p: vec3f, oct: i32, rough: f32) -> f32 {
  var s = 0.0; var a = 0.5; var q = p; var n = 0.0;
  for (var i = 0; i < 8; i++) {
    if (i >= oct) { break; }
    s += a * vnoise3(q); n += a; a *= rough; q = q * 2.03 + vec3f(17.1, 3.7, 9.2);
  }
  return s / max(n, 1e-5);
}
fn luma(c: vec4f) -> f32 { return dot(c.rgb, vec3f(0.2126, 0.7152, 0.0722)); }
fn rgb2hsv(c: vec3f) -> vec3f {
  let K = vec4f(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
  let p = mix(vec4f(c.bg, K.wz), vec4f(c.gb, K.xy), step(c.b, c.g));
  let q = mix(vec4f(p.xyw, c.r), vec4f(c.r, p.yzx), step(p.x, c.r));
  let d = q.x - min(q.w, q.y);
  return vec3f(abs(q.z + (q.w - q.y) / (6.0 * d + 1e-10)), d / (q.x + 1e-10), q.x);
}
fn hsv2rgb(c: vec3f) -> vec3f {
  let p = abs(fract(c.xxx + vec3f(1.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
  return c.z * mix(vec3f(1.0), clamp(p - 1.0, vec3f(0.0), vec3f(1.0)), c.y);
}
fn extend_uv(uv: vec2f, mode: i32) -> vec2f {
  if (mode == 1) { return fract(uv); }
  if (mode == 2) { return 1.0 - abs(1.0 - 2.0 * fract(uv * 0.5)); }
  return clamp(uv, vec2f(0.0), vec2f(1.0));
}
fn outside(uv: vec2f) -> bool { return any(uv < vec2f(0.0)) || any(uv > vec2f(1.0)); }
fn rot2(a: f32) -> mat2x2f { let c = cos(a); let s = sin(a); return mat2x2f(c, s, -s, c); }
fn blend(a: vec4f, b: vec4f, mode: i32, k: f32) -> vec4f {
  // a = top (input0), b = bottom (input1)
  var r: vec4f;
  switch mode {
    case 0: { r = vec4f(a.rgb * a.a + b.rgb * (1.0 - a.a), a.a + b.a * (1.0 - a.a)); }
    case 1: { r = a + b; }
    case 2: { r = a * b; }
    case 3: { r = 1.0 - (1.0 - a) * (1.0 - b); }
    case 4: { r = vec4f(abs(a.rgb - b.rgb), max(a.a, b.a)); }
    case 5: { r = max(a, b); }
    case 6: { r = min(a, b); }
    case 7: { r = vec4f(b.rgb - a.rgb, b.a); }
    default: { r = mix(b, a, 0.5); }
  }
  return mix(b, r, k);
}
"""

# ----------------------------------------------------------------------------------------------- image ops
op("constant", "image", [P("color", "color", [0, 0, 0, 1], "fill colour, hex or [r,g,b,a]")],
   "Solid colour.", wgsl="return color;")

op("ramp", "image", [
    P("a", "color", [0, 0, 0, 1], "start colour"), P("b", "color", [1, 1, 1, 1], "end colour"),
    P("kind", "enum", "linear", "shape of the ramp", choices=["linear", "radial", "angular"]),
    P("angle", "float", 0.0, "degrees, linear only"), P("center", "vec2", [0, 0], "centre in p space"),
    P("scale", "float", 1.0, "ramp length"), P("repeat", "bool", False, "wrap instead of clamp")],
   "Gradient between two colours.", wgsl="""
  let q = p - center;
  var k: f32;
  if (kind == 0) { k = dot(q, vec2f(cos(radians(angle)), sin(radians(angle)))) / scale + 0.5; }
  else if (kind == 1) { k = length(q) * 2.0 / scale; }
  else { k = (atan2(q.y, q.x) / 6.2831853 + 0.5) / scale; }
  if (repeat > 0.5) { k = fract(k); } else { k = clamp(k, 0.0, 1.0); }
  return mix(a, b, k);""")

op("noise", "image", [
    P("scale", "float", 3.0, "features across the frame height", 0.01, 200),
    P("speed", "float", 0.3, "how fast it evolves, per second"), P("octaves", "int", 4, "detail layers", 1, 8),
    P("rough", "float", 0.5, "gain per octave", 0, 1), P("offset", "vec3", [0, 0, 0], "moves through the field"),
    P("amp", "float", 1.0), P("bias", "float", 0.5, "added after amp; 0.5 centres it in 0..1"),
    P("color", "bool", False, "three decorrelated channels instead of grey"), P("seed", "float", 1.0)],
   "Fractal value noise, animated through time.", wgsl="""
  let q = vec3f(p * scale, time * speed) + offset + vec3f(seed * 13.7, seed * 7.1, seed * 3.3);
  let n = fbm3(q, octaves, rough);
  if (color > 0.5) {
    let g = fbm3(q + vec3f(31.0, 0.0, 5.0), octaves, rough);
    let b = fbm3(q + vec3f(0.0, 47.0, 11.0), octaves, rough);
    return vec4f(vec3f(n, g, b) * amp + bias, 1.0);
  }
  return vec4f(vec3f(n * amp + bias), 1.0);""")

op("shape", "image", [
    P("kind", "enum", "circle", choices=["circle", "ring", "box", "line"]),
    P("center", "vec2", [0, 0], "p space: x in -aspect/2..aspect/2, y -0.5..0.5, y up"),
    P("size", "vec2", [0.3, 0.3], "radius (circle/ring) or half-size (box); line: length, n/a"),
    P("width", "float", 0.02, "ring or line thickness"), P("soft", "float", 0.005, "edge softness"),
    P("rotate", "float", 0.0, "degrees"), P("color", "color", [1, 1, 1, 1]), P("bg", "color", [0, 0, 0, 0])],
   "Circle, ring, box or line with a soft edge (signed distance).", wgsl="""
  let q = rot2(-radians(rotate)) * (p - center);
  var d: f32;
  if (kind == 0) { d = length(q / max(size, vec2f(1e-5))) * min(size.x, size.y) - min(size.x, size.y); }
  else if (kind == 1) { d = abs(length(q) - size.x) - width * 0.5; }
  else if (kind == 2) { let e = abs(q) - size; d = length(max(e, vec2f(0.0))) + min(max(e.x, e.y), 0.0); }
  else { let h = clamp(q.x, -size.x, size.x); d = length(q - vec2f(h, 0.0)) - width * 0.5; }
  let k = 1.0 - smoothstep(-soft, soft, d);
  return mix(bg, color, k);""")

op("transform", "image", [
    P("translate", "vec2", [0, 0], "in p units (frame height = 1)"), P("rotate", "float", 0.0, "degrees"),
    P("scale", "vec2", [1, 1]), P("pivot", "vec2", [0, 0]),
    P("extend", "enum", "zero", "what shows outside the source", choices=EXTEND)],
   "Move, rotate and scale an image.", inputs=1, wgsl="""
  var q = p - pivot - translate;
  q = rot2(-radians(rotate)) * q;
  q = q / select(scale, vec2f(1e-5), abs(scale) < vec2f(1e-5)) + pivot;
  var s = vec2f(q.x / aspect + 0.5, 0.5 - q.y);
  if (extend == 3 && outside(s)) { return vec4f(0.0); }
  return input0(extend_uv(s, extend));""")

op("level", "image", [
    P("brightness", "float", 0.0, "added"), P("contrast", "float", 1.0, "around 0.5"),
    P("gamma", "float", 1.0), P("gain", "float", 1.0, "multiplied first"), P("opacity", "float", 1.0),
    P("invert", "float", 0.0, "0..1 blend to inverted"), P("limit", "bool", False, "clamp result to 0..1")],
   "Brightness, contrast, gamma, invert, opacity.", inputs=1, wgsl="""
  var c = input0(uv);
  var r = c.rgb * gain;
  r = mix(r, 1.0 - r, invert);
  r = (r - 0.5) * contrast + 0.5 + brightness;
  r = sign(r) * pow(abs(r), vec3f(1.0 / max(gamma, 1e-4)));
  var o = vec4f(r, c.a * opacity);
  if (limit > 0.5) { o = clamp(o, vec4f(0.0), vec4f(1.0)); }
  return o;""")

op("hsv", "image", [P("hue", "float", 0.0, "turns, 1 = full circle"), P("sat", "float", 1.0), P("val", "float", 1.0)],
   "Hue shift, saturation and value multiply.", inputs=1, wgsl="""
  let c = input0(uv);
  var h = rgb2hsv(max(c.rgb, vec3f(0.0)));
  h = vec3f(fract(h.x + hue), h.y * sat, h.z * val);
  return vec4f(hsv2rgb(h), c.a);""")

op("blur", "image", [P("size", "float", 8.0, "radius in pixels", 0, 512)],
   "Gaussian blur, two separable passes.", inputs=1, passes=[
       """
  let st = max(size, 0.0) / 12.0; var acc = vec4f(0.0); var w = 0.0;
  for (var i = -12; i <= 12; i++) { let g = exp(-f32(i * i) / 50.0); acc += input0(uv + vec2f(f32(i) * st * texel.x, 0.0)) * g; w += g; }
  return acc / w;""",
       """
  let st = max(size, 0.0) / 12.0; var acc = vec4f(0.0); var w = 0.0;
  for (var i = -12; i <= 12; i++) { let g = exp(-f32(i * i) / 50.0); acc += input0(uv + vec2f(0.0, f32(i) * st * texel.y)) * g; w += g; }
  return acc / w;"""])

op("composite", "image", [P("mode", "enum", "over", "how input0 (top) sits on input1 (bottom)", choices=BLEND),
                          P("mix", "float", 1.0, "0 = bottom only")],
   "Blend two images.", inputs=2, wgsl="return blend(input0(uv), input1(uv), mode, mix);")

op("displace", "image", [P("amount", "float", 0.05, "offset in uv per unit of the map"),
                         P("center", "float", 0.5, "map value that means no offset"),
                         P("extend", "enum", "mirror", choices=EXTEND)],
   "Push input0's pixels by input1's red/green.", inputs=2, wgsl="""
  let d = (input1(uv).rg - center) * amount;
  return input0(extend_uv(uv + vec2f(d.x / aspect, -d.y), extend));""")

op("edge", "image", [P("strength", "float", 2.0), P("mono", "bool", True)],
   "Sobel edges.", inputs=1, wgsl="""
  let tl = luma(input0(uv + texel * vec2f(-1, -1))); let t = luma(input0(uv + texel * vec2f(0, -1)));
  let tr = luma(input0(uv + texel * vec2f(1, -1)));  let l = luma(input0(uv + texel * vec2f(-1, 0)));
  let r = luma(input0(uv + texel * vec2f(1, 0)));    let bl = luma(input0(uv + texel * vec2f(-1, 1)));
  let b = luma(input0(uv + texel * vec2f(0, 1)));    let br = luma(input0(uv + texel * vec2f(1, 1)));
  let gx = (tr + 2.0 * r + br) - (tl + 2.0 * l + bl); let gy = (bl + 2.0 * b + br) - (tl + 2.0 * t + tr);
  let e = length(vec2f(gx, gy)) * strength;
  if (mono > 0.5) { return vec4f(vec3f(e), 1.0); }
  return vec4f(input0(uv).rgb * e, 1.0);""")

op("threshold", "image", [P("level", "float", 0.5), P("soft", "float", 0.02),
                          P("low", "color", [0, 0, 0, 1]), P("high", "color", [1, 1, 1, 1])],
   "Two colours split at a brightness level.", inputs=1, wgsl="""
  let k = smoothstep(level - soft, level + soft, luma(input0(uv)));
  return mix(low, high, k);""")

op("colormap", "image", [P("c0", "color", [0, 0, 0, 1]), P("c1", "color", "#1D847E"), P("c2", "color", [1, 1, 1, 1]),
                         P("c3", "color", [1, 1, 1, 1]), P("stops", "int", 3, "how many of c0..c3 are used", 2, 4),
                         P("offset", "float", 0.0, "shifts the lookup"), P("cycle", "bool", False)],
   "Map brightness to a 2-4 colour gradient.", inputs=1, wgsl="""
  var k = luma(input0(uv)) + offset;
  if (cycle > 0.5) { k = fract(k); } else { k = clamp(k, 0.0, 1.0); }
  let n = f32(max(stops, 2) - 1); let x = k * n; let i = i32(floor(min(x, n - 0.0001)));
  let f = x - f32(i);
  var cs = array<vec4f, 4>(c0, c1, c2, c3);
  return mix(cs[i], cs[min(i + 1, 3)], f);""")

op("kaleido", "image", [P("segments", "int", 6, "", 1, 64), P("rotate", "float", 0.0, "degrees"),
                        P("center", "vec2", [0, 0])],
   "Mirror the image into wedges around a centre.", inputs=1, wgsl="""
  let q = p - center; let a = atan2(q.y, q.x) - radians(rotate); let r = length(q);
  let w = 6.2831853 / f32(max(segments, 1));
  var m = a - w * floor(a / w); m = abs(m - w * 0.5);
  let s = vec2f(cos(m), sin(m)) * r + center;
  return input0(extend_uv(vec2f(s.x / aspect + 0.5, 0.5 - s.y), 2));""")

op("pixelate", "image", [P("cells", "float", 64.0, "cells across the frame height", 1, 4096)],
   "Square mosaic.", inputs=1, wgsl="""
  let c = vec2f(cells * aspect, cells);
  return input0((floor(uv * c) + 0.5) / c);""")

op("chroma", "image", [P("amount", "float", 0.004, "split in uv"), P("angle", "float", 0.0, "degrees")],
   "RGB split (chromatic offset).", inputs=1, wgsl="""
  let d = vec2f(cos(radians(angle)), sin(radians(angle))) * amount;
  return vec4f(input0(uv + d).r, input0(uv).g, input0(uv - d).b, input0(uv).a);""")

op("wgsl", "image", [
    P("code", "string", "return input0(uv);", "WGSL fragment body; see the conventions in ops.py. Params a,b,c,d."),
    P("a", "float", 0.0), P("b", "float", 0.0), P("c", "float", 0.0), P("d", "float", 0.0),
    P("color", "color", [1, 1, 1, 1]), P("inputs", "int", 1, "how many image inputs the code reads", 0, 4)],
   "Your own shader. Compile errors come back with the node id and line.", inputs=4, wgsl=None)

op("feedback", "image", [P("source", "node", "", "node whose PREVIOUS frame this outputs (makes loops legal)"),
                         P("reset", "bool", False, "clear this frame (expression, e.g. 'frame < 1')")],
   "Last frame of another node. The way to build trails, echoes and growth.", inputs=1, tags=("time",))
# feedback's optional input0 is the image it starts from (and resets to); black if unconnected.

op("image", "image", [P("file", "string", "", "png/jpg/exr path, relative to the graph file"),
                      P("fit", "enum", "fill", choices=["fill", "fit", "stretch"])],
   "Still image in.", cpu="image")

op("movie", "image", [P("file", "string", "", "video file, relative to the graph file"),
                      P("fit", "enum", "fill", choices=["fill", "fit", "stretch"]),
                      P("offset", "float", 0.0, "seconds into the clip at t=0"), P("loop", "bool", True)],
   "Video in, frame-locked to the graph clock (offline render: never drops frames).", cpu="movie")

op("text", "image", [P("text", "string", "", "may contain {expr} fields, e.g. 'low {aud.low:.2f}'"),
                     P("font", "enum", "serif", choices=["serif", "serif_light", "mono", "mono_bold"]),
                     P("size", "float", 0.06, "cap height-ish, fraction of frame height"),
                     P("pos", "vec2", [0.05, 0.9], "uv of the anchor (0,0 top-left)"),
                     P("align", "enum", "left", choices=["left", "center", "right"]),
                     P("color", "color", [1, 1, 1, 1]), P("upper", "bool", False)],
   "Text in the brand fonts (Newsreader, IBM Plex Mono).", cpu="text")

# ----------------------------------------------------------------------------------------------- signal ops
op("lfo", "signal", [P("wave", "enum", "sine", choices=["sine", "triangle", "saw", "square"]),
                     P("freq", "float", 0.5, "Hz"), P("amp", "float", 1.0), P("offset", "float", 0.0),
                     P("phase", "float", 0.0, "turns")],
   "Low-frequency oscillator.", channels=["v"])
op("wobble", "signal", [P("freq", "float", 0.5, "Hz"), P("amp", "float", 1.0), P("offset", "float", 0.0),
                        P("seed", "float", 1.0)],
   "Smooth random signal (1D noise).", channels=["v"])
op("value", "signal", [P("v", "float", 0.0, "an expression; use for shared constants or maths")],
   "A named expression, so several params can share one formula.", channels=["v"])
op("lag", "signal", [P("in", "float", 0.0, "expression to smooth"), P("up", "float", 0.05, "seconds rising"),
                     P("down", "float", 0.3, "seconds falling")],
   "Smooths a signal, separately up and down.", channels=["v"])
op("speed", "signal", [P("rate", "float", 1.0, "added per second (expression)"), P("wrap", "float", 0.0, "0 = no wrap")],
   "Integrates a rate into a running value. The one clock that stays in sync.", channels=["v"])
op("envelope", "signal", [P("in", "float", 0.0, "trigger expression"), P("threshold", "float", 0.5),
                          P("attack", "float", 0.01, "seconds"), P("decay", "float", 0.4, "seconds"),
                          P("hold", "float", 0.08, "min seconds between triggers")],
   "Fires on a rising edge of 'in' and decays: beats to punches.", channels=["v", "count"])
op("audio", "signal", [
    P("file", "string", "", "wav/mp3/m4a, relative to the graph file"),
    P("gain", "float", 1.0), P("low", "vec2", [20, 150], "Hz band"), P("mid", "vec2", [150, 2000], "Hz band"),
    P("high", "vec2", [2000, 12000], "Hz band"), P("normalize", "bool", True, "scale each channel so loud = ~1"),
    P("offset", "float", 0.0, "seconds into the file at t=0")],
   "Audio file analysis per frame (trailing window, so it matches what live input would give).",
   channels=["rms", "low", "mid", "high", "onset"])

IMAGE_CPU = {k for k, o in OPS.items() if o.cpu}
SIGNAL = {k for k, o in OPS.items() if o.family == "signal"}


def catalog():
    return {"version": 1, "conventions": __doc__.split("WGSL body conventions")[1].strip(),
            "ops": {k: o.spec() for k, o in sorted(OPS.items())}}


HEX = re.compile(r"^#([0-9a-fA-F]{6})([0-9a-fA-F]{2})?$")


def parse_hex(s):
    m = HEX.match(s)
    if not m:
        return None
    h = m.group(1) + (m.group(2) or "ff")
    return [int(h[i:i + 2], 16) / 255 for i in range(0, 8, 2)]


def frac(x):
    return x - math.floor(x)

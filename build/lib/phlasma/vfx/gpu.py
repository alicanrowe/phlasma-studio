"""WebGPU (wgpu-native) backend. Headless: renders into textures, never opens a window.

Same code on the tower (Vulkan/DX12 on the RTX 4090) and in cloud containers (Mesa lavapipe, CPU Vulkan).
"""
import re, struct

import numpy as np
import wgpu

from .ops import LIB, SLOT_SWZ

FMT = wgpu.TextureFormat.rgba16float
NSLOTS = 24
UBYTES = 16 * (2 + NSLOTS)

VS = """
struct VO { @builtin(position) pos: vec4f, @location(0) uv: vec2f };
@vertex fn vs(@builtin(vertex_index) i: u32) -> VO {
  var q = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var o: VO; o.pos = vec4f(q[i], 0.0, 1.0); o.uv = vec2f(q[i].x * 0.5 + 0.5, 0.5 - q[i].y * 0.5); return o;
}
"""


class ShaderError(Exception):
    def __init__(self, msg, line=None):
        super().__init__(msg)
        self.line = line


class GPU:
    def __init__(self, device_pref=None):
        pref = "low-power" if device_pref == "cpu" else "high-performance"
        self.adapter = wgpu.gpu.request_adapter_sync(power_preference=pref)
        self.device = self.adapter.request_device_sync()
        self.q = self.device.queue
        info = self.adapter.info
        self.info = {k: info.get(k) for k in ("vendor", "device", "backend_type", "adapter_type", "description")}
        self.is_cpu = info.get("adapter_type") == "CPU"
        d = self.device
        self.sampler = d.create_sampler(mag_filter="linear", min_filter="linear",
                                        address_mode_u="clamp-to-edge", address_mode_v="clamp-to-edge")
        self.layouts = {}
        self.pipes = {}
        self.black = self.texture(1, 1)
        self.upload(self.black, np.zeros((1, 1, 4), np.float32))

    # --------------------------------------------------------------------------------------- resources
    def texture(self, w, h):
        U = wgpu.TextureUsage
        return self.device.create_texture(size=(w, h, 1), format=FMT,
                                          usage=U.RENDER_ATTACHMENT | U.TEXTURE_BINDING | U.COPY_SRC | U.COPY_DST)

    def ubuf(self):
        return self.device.create_buffer(size=UBYTES, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)

    def upload(self, tex, arr):
        """arr: HxWx4 float array (0..1)."""
        a = np.ascontiguousarray(arr, dtype=np.float16)
        h, w = a.shape[:2]
        self.q.write_texture({"texture": tex, "mip_level": 0, "origin": (0, 0, 0)}, a.tobytes(),
                             {"offset": 0, "bytes_per_row": w * 8, "rows_per_image": h}, (w, h, 1))

    def read(self, tex):
        w, h = tex.size[0], tex.size[1]
        bpr = ((w * 8 + 255) // 256) * 256
        buf = self.q.read_texture({"texture": tex, "mip_level": 0, "origin": (0, 0, 0)},
                                  {"offset": 0, "bytes_per_row": bpr, "rows_per_image": h}, (w, h, 1))
        a = np.frombuffer(buf, np.float16).reshape(h, bpr // 8, 4)[:, :w]
        return a.astype(np.float32)

    def copy(self, src, dst):
        enc = self.device.create_command_encoder()
        enc.copy_texture_to_texture({"texture": src}, {"texture": dst}, src.size)
        self.q.submit([enc.finish()])

    # --------------------------------------------------------------------------------------- pipelines
    def layout(self, n_in):
        if n_in not in self.layouts:
            S = wgpu.ShaderStage.FRAGMENT
            ent = [{"binding": 0, "visibility": S, "sampler": {"type": "filtering"}},
                   {"binding": 1, "visibility": S, "buffer": {"type": "uniform"}}]
            ent += [{"binding": 2 + i, "visibility": S, "texture": {"sample_type": "float"}} for i in range(n_in)]
            bgl = self.device.create_bind_group_layout(entries=ent)
            self.layouts[n_in] = (bgl, self.device.create_pipeline_layout(bind_group_layouts=[bgl]))
        return self.layouts[n_in]

    @staticmethod
    def source(body, params, n_in):
        """Build the full WGSL module for one pass. Returns (code, first_body_line)."""
        head = [VS, LIB,
                "struct U { res: vec4f, clock: vec4f, p: array<vec4f, %d> };" % NSLOTS,
                "@group(0) @binding(0) var samp: sampler;",
                "@group(0) @binding(1) var<uniform> u: U;"]
        for i in range(n_in):
            head.append(f"@group(0) @binding({2 + i}) var tex{i}: texture_2d<f32>;")
            head.append(f"fn input{i}(c: vec2f) -> vec4f {{ return textureSampleLevel(tex{i}, samp, c, 0.0); }}")
        head += ["@fragment fn fs(v: VO) -> @location(0) vec4f {",
                 "  let uv = v.uv; let res = u.res.xy; let texel = u.res.zw; let px = uv * res;",
                 "  let time = u.clock.x; let frame = u.clock.y; let aspect = u.clock.w;",
                 "  let p = (uv - 0.5) * vec2f(aspect, -1.0);",
                 "  _ = px; _ = time; _ = frame; _ = texel;"]
        for i, prm in enumerate(params):
            sw = SLOT_SWZ[prm.type]
            cast = "i32(u.p[%d].x)" % i if prm.type in ("int", "enum") else f"u.p[{i}]{sw}"
            head.append(f"  let {prm.name} = {cast}; _ = {prm.name};")
        code = "\n".join(head)
        first = code.count("\n") + 2
        return code + "\n" + body + "\n}\n", first

    def pipeline(self, key, body, params, n_in):
        if key in self.pipes:
            return self.pipes[key]
        code, first = self.source(body, params, n_in)
        try:
            mod = self.device.create_shader_module(code=code)
            _, pl = self.layout(n_in)
            pipe = self.device.create_render_pipeline(
                layout=pl, vertex={"module": mod, "entry_point": "vs"},
                fragment={"module": mod, "entry_point": "fs", "targets": [{"format": FMT}]},
                primitive={"topology": "triangle-list"})
        except Exception as e:  # wgpu raises GPUValidationError with the naga report
            msg = str(e)
            m = re.search(r":(\d+):(\d+)", msg)
            line = int(m.group(1)) - first + 1 if m else None
            skip = ("Validation Error", "Caused by", "In wgpuDevice", "In wgpu")
            short = [l for l in msg.splitlines() if l.strip() and not l.strip().startswith(skip)][:12]
            if line and line > 0:
                short = [re.sub(r"wgsl:\d+:", f"code line {line}:", l) for l in short]
            raise ShaderError("\n".join(short), line if line and line > 0 else None) from None
        self.pipes[key] = pipe
        return pipe

    def run(self, pipe, n_in, ubuf, inputs, target):
        bgl, _ = self.layout(n_in)
        ent = [{"binding": 0, "resource": self.sampler},
               {"binding": 1, "resource": {"buffer": ubuf, "offset": 0, "size": UBYTES}}]
        for i in range(n_in):
            t = inputs[i] if i < len(inputs) and inputs[i] is not None else self.black
            ent.append({"binding": 2 + i, "resource": t.create_view()})
        bg = self.device.create_bind_group(layout=bgl, entries=ent)
        enc = self.device.create_command_encoder()
        rp = enc.begin_render_pass(color_attachments=[{"view": target.create_view(), "load_op": "clear",
                                                       "store_op": "store", "clear_value": (0, 0, 0, 0)}])
        rp.set_pipeline(pipe)
        rp.set_bind_group(0, bg)
        rp.draw(3)
        rp.end()
        self.q.submit([enc.finish()])

    def write_uniforms(self, ubuf, w, h, t, frame, fps, slots):
        data = struct.pack("8f", w, h, 1 / w, 1 / h, t, frame, fps, w / h)
        flat = []
        for s in slots[:NSLOTS]:
            s = list(s) + [0.0] * (4 - len(s))
            flat += s[:4]
        flat += [0.0] * (4 * NSLOTS - len(flat))
        self.q.write_buffer(ubuf, 0, data + struct.pack(f"{4 * NSLOTS}f", *flat))

    def sync(self):
        # read 1 texel to force the queue to finish (wgpu-py has no explicit fence API)
        self.q.read_texture({"texture": self.black, "mip_level": 0, "origin": (0, 0, 0)},
                            {"offset": 0, "bytes_per_row": 256, "rows_per_image": 1}, (1, 1, 1))

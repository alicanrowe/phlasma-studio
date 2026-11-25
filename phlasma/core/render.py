"""Shared studio renderer (Blender as a Python module, Cycles). Headless, no GUI, GPU when present.

render_views(items, out_dir, views=("front","back","side","34"), ...) where items are dicts:
  {"name", "V": (n,3) m, "F": (m,3), "mat": "leather"|"mannequin"|"clay"|"metal", "color": (r,g,b) linear,
   "smooth": True, "thick": 0.0013 (solidify, m)}
or trimesh objects (rendered as clay). Z up, the subject faces +Y. Output: PNGs composited on white + run metadata.
"""
import math
import os

import numpy as np

VIEWS = {"front": 0.0, "34": 35.0, "side": 90.0, "back": 180.0}


def _bpy():
    import bpy  # noqa: imported lazily so modules that never render don't pay for it
    return bpy


def _mat(bpy, kind, color):
    m = bpy.data.materials.new(kind)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    if kind == "mannequin":
        bsdf.inputs["Base Color"].default_value = (0.9, 0.9, 0.89, 1)
        bsdf.inputs["Roughness"].default_value = 0.6
        return m
    if kind == "clay":
        bsdf.inputs["Base Color"].default_value = (*color, 1) if color else (0.6, 0.6, 0.6, 1)
        bsdf.inputs["Roughness"].default_value = 0.7
        return m
    if kind == "metal":
        bsdf.inputs["Base Color"].default_value = (*color, 1) if color else (0.9, 0.75, 0.45, 1)
        bsdf.inputs["Metallic"].default_value = 1.0
        bsdf.inputs["Roughness"].default_value = 0.25
        return m
    # leather: cloudy hand-dye colour, grain bump, waxy (not lacquer) sheen
    c = color or (0.16, 0.022, 0.026)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    n1 = nt.nodes.new("ShaderNodeTexNoise")
    n1.inputs["Scale"].default_value = 6.0
    n1.inputs["Detail"].default_value = 3.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (c[0] * 0.65, c[1] * 0.65, c[2] * 0.65, 1)
    ramp.color_ramp.elements[1].color = (min(1, c[0] * 1.35), min(1, c[1] * 1.35), min(1, c[2] * 1.35), 1)
    nt.links.new(tc.outputs["Object"], n1.inputs["Vector"])
    nt.links.new(n1.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    grain = nt.nodes.new("ShaderNodeTexVoronoi")
    grain.inputs["Scale"].default_value = 900.0
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.18
    bump.inputs["Distance"].default_value = 0.0004
    nt.links.new(tc.outputs["Object"], grain.inputs["Vector"])
    nt.links.new(grain.outputs["Distance"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    rough = nt.nodes.new("ShaderNodeMapRange")
    rough.inputs["To Min"].default_value = 0.52
    rough.inputs["To Max"].default_value = 0.78
    nt.links.new(n1.outputs["Fac"], rough.inputs["Value"])
    nt.links.new(rough.outputs["Result"], bsdf.inputs["Roughness"])
    bsdf.inputs["Coat Weight"].default_value = 0.04  # waxy pull-up, not lacquer
    bsdf.inputs["Coat Roughness"].default_value = 0.5
    bsdf.inputs["Specular IOR Level"].default_value = 0.35
    return m


def _add_curve(bpy, it):
    cu = bpy.data.curves.new(it.get("name", "curve"), "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = it.get("radius", 0.001)
    cu.bevel_resolution = 2
    sp = cu.splines.new("POLY")
    P = np.asarray(it["curve"], float)
    sp.points.add(len(P) - 1)
    for p, q in zip(sp.points, P):
        p.co = (q[0], q[1], q[2], 1.0)
    ob = bpy.data.objects.new(it.get("name", "curve"), cu)
    bpy.context.scene.collection.objects.link(ob)
    ob.data.materials.append(_mat(bpy, it.get("mat", "clay"), it.get("color")))
    return ob


def _add_mesh(bpy, it):
    if isinstance(it, dict) and "curve" in it:
        return _add_curve(bpy, it)
    if not isinstance(it, dict):  # trimesh
        it = {"name": "mesh", "V": np.asarray(it.vertices), "F": np.asarray(it.faces), "mat": "clay"}
    me = bpy.data.meshes.new(it.get("name", "mesh"))
    me.from_pydata(np.asarray(it["V"], float).tolist(), [], np.asarray(it["F"], int).tolist())
    me.update()
    ob = bpy.data.objects.new(it.get("name", "mesh"), me)
    bpy.context.scene.collection.objects.link(ob)
    if it.get("smooth", True):
        for p in me.polygons:
            p.use_smooth = True
    if it.get("thick"):
        mod = ob.modifiers.new("solid", "SOLIDIFY")
        mod.thickness = it["thick"]
        mod.offset = 1.0
    if it.get("subsurf"):
        mod = ob.modifiers.new("sub", "SUBSURF")
        mod.levels = mod.render_levels = it["subsurf"]
    ob.data.materials.append(_mat(bpy, it.get("mat", "clay"), it.get("color")))
    return ob


def render_views(items, out_dir, views=("front", "back", "side", "34"), size=(1080, 1350), samples=96,
                 target=None, height=None, device=None, log=print, prefix="view", floor=True):
    bpy = _bpy()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "OPTIX"
        prefs.get_devices()
        gpu = any(d.type in ("OPTIX", "CUDA") for d in prefs.devices)
        for d in prefs.devices:
            d.use = True
        sc.cycles.device = "GPU" if gpu and device != "cpu" else "CPU"
    except Exception:
        sc.cycles.device = "CPU"
    sc.render.resolution_x, sc.render.resolution_y = size
    sc.render.film_transparent = True
    sc.view_settings.view_transform = "AgX"
    try:
        sc.view_settings.look = "AgX - Base Contrast"
    except TypeError:
        pass
    obs = [_add_mesh(bpy, it) for it in items]
    allV = np.vstack([np.asarray(it["V"] if isinstance(it, dict) else it.vertices) for it in items
                      if not (isinstance(it, dict) and (it.get("frame") is False or "curve" in it))])
    lo, hi = allV.min(0), allV.max(0)
    ctr = (lo + hi) / 2 if target is None else np.asarray(target, float)
    h = (hi - lo).max() if height is None else height
    # world: neutral grey studio
    w = bpy.data.worlds.new("studio")
    sc.world = w
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs["Color"].default_value = (0.55, 0.55, 0.55, 1)
    w.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.55
    # key: big soft source front-above; fill from camera side; floor catcher for a contact shadow
    def area(name, loc, rot, size_, power):
        L = bpy.data.lights.new(name, "AREA")
        L.size = size_
        L.energy = power
        o = bpy.data.objects.new(name, L)
        o.location = loc
        o.rotation_euler = rot
        sc.collection.objects.link(o)
        return o
    rig = bpy.data.objects.new("rig", None)
    sc.collection.objects.link(rig)
    rig.location = (ctr[0], ctr[1], 0)
    k = area("key", (ctr[0] + 1.2, ctr[1] + 2.2, ctr[2] + 2.0), (math.radians(-50), 0, math.radians(150)), 2.5, 900)
    f = area("fill", (ctr[0] - 2.2, ctr[1] + 1.8, ctr[2] + 0.3), (math.radians(-80), 0, math.radians(-130)), 3.0, 350)
    b = area("top", (ctr[0], ctr[1] - 0.5, ctr[2] + 2.6), (0, 0, 0), 3.0, 300)
    for o in (k, f, b):
        o.parent = rig
        o.matrix_parent_inverse = rig.matrix_world.inverted()
    if floor:  # contact shadow for things that stand (boots, bags); off for jackets on a form
        bpy.ops.mesh.primitive_plane_add(size=20, location=(ctr[0], ctr[1], lo[2]))
        bpy.context.active_object.is_shadow_catcher = True
    cam_d = bpy.data.cameras.new("cam")
    cam_d.lens = 85
    cam = bpy.data.objects.new("cam", cam_d)
    sc.collection.objects.link(cam)
    sc.camera = cam
    # distance so the subject height fills ~84% of the frame
    sensor_h = 36.0 * size[1] / size[0] if size[1] > size[0] else 36.0
    cam_d.sensor_fit = "VERTICAL" if size[1] > size[0] else "HORIZONTAL"
    cam_d.sensor_height = sensor_h
    fov = 2 * math.atan(sensor_h / 2 / cam_d.lens)
    dist = (h / 0.84) / 2 / math.tan(fov / 2)
    outs = []
    os.makedirs(out_dir, exist_ok=True)
    for v in views:
        ang = math.radians(VIEWS.get(v, float(v) if str(v).replace(".", "").isdigit() else 0))
        # camera orbits the subject; the light rig turns with it so every view is lit the same
        cam.location = (ctr[0] + dist * math.sin(ang), ctr[1] + dist * math.cos(ang), ctr[2])
        cam.rotation_euler = (math.radians(90), 0, math.radians(180) + ang * -1)
        rig.rotation_euler = (0, 0, -ang)
        raw = os.path.join(out_dir, f"_{prefix}_{v}_rgba.png")
        sc.render.filepath = raw
        bpy.ops.render.render(write_still=True)
        final = os.path.join(out_dir, f"{prefix}_{v}.png")
        _on_white(raw, final)
        os.remove(raw)
        outs.append(final)
        log("render", view=v, file=final)
    return outs


def _on_white(src, dst):
    from PIL import Image
    im = Image.open(src).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    Image.alpha_composite(bg, im).convert("RGB").save(dst)

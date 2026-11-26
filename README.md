# PHLASMA Studio

Agent-first garment draping, VFX nodes and CAD in one tool (`phlasma`). Built so Claude can drive it: JSON jobs in,
run folders out, no dialogs, many runs at once. Clean-room, open-source stack. Architecture: [ARCHITECTURE.md](ARCHITECTURE.md).

```
pip install -e .[render]          # garment + renderer
phlasma list                      # modules and verbs
phlasma garment drape jobs/05-gore-jacket.json --out out/gore --json
phlasma batch tower/queue_jackets.txt -j 4
phlasma mcp                       # MCP server: tools garment_drape, cad_build, vfx_render ...
```

## Garment (`phlasma.garment`)
Pattern job (the same JSON the CLO pipeline used: pieces as 2D outlines in mm, seams as line spans) ->
1. **pattern.py** samples every outline so both sides of each seam get matching vertices, then meshes each piece
   (Shewchuk's Triangle, quality mesh, ~12 mm edges).
2. **mannequin.py** builds our own faceless 40R mannequin as a smooth implicit body (no third-party avatar), A-pose,
   with linear-blend arm posing so the photo pose (arms 15 degrees from the body) is reached during the sim.
3. **place.py** wraps body pieces round the torso, sleeves round the bent arm axis, collars round the neck.
4. **sim_xpbd.py** XPBD cloth (Macklin et al. 2016, small-steps variant) in NVIDIA Warp: coloured Gauss-Seidel stretch,
   bend and seam constraints, soft compression (leather buckles but barely stretches), mesh-BVH body collision with
   friction, hash-grid self-collision. Same code on the RTX 4090 (CUDA) and on CPU.
5. **drape.py** runs sew (A-pose, seams shrink to zero) -> pose (arms come down) -> settle, then checks and renders.

Every run writes `run.json` with checks: `seams_closed` (seam gap p95), `no_penetration` (cloth inside the body),
`stretch` (edge strain p95), plus timings per stage.

## Hardware and contents in a job
- `buttons`: parts that sit on the leather (buckles, tags, tabs): a box or disc on the cloth normal, world-up aligned.
- `inserts`: hardware that lives inside the leather (`kind`: magnet | stiffener | weight; `d` and `h` in m; `ring`:
  stitch offset; `pair`: the mating piece). It renders as a soft dome in the hide, conformed to the drape, plus a
  stitched ring round it, never as a visible part on top. Rule: anything sewn between two layers shows only as
  displacement + stitch.
- `held`: a real object in a pocket, sleeve or holster (`obj`: OBJ/GLB, Z up, length on +X; `at`: the pocket mouth on
  the piece; `at_mm`: the point on the object's length that sits at the mouth; `dir`: down | up). The part inside the
  pocket is cut away at the mouth; image textures (wood grain, etched scales, decals) are kept.
- Joints such as dovetails are cut lines between panels (`Garment.cut` with the joint profile), not applied shapes.
  Corners are kept at every stage (cut, sampling, seam smoothing, subdivision creases), so cut joints stay sharp.
- Per-piece `stiff` (e.g. 6 for an interfaced collar) scales bend and compression stiffness; job `stance_deg` spreads
  the mannequin's legs (trousers and culottes).

## Pattern jobs
A job is JSON: `patterns[i].points = [[x_mm, y_mm, flag], ...]` (flag 0 marks a corner; line k runs corner k to
corner k+1) and `seams[j] = {a, la, a0, a1, b, lb, b0, b1, kind}` (pattern a, line la, from fraction a0 to a1, sewn to
pattern b line lb from b0 to b1). Optional `sim` block overrides drape options.

## Licence
Apache-2.0. Uses NVIDIA Warp, Triangle, trimesh, scikit-image and Blender (bpy) under their own licences.

## Status (25 Nov 2026)
Prototype. Drapes 05 GORE and 01 CLINKER from their r2 patterns with arms down. Known gaps: no lap/overlap offsets
yet (lapped seams are joined edge to edge), no zips/topstitch geometry, particle (not triangle) self-collision.

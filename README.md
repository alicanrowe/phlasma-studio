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

## Pattern jobs
A job is JSON: `patterns[i].points = [[x_mm, y_mm, flag], ...]` (flag 0 marks a corner; line k runs corner k to
corner k+1) and `seams[j] = {a, la, a0, a1, b, lb, b0, b1, kind}` (pattern a, line la, from fraction a0 to a1, sewn to
pattern b line lb from b0 to b1). Optional `sim` block overrides drape options.

## Licence
Apache-2.0. Uses NVIDIA Warp, Triangle, trimesh, scikit-image and Blender (bpy) under their own licences.

## Status (25 Nov 2026)
Prototype. Drapes 05 GORE and 01 CLINKER from their r2 patterns with arms down. Known gaps: no lap/overlap offsets
yet (lapped seams are joined edge to edge), no zips/topstitch geometry, particle (not triangle) self-collision.

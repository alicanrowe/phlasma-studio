# PHLASMA Studio: suite architecture (v0, 25 Nov 2026)

One app, one exe (`phlasma` / `phlasma.exe`), three modules that share a core. Built agent-first: everything a person
can do, Claude can do from a script, headless, in parallel, with machine-readable output. Clean-room: public papers and
open-source libraries only. Nothing decompiled, copied or file-format-cloned from CLO, TouchDesigner, Rhino, etc.

| Module | Package | Replaces the need for | Owner thread |
|---|---|---|---|
| Garment (drape) | `phlasma.garment` | CLO3D | "Build our own garment 3D software" |
| VFX nodes (flow) | `phlasma.vfx` | TouchDesigner | VFX node tool thread |
| CAD (form) | `phlasma.cad` | Onshape/Rhino for furniture + jewelry | "Build our own CAD for furniture and jewelry" |

## Rules every module follows (the agent contract)
1. **No modal UI, ever.** Nothing blocks on a dialog. A problem is an error record in the log plus a non-zero exit, never a popup.
   (CLO r3 stalled the whole tower batch on one "cannot remesh" dialog. That is the failure this suite exists to remove.)
2. **Job in, run folder out.** Every operation takes a JSON job file (or stdin) and writes a run folder:
   `run.json` (status, timings, versions, input hash), `log.jsonl` (one JSON event per line), outputs, `preview.png`.
3. **Deterministic + resumable.** Same job + seed = same output. Long steps checkpoint so a killed run resumes.
4. **Many at once.** No licence seat, no single-instance lock. N jobs run as N processes (or N batched sims on one GPU).
5. **Self-checking.** Each module ships checks that run by default and land in `run.json["checks"]`
   (garment: seam gap, penetration, stretch; CAD: manifold, wall thickness, min feature; VFX: frame time, NaN frames).
6. **Same CLI shape.** `phlasma <module> <verb> <job.json> --out <dir> [--device cuda|cpu] [--json]`
   Exit codes: 0 ok, 2 bad job (schema), 3 check failed, 4 runtime error. `--json` prints only the final `run.json`.
   Job paths are relative to the job file (`job['_dir']`); `-` reads stdin; no job = `{}` for introspection verbs.
   A verb fn may carry `SCHEMA` (JSON Schema); core validates it first and returns bad_job.
7. **Same Python API.** `from phlasma.garment import drape; run = drape(job, out=..., device=...)` returns the same dict.
8. **MCP server.** `phlasma mcp` exposes each verb as a tool (input = job schema, output = run.json + file paths), so
   Claude drives it directly instead of through a GUI.
9. **Viewer, not editor-first.** `phlasma view <run>` serves a local web viewer (three.js) to look at any run or live sim.
   The GUI calls the same API; it never holds state the CLI can't reach.

## Shared core (`phlasma.core`)
- `jobs`: JSON-schema validation per verb, input hashing, run-folder creation.
- `log`: JSONL event logger (`{"t":..,"lvl":..,"ev":..,...}`) + progress events.
- `geom`: mesh IO (OBJ, GLB, STL, DXF/SVG for flats), triangulation, transforms. Units: millimetres, Z up, Y forward.
- `render`: one studio renderer for every module (grey studio look, composited on white, brand tags low-left, the
  turntable/standard views from boots-bags PHOTO_STANDARD). Backend: Blender as a pip module (`bpy`, Cycles GPU),
  headless. Later optional own path tracer.
- `brand`: reads /mnt/project-files/brand/ theme so every output is on brand.
- `queue`: local job runner (`phlasma batch queue.txt -j 8`) that fans jobs over GPU/CPU workers.

## Stack
Python 3.11+, NumPy, NVIDIA Warp (GPU kernels, CPU fallback), `triangle` (meshing), trimesh, scikit-image, bpy (render).
Garment can later swap its solver for libuipc (IPC, intersection-free). CAD: OpenCascade via `cadquery`/`build123d`,
`manifold3d` for booleans. VFX: own node graph, wgpu-py (WebGPU over Vulkan/DX12, WGSL; same code headless in cloud via lavapipe and on the 4090), ffmpeg for output.

## Repo layout (one repo: `alicanrowe/phlasma-studio`)
```
phlasma/
  core/ jobs.py log.py geom.py render/ brand.py queue.py cli.py mcp.py
  garment/  pattern.py mannequin.py place.py sim_xpbd.py drape.py checks.py schema/
  vfx/      graph.py nodes/ runtime.py schema/
  cad/      model.py ops.py export.py checks.py schema/
  viewer/   (static web viewer)
tests/  examples/  packaging/ (PyInstaller spec -> phlasma.exe, Windows tower first)
```
Each module registers its verbs in `phlasma/<module>/__init__.py` as `VERBS = {"drape": fn, ...}`; `core/cli.py` and
`core/mcp.py` discover them, so modules never touch each other.

## Single exe
PyInstaller one-folder build (`phlasma.exe` + `_internal/`) for the tower (Windows 11, RTX 4090); Linux build for cloud.
Warp compiles kernels at first run into a cache folder next to the exe, so the bundle works offline.

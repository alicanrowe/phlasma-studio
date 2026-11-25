# phlasma cad (form): agent-first CAD for furniture and jewelry

One module of the PHLASMA Studio exe (see ../../ARCHITECTURE.md). Clean-room on open kernels: OpenCascade
(build123d) for exact solids and STEP, shapely for flat plates, manifold3d for exact mesh booleans, and its own
NumPy SDF engine for sculpted castings. No GUI state, no dialogs: JSON job in, run folder out.

```
phlasma cad build  job.json --out run/     # model + every check + STEP/STL/GLB/DXF/SVG + preview.png
phlasma cad check  job.json --out run/     # checks only, the fast loop while designing
phlasma cad check_mesh job.json --out run/ # check existing STL/GLB files
phlasma cad schema x.json --out run/       # what the module takes (for an agent discovering it)
  --set p.Li=12   override a model parameter     exit 0 ok, 2 bad job, 3 check failed, 4 error
```

## Three ways to make a part (mix them in one model)
| Kind | Made from | Best for | Exports |
|---|---|---|---|
| plate | 2D outline (u = grain) x thickness, placed by a frame | laser/CNC sheet parts, flat-pack joinery | STEP, STL, GLB, nested DXF+SVG |
| brep | JSON feature tree (brep.py) or build123d in Python | furniture parts, machined jewelry, bands, bezels, prongs, stones | STEP, STL, GLB |
| mesh | SDF scene (sdf.py) meshed by marching cubes | sculpted cast jewelry, organic forms | STL, GLB |

A model is either a Python file with `PARAMS` and `build(p) -> Model` (examples/cad/lap_box.py, gloss_ring.py) or
pure JSON parts (examples/cad/solitaire.job.json). Jewelry blocks: `jewelry.band/bezel_cup/prong_head/brilliant`,
ring sizes (US -> mm), calibrated stock stones. Flat-pack joinery: `sketch.slot/mortise/tab/fingers/arch/kerf`.

## Checks (run by default, in run.json["checks"]; any fail = exit 3)
bodies (one closed body per part), interference (exact overlap volume, contact fine, layers skip their host),
walls (cast metal: 5th-percentile thickness >= 0.8 mm), laser (fits the xTool bed grain-true, no bar thinner than
the sheet), stones (stock size, held round the girdle, seated), ring_size (measured hole = asked size +-0.1 mm),
stands (furniture rests with its centre of mass inside its contacts). tests/test_cad.py plants one mistake per
check and must catch all of them.

## Prototypes (25 Nov 2026), runs/cad-prototypes/
- LAP laser joinery box: 12 plates, identical to laser-boxes/scripts/boxes.py to 1e-16 in; all checks pass; one
  36 x 18 sheet with kerf, blue inner / red outer cuts. 3 s.
- HELIUM Gloss cocktail ring: same recipe as helium.py, own SDF engine, metal + 4 enamel layers + 6 mm stone, all
  checks pass. 19 s. The ring-size check caught that the original recipe's cushion seat and shank sit inside the
  finger: the hole came out 16.05 mm, about one US size small. Fixed here (shank radius + finger hole cut last).

## Not in v0 (next)
MCP tool wrapper (core owns mcp.py), sketch constraints, CNC toolpaths and dogbones, 3D mortise-and-tenon
library + the furniture skill's cadcheck wood-movement rules, cut lists / drawings from plates, faceted stone
render, Blender render through core.render, Windows exe build with the rest of the suite.

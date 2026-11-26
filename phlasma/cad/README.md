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

A model is either a Python file with `PARAMS` and `build(p) -> Model` (examples/cad/finger_box.py, gloss_ring.py) or
pure JSON parts (examples/cad/solitaire.job.json). Jewelry blocks: `jewelry.band/bezel_cup/prong_head/brilliant`,
ring sizes (US -> mm), calibrated stock stones. Flat-pack joinery: `sketch.slot/mortise/tab/fingers/arch/kerf`.

## Checks (run by default, in run.json["checks"]; any fail = exit 3)
bodies (one closed body per part), interference (exact overlap volume, contact fine, layers skip their host),
walls (cast metal: 5th-percentile thickness >= 0.8 mm), laser (fits the xTool bed grain-true, no bar thinner than
the sheet), stones (stock size, held round the girdle, seated), ring_size (measured hole = asked size +-0.1 mm),
stands (furniture rests with its centre of mass inside its contacts). tests/test_cad.py plants one mistake per
check and must catch all of them.

## Examples (examples/cad/)
- finger_box.py: a finger-jointed laser box from 1/4 in sheet; 5 plates, one nested sheet with kerf, blue inner /
  red outer cuts. About 3 s.
- gloss_ring.py: a sculpted cocktail ring (balloon shank, four puffs, thorns, crater-set 6 mm stone, four cold
  enamel fields as layer parts, cast hollow). About 20 s. Its ring_size check is what caught a recipe whose cushion
  seat sat inside the finger hole (about one US size small): cut the finger hole last.
- stool.py: a four-leg stool in solid white oak: legs and aprons as sticks, blind mortise and tenon at every
  corner (joinery.py), top resting on the aprons; cut list with rough blanks (tenons included in the length).
- solitaire.job.json: a bezel solitaire in pure JSON, no Python.

## Joinery, CNC and cut lists (v0.2)
- `joinery.stick()` + `joinery.mortise_tenon()`: tenon on the rail, mortise (fit + bottom clearance) in the post,
  recorded and checked: walls >= 1/4 in, webs between mortises >= 1/4 in, blind far face >= 1/8 in, mortise
  >= 1/2 in from a leg end (or say `haunch=True`).
- `"cnc": {"tool_d": 0.125}` on a job: dogbone relief at every inside corner (`sketch.dogbone`), CNC sheets in
  `cut_cnc/`, and a check that no slot, notch or hole is narrower than the bit.
- `build/cutlist.csv|json` for furniture: identical parts grouped, finished L x W x T along the grain, shop
  fractions, rough blanks (+1 in long, +1/4 in wide, 12 in minimum for solid sticks).

## Not in v0 (next)
Sketch constraints, the furniture skill's wood-movement rules (cross-grain glue, panel room), drawings, faceted stone
render, Blender render through core.render, Windows exe build with the rest of the suite.

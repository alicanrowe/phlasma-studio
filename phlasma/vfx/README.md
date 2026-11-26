# phlasma vfx: node graphs an agent can drive

Real-time image and signal node graphs, built so Claude can write, run and read them without a GUI.
Original design on open tech: WebGPU via wgpu-native (`pip install wgpu`), WGSL shaders, NumPy, ffmpeg.
Nothing is taken from TouchDesigner's code, file formats or UI; it only shares the general idea of a node graph.

## Why it suits agents (the bottlenecks it removes)
| Usual pain with GUI node tools | Here |
|---|---|
| Graph lives in a binary project file you edit by clicking | One JSON file. Diff it, patch it, generate it. |
| Errors show as a red node you have to see | `validate` lists every problem at once as JSON: node, param, code, a "did you mean" hint, shader line numbers |
| Can't tell what a node is doing without opening its viewer | `probe` writes one image with every node's output, labelled, plus per-node mean/min/max/NaN/over-1 and timing |
| Rendering needs the app open, real time, a person watching | `render` is headless and frame-exact (offline mode never drops frames), video with audio muxed |
| Popups block a batch | No dialogs. Exit codes: 0 ok, 2 bad job, 3 check failed, 4 runtime error |
| Live tweaking needs a human at the screen | `live` hot-reloads the graph file on save, keeps the last good graph if the new one breaks, and writes `latest.png` + `status.json` (fps, dropped frames, signal values, errors) |
| Licence seats, one instance | N processes at once (`phlasma batch`) |

## The loop Claude runs
1. `phlasma vfx ops --out runs/ops --json` once (no job file needed; `phlasma mcp` also serves every verb as a `vfx_<verb>` tool), to read the catalog (every op, param, type, default, range).
2. Write `graph.json`.
3. `phlasma vfx validate job.json --out runs/v --json` until it passes (shaders are compiled here too).
4. `phlasma vfx probe job.json --out runs/p` with `"t": [1.0, 4.0]`, look at `nodes_*.png`, read the stats.
5. Fix with `patch` (or by editing the JSON), then `render` for the video, contact sheet and checks.

```
phlasma vfx render job.json --out runs/r1 --json [--set frames=60] [--set res=[1080,1920]]
```

## Graph format
```json
{
 "phlasma_vfx": 1, "res": [960, 540], "fps": 30, "duration": 8,
 "nodes": {
  "aud":  {"op": "audio",    "params": {"file": "beat.wav"}},
  "kick": {"op": "envelope", "params": {"in": "aud.low", "threshold": 0.6, "decay": 0.25}},
  "ring": {"op": "shape",    "params": {"kind": "ring", "size": ["0.12 + kick.v * 0.18", 0]}},
  "fb":   {"op": "feedback", "params": {"source": "out"}},
  "warp": {"op": "transform","inputs": ["fb"], "params": {"scale": [1.01, 1.01], "rotate": 0.3}},
  "out":  {"op": "composite","inputs": ["ring", "warp"], "params": {"mode": "max"}}
 },
 "output": "out"
}
```
- `inputs` are image nodes, in order (`composite`: input0 = top, input1 = bottom). `null` leaves a slot empty (black).
- Any number param can be an **expression**: `"0.2 + aud.low * 3"`, `"sin(t * 2)"`. Names: `t`, `frame`, `fps`, `dur`,
  `aspect`, every signal node as `node.channel`, and `sin cos tan abs min max pow sqrt floor ceil exp log atan2 fract
  clamp mix step smoothstep remap pulse pi tau`. Vectors take a mix of numbers and expressions per component.
- Colours: `"#1D847E"`, `[r,g,b]` or `[r,g,b,a]` in 0..1.
- Loops are only legal through `feedback` (`params.source` = node to read last frame from).
- `bypass: true` on a node passes input0 through. `note` is free text. Relative file paths resolve from the graph file.
- Coordinates: `uv` is 0..1 from the top-left; `p` (centre, positions, sizes) is centred with y up and the frame
  height = 1, so x runs -aspect/2..aspect/2.

## Operators (v1)
Image: `constant ramp noise shape transform level hsv blur composite displace edge threshold colormap kaleido
pixelate chroma wgsl feedback image movie text`.
Signal: `audio` (rms, low, mid, high, onset from a file, trailing window), `envelope` (beats to punches), `lag`,
`speed` (integrated clock), `lfo`, `wobble` (smooth random), `value`.
`wgsl` takes your own fragment body (`return input0(uv) * a;`), params `a b c d color`, up to 4 inputs.
The full list with params: `phlasma vfx ops`.

## Verbs
| Verb | Job keys | Writes |
|---|---|---|
| `ops` | none | `ops.json` |
| `validate` | `graph` | issues in run.json |
| `render` | `graph`, `start`, `frames` or `duration`, `res`, `fps`, `video`, `audio`, `contact`, `png_every`, `profile`, `expect_motion` | `out.mp4`, `preview.png`, `contact.png`, `frames/` |
| `probe` | `graph`, `t` (list of seconds) or `frames`, `nodes` | `nodes_<t>.png`, `out_<t>.png`, per-node stats, signal values |
| `patch` | `graph`, `edits`, `write` | `graph.json` (+ writes back if valid) |
| `live` | `graph` (file path), `seconds`, `snapshot_every`, `pace` | `live/latest.png`, `live/status.json`; delete `live/RUN` to stop |

Patch edits: `{"set":"node.param","value":v}`, `{"add":"id","op":..,"inputs":[..],"params":{..}}`, `{"remove":"id"}`,
`{"connect":"node.1","from":"src"}`, `{"rename":"a","to":"b"}` (rewires inputs, sources and expressions),
`{"output":"id"}`, `{"graph":"fps","value":60}`.

Render checks (in run.json): `nan_frames`, `black_frames` (fails only if all black), `motion` (fails if nothing moves),
`frame_time` (p95 vs the fps budget, first frame excluded; informational on a CPU adapter or with `realtime_check: false`). Metrics: frame ms mean/p95/max, real-time
factor, per-node ms.

## Python
```python
from phlasma.vfx import run
r = run("render", {"graph": "examples/vfx/pulse.json", "frames": 60}, out="runs/r1")
r["checks"], r["metrics"]["node_ms_mean"], r["outputs"]["video"]
```

## Where it runs
Any machine with a Vulkan, DX12 or Metal driver. Cloud containers: `apt-get install mesa-vulkan-drivers` (lavapipe,
CPU): about 10 fps at 960x540 for a 10-node graph. Tower (RTX 4090): GPU-native, expected well past real time.
Heavy renders go to a GPU machine, not a laptop.

## Next (not in v1)
On-screen live window on the tower (same engine, a wgpu canvas), live mic/line-in audio, MIDI/OSC in, NDI/Spout out
for projection, particle/point ops (GPU compute), 3D render op (instanced geometry + camera), time-machine/slit-scan
(texture history), node viewer in the shared `phlasma view` web UI, MCP tools via `phlasma mcp`.

# PHLASMA Studio garment drape on the tower (Windows 11, RTX 4090)

No GUI, no sign-in, no licence. Safe to run alongside anything; it never opens a window.

## One-time setup (PowerShell)
1. Python 3.13 x64 from python.org if `py -3.13 --version` fails (tick "Add to PATH").
2. Unzip `phlasma-suite.zip` to `C:\phlasma` (so `C:\phlasma\phlasma\__init__.py` exists).
3. ```
   cd C:\phlasma
   py -3.13 -m venv .venv
   .venv\Scripts\python -m pip install -U pip
   .venv\Scripts\python -m pip install numpy scipy warp-lang triangle trimesh scikit-image shapely pillow jsonschema bpy
   .venv\Scripts\python -c "import warp as wp; wp.init(); print(wp.get_cuda_devices())"
   ```
   The last line must list `cuda:0` (the 4090).

## Run
```
cd C:\phlasma
.venv\Scripts\python -m phlasma garment drape jobs\05-gore-jacket.json --out out\05-gore-jacket
.venv\Scripts\python -m phlasma batch tower\queue_jackets.txt -j 4
```
Each run folder gets run.json (status, checks, timings), log.jsonl, garment.obj, mannequin.obj, view_*.png.

## Send back
Zip `C:\phlasma\out` (pngs + run.json + log.jsonl; skip the .obj/.npz if over 12 MB per part) and the stdout of the batch.

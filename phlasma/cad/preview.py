"""Quick look: three orthographic views (3/4, front, top) on white, with a lowercase mono tag low-left in the studio
green. Own z-buffer rasteriser (NumPy + numba), CPU, about a second. Final renders go through core.render (Blender)."""
import os
import numpy as np

from .part import color_of

# brand fonts folder: $PHLASMA_BRAND, else ./brand next to the package (falls back to the default font)
BRAND = os.environ.get("PHLASMA_BRAND", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "brand"))
ACCENT = "#1D847E"

try:
    from numba import njit
except Exception:  # numba missing: same code, slower
    def njit(*a, **k):
        return a[0] if a and callable(a[0]) else (lambda f: f)


@njit(cache=True)
def _raster(P, F, N, pid, W, H, Z, NB, ID):
    for t in range(F.shape[0]):
        i0, i1, i2 = F[t, 0], F[t, 1], F[t, 2]
        x0, y0, z0 = P[i0, 0], P[i0, 1], P[i0, 2]
        x1, y1, z1 = P[i1, 0], P[i1, 1], P[i1, 2]
        x2, y2, z2 = P[i2, 0], P[i2, 1], P[i2, 2]
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) < 1e-12:
            continue
        xa = int(max(0.0, np.floor(min(x0, min(x1, x2)))))
        xb = int(min(W - 1.0, np.ceil(max(x0, max(x1, x2)))))
        ya = int(max(0.0, np.floor(min(y0, min(y1, y2)))))
        yb = int(min(H - 1.0, np.ceil(max(y0, max(y1, y2)))))
        for yy in range(ya, yb + 1):
            for xx in range(xa, xb + 1):
                px, py = xx + 0.5, yy + 0.5
                a = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / den
                b = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / den
                c = 1.0 - a - b
                if a < 0 or b < 0 or c < 0:
                    continue
                z = a * z0 + b * z1 + c * z2
                if z > Z[yy, xx]:
                    Z[yy, xx] = z
                    ID[yy, xx] = pid[t]
                    for k in range(3):
                        NB[yy, xx, k] = a * N[i0, k] + b * N[i1, k] + c * N[i2, k]
    return 0


def _gather(model):
    Ps, Fs, Ns, ids, cols, roles = [], [], [], [], [], []
    off = 0
    for j, p in enumerate(model.parts):
        m = p.mesh()
        if p.kind == "mesh":            # sculpted: smooth vertex normals
            v, f, n = m.vertices, m.faces, m.vertex_normals
        else:                           # exact parts: flat faces, so unshare vertices per face
            v = m.vertices[m.faces].reshape(-1, 3)
            f = np.arange(len(v)).reshape(-1, 3)
            n = np.repeat(m.face_normals, 3, axis=0)
            # but keep curved B-rep faces smooth: use vertex normals where the mesh is fine
            if p.kind == "brep" and len(m.faces) > 200:
                v, f, n = m.vertices, m.faces, m.vertex_normals
        Ps.append(v); Fs.append(f + off); Ns.append(n); off += len(v)
        ids.append(np.full(len(f), j, np.int32))
        cols.append(color_of(p)); roles.append(p.role)
    return np.vstack(Ps), np.vstack(Fs).astype(np.int64), np.vstack(Ns), np.concatenate(ids), np.array(cols), roles


def view_image(model, elev, azim, px=700, ss=2):
    P, F, N, pid, cols, roles = _gather(model)
    e, a = np.radians(elev), np.radians(azim)
    to_cam = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    up0 = np.array([0, 0, 1.0]) if abs(to_cam[2]) < 0.99 else np.array([0, 1.0, 0])
    right = np.cross(up0, to_cam); right /= np.linalg.norm(right)
    up = np.cross(to_cam, right)
    R = np.stack([right, up, to_cam])        # camera space: x right, y up, z toward camera
    Q, NQ = P @ R.T, N @ R.T
    lo, hi = Q.min(0), Q.max(0)
    span = (hi - lo)[:2].max() * 1.12
    S = px * ss
    c = (lo + hi) / 2
    sc = S / span
    X = np.empty_like(Q)
    X[:, 0] = (Q[:, 0] - c[0]) * sc + S / 2
    X[:, 1] = (c[1] - Q[:, 1]) * sc + S / 2
    X[:, 2] = Q[:, 2]
    Z = np.full((S, S), -1e30)
    NB = np.zeros((S, S, 3))
    ID = np.full((S, S), -1, np.int32)
    _raster(X, F, NQ, pid, S, S, Z, NB, ID)
    mask = ID >= 0
    n = NB / np.maximum(np.linalg.norm(NB, axis=2, keepdims=True), 1e-9)
    key = np.array([-0.45, 0.55, 0.70]); key /= np.linalg.norm(key)
    fill = np.array([0.6, -0.2, 0.5]); fill /= np.linalg.norm(fill)
    lam, fil = np.clip(n @ key, 0, 1), np.clip(n @ fill, 0, 1)
    rim = (1 - np.clip(n[..., 2], 0, 1)) ** 3
    idx = np.clip(ID, 0, None)
    base = cols[idx]
    stone = np.array([r == "stone" for r in roles])[idx]
    spec = np.clip(n @ ((key + np.array([0, 0, 1.0])) / np.linalg.norm(key + np.array([0, 0, 1.0]))), 0, 1) ** 40
    shade = 0.30 + 0.58 * lam + 0.16 * fil + 0.10 * rim
    shade = np.where(stone, 0.6 + 0.4 * lam, shade)
    img = np.clip(base * shade[..., None] + 0.25 * spec[..., None], 0, 1)
    Zm = np.where(mask, Z, (Z[mask].min() if mask.any() else 0) - span)
    dz = np.zeros_like(Z)
    dz[1:-1, 1:-1] = np.maximum(np.abs(Zm[2:, 1:-1] - Zm[:-2, 1:-1]), np.abs(Zm[1:-1, 2:] - Zm[1:-1, :-2]))
    idc = np.zeros_like(mask)
    idc[1:-1, 1:-1] = (ID[2:, 1:-1] != ID[:-2, 1:-1]) | (ID[1:-1, 2:] != ID[1:-1, :-2])
    edge = (dz > span * 0.006) | idc
    img[edge] *= 0.4
    img[~mask & ~edge] = 1.0
    return img.reshape(px, ss, px, ss, 3).mean((1, 3))


def render(model, path, views=((24, -58), (0, -90), (90, -90)), px=700, tag=None):
    from PIL import Image, ImageDraw, ImageFont
    tiles = [view_image(model, el, az, px) for el, az in views]
    img = Image.fromarray((np.hstack(tiles) * 255).astype(np.uint8))
    V = np.vstack([p.mesh().vertices for p in model.parts])
    ext = (V.max(0) - V.min(0)) * model.mm
    txt = (tag or f"{model.name}  ·  {ext[0]:.1f} x {ext[1]:.1f} x {ext[2]:.1f} mm  ·  {len(model.parts)} parts").lower()
    d = ImageDraw.Draw(img)
    f = os.path.join(BRAND, "fonts", "IBMPlexMono-Regular.ttf")
    font = ImageFont.truetype(f, 17) if os.path.exists(f) else ImageFont.load_default()
    x0, y0 = 18, img.height - 46
    bb = d.textbbox((x0 + 10, y0 + 6), txt, font=font)
    d.rectangle([x0, y0, bb[2] + 10, bb[3] + 8], fill=ACCENT)
    d.text((x0 + 10, y0 + 6), txt, fill="white", font=font)
    img.save(path)
    return path

"""Parametric faceless men's mannequin (own design), built as a smooth implicit field and meshed.

Units: metres, Z up, the figure faces +Y, wearer's right is +X. Default = men's 40R, 180 cm, A-pose.
The pose and joint points are returned so garment placement can wrap sleeves round the real arm axis.
"""
import numpy as np

# torso sections: z (cm from floor), half-width a, half-depth b, centre y offset (cm), superellipse exponent
TORSO_40R = [
    (84, 16.5, 11.0, -0.5, 2.2),
    (92, 18.4, 12.3, -1.0, 2.3),   # hip ~100 cm
    (101, 17.2, 11.6, -0.4, 2.3),
    (110, 15.9, 11.2, 0.3, 2.3),   # waist ~86 cm
    (121, 16.6, 12.0, 0.8, 2.35),
    (131, 18.3, 13.3, 1.0, 2.4),   # chest ~101.6 cm
    (139, 18.8, 12.3, 0.4, 2.6),
    (145, 18.6, 10.4, -0.6, 2.7),  # across shoulders
    (149.5, 14.0, 8.0, -1.0, 2.3),
    (153, 6.6, 5.9, -1.3, 2.0),    # neck base ~41 cm
]


def _smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)


def _round_cone(p, a, b, ra, rb):
    """Distance to a capsule from a (radius ra) to b (radius rb). p (...,3)."""
    ba = b - a
    pa = p - a
    L2 = ba @ ba
    t = np.clip((pa @ ba) / L2, 0, 1)
    r = ra + (rb - ra) * t
    return np.linalg.norm(pa - t[..., None] * ba, axis=-1) - r


def _ellipsoid(p, c, r):
    q = (p - c) / r
    k0 = np.linalg.norm(q, axis=-1)
    k1 = np.linalg.norm(q / r, axis=-1)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)


def _torso(p, T):
    z = p[..., 2]
    zs = np.array([t[0] for t in T])
    A, B, Y, N = (np.interp(z, zs, [t[i] for t in T]) for i in (1, 2, 3, 4))
    x, y = np.abs(p[..., 0]), np.abs(p[..., 1] - Y)
    rho = ((x / A) ** N + (y / B) ** N) ** (1 / N)
    d = (rho - 1) * np.minimum(A, B) * 0.92
    d = np.maximum(d, zs[0] - z)          # open bottom closed by the legs union
    d = np.maximum(d, z - zs[-1] - 0.5)
    return d


def joints(arm_angle_deg=50.0, cm=1.0):
    """Key points (cm). arm_angle = angle of the arm from vertical (A-pose)."""
    th = np.radians(arm_angle_deg)
    J = {}
    for s, name in ((1, "R"), (-1, "L")):
        sh = np.array([s * 16.8, -0.8, 143.5])
        d = np.array([s * np.sin(th), 0.08, -np.cos(th)])
        d /= np.linalg.norm(d)
        el = sh + d * 30.5
        d2 = np.array([s * np.sin(th - 0.06), 0.22, -np.cos(th - 0.06)])
        d2 /= np.linalg.norm(d2)
        wr = el + d2 * 25.5
        J[f"shoulder_{name}"], J[f"elbow_{name}"], J[f"wrist_{name}"] = sh, el, wr
        J[f"arm_dir_{name}"] = d
        J[f"fore_dir_{name}"] = d2
        hip = np.array([s * 9.4, -0.5, 90.0])
        kn = np.array([s * 10.6, 0.4, 50.0])
        an = np.array([s * 11.2, -1.2, 8.5])
        J[f"hip_{name}"], J[f"knee_{name}"], J[f"ankle_{name}"] = hip, kn, an
    J["nape"] = np.array([0.0, -6.5, 153.0])
    return J


def field(p, arm_angle_deg=50.0):
    """Signed distance-ish field in cm, p (...,3) in cm."""
    J = joints(arm_angle_deg)
    d = _torso(p, TORSO_40R)
    neck = _round_cone(p, np.array([0, -1.3, 150.0]), np.array([0, -0.6, 165.0]), 6.0, 5.6)
    d = _smin(d, neck, 2.5)
    head = _ellipsoid(p, np.array([0, 0.9, 172.5]), np.array([7.6, 9.6, 11.8]))
    d = _smin(d, head, 2.0)
    for n in ("R", "L"):
        sh, el, wr = J[f"shoulder_{n}"], J[f"elbow_{n}"], J[f"wrist_{n}"]
        delt = _ellipsoid(p, sh + np.array([0.9 * np.sign(sh[0]), 0, 0.3]), np.array([5.7, 5.9, 5.6]))
        d = _smin(d, delt, 3.5)
        up = _round_cone(p, sh, el, 4.9, 4.1)
        d = _smin(d, up, 3.0)
        fo = _round_cone(p, el, wr, 4.3, 2.8)
        d = _smin(d, fo, 1.5)
        u = (wr - el) / np.linalg.norm(wr - el)
        palm = _round_cone(p, wr + u * 1.5, wr + u * 9.5, 2.9, 2.3)
        fing = _round_cone(p, wr + u * 9.5, wr + u * 17.0, 2.2, 1.3)
        d = _smin(d, _smin(palm, fing, 1.0), 1.0)
        hip, kn, an = J[f"hip_{n}"], J[f"knee_{n}"], J[f"ankle_{n}"]
        th = _round_cone(p, hip, kn, 9.0, 5.6)
        d = _smin(d, th, 4.0)
        sh_ = _round_cone(p, kn, an, 5.4, 3.4)
        d = _smin(d, sh_, 1.5)
        foot = _round_cone(p, an + np.array([0, -4, -5.5]), an + np.array([0.6, 15.5, -6.2]), 3.6, 2.6)
        d = _smin(d, foot, 2.0)
    return d


def build_mesh(res_cm=0.7, arm_angle_deg=50.0):
    """Returns (verts [m], faces) via marching cubes."""
    from skimage.measure import marching_cubes
    lo = np.array([-78.0, -26.0, -1.0])
    hi = np.array([78.0, 30.0, 186.0])
    n = np.ceil((hi - lo) / res_cm).astype(int) + 1
    xs, ys, zs = (np.linspace(lo[i], lo[i] + (n[i] - 1) * res_cm, n[i]) for i in range(3))
    vol = np.empty(n, np.float32)
    for k, z in enumerate(zs):  # slice by slice keeps memory small
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        P = np.stack([X, Y, np.full_like(X, z)], -1)
        vol[:, :, k] = field(P, arm_angle_deg)
    v, f, _, _ = marching_cubes(vol, 0.0, spacing=(res_cm,) * 3)
    v = (v + lo) / 100.0
    return v.astype(np.float64), f.astype(np.int32)  # outward winding (volume > 0)


if __name__ == "__main__":
    import sys, trimesh
    v, f = build_mesh()
    m = trimesh.Trimesh(v, f)
    print(len(v), len(f), m.is_watertight, m.bounds)
    m.export(sys.argv[1] if len(sys.argv) > 1 else "mannequin.obj")


def arm_weights(V, arm_angle_deg=50.0):
    """Per-vertex weight (0 torso .. 1 arm) for each side, from the A-pose mesh V [m]."""
    J = joints(arm_angle_deg)
    W = {}
    for n in ("R", "L"):
        S = J[f"shoulder_{n}"] / 100.0
        d = J[f"arm_dir_{n}"]
        t = (V - S) @ d
        radial = np.linalg.norm((V - S) - np.outer(t, d), axis=1)
        side = np.sign(V[:, 0]) == np.sign(S[0])
        w = np.clip((t + 0.02) / 0.09, 0, 1)
        w = w * w * (3 - 2 * w)
        w *= (radial < 0.13) & side
        W[n] = w
    return W


def pose(V, W, from_deg, to_deg, arm_angle_deg=50.0):
    """Swing both arms in their plane about the shoulder joint, linear-blend skinned."""
    J = joints(arm_angle_deg)
    out = V.copy()
    for n in ("R", "L"):
        S = J[f"shoulder_{n}"] / 100.0
        a = np.radians(to_deg - from_deg) * (1 if n == "R" else -1)
        # rotation about Y: positive a moves +X arm toward -Z when angle shrinks, so use -a
        c, s = np.cos(-a), np.sin(-a)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        P = (V - S) @ R.T + S
        w = W[n][:, None]
        out = out * (1 - w) + P * w if n == "R" else out + (P - V) * w
    return out

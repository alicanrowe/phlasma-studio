"""Initial 3D placement of flat pieces around the mannequin (own method).

Body pieces: block x (centre back 0 -> centre front 0.57 m, signed L/R) wraps round an elliptical cylinder, then each point
is pushed out along a ray from the torso axis until it clears the body by `clear`. Sleeves wrap round the arm axis
from the shoulder joint. Collars wrap round the neck.
"""
import numpy as np

from . import mannequin as mq

NAPE_Z = 1.535


def classify(base):
    b = base.lower()
    if b.startswith(("collar", "neck", "stand", "hood")):
        return "neck"
    if b.startswith(("sleeve", "cuff", "elbow", "arm")):
        return "arm"
    return "body"


def _clear_along(P, C, clear, arm_angle, steps=60, rmax=0.45):
    """Move each point P along ray C->P outward until field (cm) > clear (cm)."""
    D = P - C
    r0 = np.linalg.norm(D, axis=1, keepdims=True)
    U = D / np.maximum(r0, 1e-9)
    rs = np.linspace(0.02, rmax, steps)
    Q = C[:, None, :] + U[:, None, :] * rs[None, :, None]
    f = mq.field(Q * 100.0, arm_angle)  # (n, steps)
    inside = f < clear * 100
    last = np.where(inside.any(1), steps - 1 - np.argmax(inside[:, ::-1], axis=1), 0)
    r_need = rs[np.minimum(last + 1, steps - 1)]
    r = np.maximum(r0[:, 0], r_need)
    return C + U * r[:, None]


def place(meshes, arm_angle=50.0, clear=0.025, log=print):
    J = mq.joints(arm_angle)
    out = []
    for M in meshes:
        spec = M["piece"].spec
        uv = M["uv"]
        kind = classify(spec.get("base", spec["name"]))
        side = spec.get("side", "")
        x, y = uv[:, 0], uv[:, 1]
        lay = float(spec.get("layer", 0) or 0)
        clear_p = clear + 0.008 * lay  # under-layers (layer -1) start inside their outer piece
        if kind == "body":
            th = np.pi * x / 0.57
            a, b = 0.215, 0.165
            X = np.c_[-a * np.sin(th), -b * np.cos(th) - 0.005, NAPE_Z + y]
            C = np.c_[np.zeros(len(X)), np.full(len(X), -0.005), np.minimum(X[:, 2], 1.30)]
            X = _clear_along(X, C, clear_p, arm_angle)
        elif kind == "arm":
            s = "L" if side in ("L", "") else "R"
            S = J[f"shoulder_{s}"] / 100.0
            d = J[f"arm_dir_{s}"]
            sg = -1.0 if s == "L" else 1.0  # wearer's left is -X
            up = np.array([sg * np.cos(np.radians(arm_angle)), 0.0, np.sin(np.radians(arm_angle))])
            up -= d * (up @ d)
            up /= np.linalg.norm(up)
            fw = np.cross(up, d) if s == "R" else np.cross(d, up)
            if fw[1] < 0:
                fw = -fw
            xs = x if s == "L" else -x  # mirrored R pieces: undo the mirror so +x is the front for both
            R = 0.088
            phi = xs / R
            t = -y + 0.07  # distance down the arm from the shoulder joint (sleeve starts past the armpit)
            E = J[f"elbow_{s}"] / 100.0
            d2 = J[f"fore_dir_{s}"]
            L1 = np.linalg.norm(E - S)
            X = np.zeros((len(t), 3))
            C = np.zeros((len(t), 3))
            for k in range(len(t)):
                # follow the bent arm: upper arm, then forearm (blend round the elbow)
                b = np.clip((t[k] - (L1 - 0.06)) / 0.12, 0, 1)
                dk = d * (1 - b) + d2 * b
                dk /= np.linalg.norm(dk)
                ck = S + d * t[k] if t[k] <= L1 else E + d2 * (t[k] - L1)
                uk = up - dk * (up @ dk)
                uk /= np.linalg.norm(uk)
                fk = fw - dk * (fw @ dk) - uk * (fw @ uk)
                fk /= np.linalg.norm(fk)
                Rk = R - 0.023 * np.clip((t[k] - 0.30) / 0.35, 0, 1)  # taper toward the wrist: less gap to close
                X[k] = ck + Rk * (np.cos(xs[k] / Rk) * uk + np.sin(xs[k] / Rk) * fk) + uk * 0.012
                C[k] = ck
            X = _clear_along(X, C, clear_p * 0.8, arm_angle, rmax=0.25)
        else:  # neck
            th = np.pi * x / max(np.abs(x).max(), 1e-6)
            R = 0.082
            X = np.c_[-R * np.sin(th), -R * np.cos(th) - 0.006, NAPE_Z + 0.012 + y]
        out.append(X)
        log("place", piece=M["name"], kind=kind, n=len(X))
    return out

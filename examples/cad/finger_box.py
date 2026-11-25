"""A plain finger-jointed box from 1/4 in sheet, to show the plate workflow: four walls with alternating fingers at
the corners, a floor that floats on tabs in closed mortises (room across the grain), every part nested grain-true
on a laser bed with kerf compensation. Units inches, Z up, x along the box."""
import numpy as np
from phlasma.cad import Model, frame
from phlasma.cad.sketch import rect, mortise

PARAMS = {"L": 8.0, "W": 5.0, "H": 3.0, "T": 0.25, "FIT": 0.006, "MOVE": 0.03, "n": 5,
          "species": "walnut", "kerf": 0.006, "bed_w": 36.0, "bed_h": 18.0}

X, Y, Z = np.array([1., 0, 0]), np.array([0, 1., 0]), np.array([0, 0, 1.])


def build(p):
    L, W, H, T, FIT, MOVE, n = (p[k] for k in ("L", "W", "H", "T", "FIT", "MOVE", "n"))
    m = Model("finger_box", units="in", kind="furniture",
              sheet={"w": p["bed_w"], "h": p["bed_h"], "thick": T, "kerf": p["kerf"]})
    hz = H / n
    zf = 0.25                                         # floor height
    tab_w = min(2.0, L / 3)

    def wall(length, own_even, tab_len):
        g = rect(-length / 2, 0, length / 2, H)
        for i in range(n):                            # corner fingers: this wall keeps even or odd ones
            if (i % 2 == 0) != own_even:
                for s in (-1, 1):
                    x0 = s * length / 2
                    g = g - rect(x0 - s * (T + FIT / 2), i * hz - (FIT / 2 if i else 0.01),
                                 x0 + s * 0.01, (i + 1) * hz + (FIT / 2 if i < n - 1 else 0.01))
        if tab_len:
            g = g - mortise(0, zf + T / 2, tab_len, T, FIT)
        return g

    side, end = wall(L, True, tab_w), wall(W, False, 0)
    for s, y0 in (("A", W / 2 - T), ("B", -W / 2)):
        m.plate(f"Side_{s}", side, frame(X, Z, Y, [0, y0, 0]), T, p["species"])
    for s, x0 in (("A", L / 2 - T), ("B", -L / 2)):
        m.plate(f"End_{s}", end, frame(Y, Z, X, [x0, 0, 0]), T, p["species"])
    fl = rect(-L / 2 + T + 0.01, -W / 2 + T + MOVE, L / 2 - T - 0.01, W / 2 - T - MOVE) | \
        rect(-tab_w / 2, -W / 2, tab_w / 2, W / 2)
    m.plate("Floor", fl, frame(X, Y, Z, [0, 0, zf]), T, p["species"], floats=True)
    return m

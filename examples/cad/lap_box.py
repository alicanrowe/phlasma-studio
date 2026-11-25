"""LAP: the cross-lap crate from the laser joinery boxes (laser-boxes/scripts/boxes.py lap()), rebuilt in phlasma.cad.

Every corner is a half-lap running 1/2 in past the box, the floor hangs on tabs through closed mortises in the end
walls, the walls stand on their lapped corners (a shallow arch under each), and the lid is the same crate made
shallow with a centre spine and two floating panels; tongues under the lid rails key into notches in the wall tops.
Units inches, Z up, x along the box. Cut from 1/4 in hardwood on the xTool P3 (36.02 x 18.03 in bed).
"""
import numpy as np
from phlasma.cad import Model, frame
from phlasma.cad.sketch import rect, slot, mortise, arch, union

PARAMS = {"Li": 10.0, "Wi": 4.5, "H": 3.75, "hs": 1.0, "T": 0.25, "FIT": 0.006, "MOVE": 0.03,
          "species": "walnut", "tag": "L", "kerf": 0.006}

X, Y, Z = np.array([1., 0, 0]), np.array([0, 1., 0]), np.array([0, 0, 1.])


def build(p):
    Li, Wi, H, hs, T, FIT, MOVE, tag = (p[k] for k in ("Li", "Wi", "H", "hs", "T", "FIT", "MOVE", "tag"))
    m = Model(f"lap_{tag}", units="in", kind="furniture",
              sheet={"w": 36.02, "h": 18.03, "thick": T, "kerf": p["kerf"]})
    over = 0.5                                   # laps run past the box
    xo, yo = Li / 2 + T + over, Wi / 2 + T + over
    end_tab = min(1.5, Wi * 0.4)
    zf, foot = 0.625, 0.3
    key_w, key_d = (1.0 if Li > 6 else 0.75), 0.125
    w = T + FIT                                  # slot width

    def tabs(split):
        if not split:
            return [(0.0, end_tab)]
        q = (Wi / 2 - T / 2) / 2 + T / 2
        return [(-q, min(1.0, Wi / 4)), (q, min(1.0, Wi / 4))]

    def crate(Hh, z0, nm, keys_down=False, notches_up=False, panel_z=None, feet=False, spine=False):
        side = rect(-xo, 0, xo, Hh)              # long rails: laps open from the bottom
        end = rect(-yo, 0, yo, Hh)               # end rails: laps open from the top
        if feet:
            side = side - arch(-Li / 2 + 0.5, Li / 2 - 0.5, foot)
            end = end - arch(-Wi / 2 + 0.5, Wi / 2 - 0.5, foot)
        for s in (-1, 1):
            side = side - slot(s * (Li / 2 + T / 2), 0, w, Hh / 2)
            end = end - slot(s * (Wi / 2 + T / 2), Hh, w, Hh / 2, from_top=True)
        if spine:
            end = end - slot(0, Hh, w, Hh / 2, from_top=True)
        if keys_down:
            side, end = side | rect(-key_w / 2, -key_d, key_w / 2, 0), end | rect(-key_w / 2, -key_d, key_w / 2, 0)
        if notches_up:
            n = slot(0, Hh, key_w + FIT, key_d + FIT / 2, from_top=True)
            side, end = side - n, end - n
        if panel_z is not None:
            for yc, tw in tabs(spine):
                end = end - mortise(yc, panel_z + T / 2, tw, T, FIT)
        for s, y0 in (("A", Wi / 2), ("B", -Wi / 2 - T)):
            m.plate(f"{tag}_{nm}Side_{s}", side, frame(X, Z, Y, [0, y0, z0]), T, p["species"])
        for s, x0 in (("A", Li / 2), ("B", -Li / 2 - T)):
            m.plate(f"{tag}_{nm}End_{s}", end, frame(Y, Z, X, [x0, 0, z0]), T, p["species"])

    crate(H, 0.0, "", notches_up=True, panel_z=zf, feet=True)
    floor = rect(-Li / 2 + 0.01, -Wi / 2 + MOVE, Li / 2 - 0.01, Wi / 2 - MOVE) | \
        rect(-(Li / 2 + T), -end_tab / 2, Li / 2 + T, end_tab / 2)
    m.plate(f"{tag}_Floor", floor, frame(X, Y, Z, [0, 0, zf]), T, p["species"], role="part", floats=True)

    pz = hs - T - 0.25
    crate(hs, H, "Lid", keys_down=True, panel_z=pz, spine=True)
    sp = rect(-xo, 0, xo, hs)
    for s in (-1, 1):
        sp = sp - slot(s * (Li / 2 + T / 2), 0, w, hs / 2)
    m.plate(f"{tag}_LidSpine", sp, frame(X, Z, Y, [0, -T / 2, H]), T, p["species"])
    for sy, s in ((1, "A"), (-1, "B")):
        y_in, y_out = T / 2 + MOVE, Wi / 2 - MOVE
        yc, tw = tabs(True)[0 if sy < 0 else 1]
        pn = rect(-Li / 2 + 0.01, sy * y_in, Li / 2 - 0.01, sy * y_out) | \
            rect(-(Li / 2 + T), yc - tw / 2, Li / 2 + T, yc + tw / 2)
        m.plate(f"{tag}_Lid_Panel_{s}", pn, frame(X, Y, Z, [0, 0, H + pz]), T, p["species"], floats=True)
    return m

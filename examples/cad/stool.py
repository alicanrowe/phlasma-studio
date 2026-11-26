"""A plain four-leg stool to show solid-wood joinery: square legs, aprons joined by blind mortise and tenon (shoulders
top and bottom, tenons centred in the aprons, kept short so the two mortises in each leg leave a web), and a top
that sits on the aprons (fix it with figure-8s or buttons so it can move). Units inches, Z up."""
from phlasma.cad import Model
from phlasma.cad.joinery import stick, mortise_tenon

PARAMS = {"W": 14.0, "H": 18.0, "leg": 1.75, "apron_t": 0.75, "apron_h": 3.0, "setback": 0.125, "top_t": 1.0,
          "overhang": 0.75, "tenon_t": 0.375, "tenon_len": 0.75, "shoulder_top": 0.625, "shoulder_bot": 0.5, "species": "white_oak"}


def build(p):
    m = Model("stool", units="in", kind="furniture")
    W, L, t, ah, sb = p["W"], p["leg"], p["apron_t"], p["apron_h"], p["setback"]
    zt = p["H"] - p["top_t"]                     # leg tops and apron tops meet the underside of the top
    inner = W - 2 * p["overhang"]                # leg-to-leg outside span
    x0, x1 = -inner / 2, inner / 2
    corners = {"FL": (x0, x0), "FR": (x1 - L, x0), "BL": (x0, x1 - L), "BR": (x1 - L, x1 - L)}
    for k, (x, y) in corners.items():
        stick(m, f"Leg_{k}", (x, y, 0), (x + L, y + L, zt), p["species"], grain="z")
    za = (zt - ah, zt)
    aprons = {
        "F": ((x0 + L, x0 + sb, za[0]), (x1 - L, x0 + sb + t, za[1]), "x", ("FL", "FR")),
        "B": ((x0 + L, x1 - sb - t, za[0]), (x1 - L, x1 - sb, za[1]), "x", ("BL", "BR")),
        "L": ((x0 + sb, x0 + L, za[0]), (x0 + sb + t, x1 - L, za[1]), "y", ("FL", "BL")),
        "R": ((x1 - sb - t, x0 + L, za[0]), (x1 - sb, x1 - L, za[1]), "y", ("FR", "BR")),
    }
    for k, (lo, hi, g, (a, b)) in aprons.items():
        stick(m, f"Apron_{k}", lo, hi, p["species"], grain=g)
        mortise_tenon(m, f"Apron_{k}", f"Leg_{a}", f"-{g}", p["tenon_len"], p["tenon_t"], shoulders=(p["shoulder_bot"], p["shoulder_top"]))
        mortise_tenon(m, f"Apron_{k}", f"Leg_{b}", f"+{g}", p["tenon_len"], p["tenon_t"], shoulders=(p["shoulder_bot"], p["shoulder_top"]))
    stick(m, "Top", (-W / 2, -W / 2, zt), (W / 2, W / 2, p["H"]), p["species"], grain="x", floats=True)
    return m

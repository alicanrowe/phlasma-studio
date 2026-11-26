"""Grain-true shelf nesting for laser/CNC sheets. Parts keep u (their grain) along the sheet's long side; the only
turn allowed is 180 deg, which keeps the grain. Returns [[(name, placed_polygon), ...] per sheet], report."""
from shapely import affinity


def nest(plates, W, H, gap=0.125, margin=0.125):
    items = []
    for p in plates:
        x0, y0, x1, y1 = p.poly.bounds
        items.append((p.name, p.poly, x1 - x0, y1 - y0))
    too_big = [n for n, _, w, h in items if w > W - 2 * margin or h > H - 2 * margin]
    items = [it for it in items if it[0] not in too_big]
    items.sort(key=lambda it: (-it[3], -it[2]))
    sheets, cur = [], []
    x, y, shelf_h = margin, margin, 0.0
    for name, g, w, h in items:
        if x + w > W - margin:                 # next shelf
            x, y, shelf_h = margin, y + shelf_h + gap, 0.0
        if y + h > H - margin:                 # next sheet
            sheets.append(cur)
            cur, x, y, shelf_h = [], margin, margin, 0.0
        gx0, gy0 = g.bounds[0], g.bounds[1]
        cur.append((name, affinity.translate(g, x - gx0, y - gy0)))
        x += w + gap
        shelf_h = max(shelf_h, h)
    if cur:
        sheets.append(cur)
    used = [sum(g.area for _, g in s) / (W * H) for s in sheets]
    return sheets, {"sheets": len(sheets), "utilisation": [round(u, 3) for u in used], "too_big": too_big,
                    "sheet_size": [W, H]}

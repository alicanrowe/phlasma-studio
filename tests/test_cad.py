"""phlasma.cad tests: the prototypes pass, and planted mistakes are caught. Run: python3 tests/test_cad.py"""
import json, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phlasma.core.jobs import run_verb
from phlasma.cad import VERBS

EX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "cad")


def run(job, verb="check", **o):
    d = tempfile.mkdtemp()
    return run_verb("cad", verb, VERBS[verb], job, d, echo=False, **o)


def load(name):
    return json.load(open(os.path.join(EX, name)))


def t_solitaire_ok():
    r = run(load("solitaire.job.json"), "build")
    assert r["status"] == "ok", r.get("checks")


def t_lap_ok_and_one_sheet():
    r = run({"model": "examples/cad/lap_box.py"}, "build")
    assert r["status"] == "ok" and r["metrics"]["nest"]["sheets"] == 1, r


def t_lap_parity():
    p = "/mnt/project-files/laser-boxes/scripts"
    if not os.path.exists(p):
        return "skip"
    sys.path.insert(0, p)
    import boxes as B
    from phlasma.cad.model import load as L
    orig = {x.name: x for x in B.lap(10, 4.5, 3.75, "L")}
    m = L({"model": "examples/cad/lap_box.py"}, {})
    assert max(orig[q.name].poly.symmetric_difference(q.poly).area for q in m.parts) < 1e-9


def t_planted_stone_clips():
    j = load("solitaire.job.json"); j["parts"][1]["macro"]["at"] = [0, 0, 12.6]   # stone sunk into the cup floor
    r = run(j)
    assert not r["checks"]["interference"]["pass"]


def t_planted_wrong_size():
    j = load("solitaire.job.json"); j["ring_size_us"] = 9
    assert not run(j)["checks"]["ring_size"]["pass"]


def t_planted_thin_bar():
    j = {"name": "thin", "units": "in", "kind": "furniture", "sheet": {"w": 36, "h": 18},
         "checks": {"stands": False},
         "parts": [{"name": "bar", "thick": 0.25, "outline": [[0, 0], [4, 0], [4, 1], [2.1, 1], [2.1, 3], [2, 3], [2, 1], [0, 1]]}]}
    r = run(j)
    assert not r["checks"]["laser"]["pass"], r["checks"]


def t_planted_unknown_param():
    r = run({"model": "examples/cad/lap_box.py", "params": {"Lenght": 9}})
    assert r["status"] == "bad_job"


def t_planted_stone_not_stock():
    j = load("solitaire.job.json"); j["parts"][1]["macro"]["stone"]["D"] = 6.13
    assert not run(j)["checks"]["stones"]["pass"]


if __name__ == "__main__":
    bad = 0
    for k, f in list(globals().items()):
        if k.startswith("t_"):
            try:
                print(k, f() or "ok")
            except AssertionError as e:
                bad += 1; print(k, "FAIL", str(e)[:300])
    print("failed", bad); sys.exit(1 if bad else 0)

"""Smoke tests for phlasma.vfx. Run: PYTHONPATH=. python3 -m pytest tests/test_vfx.py -q"""
import json, os

import pytest

from phlasma.vfx import run
from phlasma.vfx.graph import patch, validate

EX = os.path.join(os.path.dirname(__file__), "..", "examples", "vfx")


def tiny(nodes, output="o", **kw):
    return {"phlasma_vfx": 1, "res": [64, 36], "fps": 10, "duration": 1, "nodes": nodes, "output": output, **kw}


def test_catalog(tmp_path):
    r = run("ops", {}, str(tmp_path))
    assert r["status"] == "ok" and "noise" in r["ops"]["ops"] and "audio" in r["ops"]["ops"]


def test_validate_reports_everything_with_hints():
    g = tiny({"n": {"op": "noize"}, "m": {"op": "noise", "params": {"scael": 1, "amp": "w.vv"}},
              "w": {"op": "wobble"}, "o": {"op": "blur", "inputs": ["m", "nope"]}})
    g["_dir"] = "."
    issues, _ = validate(g)
    codes = {i["code"] for i in issues}
    assert {"unknown_op", "unknown_param", "unknown_channel", "too_many_inputs"} <= codes
    assert any(i.get("hint") == "did you mean 'noise'?" for i in issues)


def test_cycle_needs_feedback():
    g = tiny({"a": {"op": "level", "inputs": ["o"]}, "o": {"op": "blur", "inputs": ["a"]}})
    g["_dir"] = "."
    assert any(i["code"] == "cycle" for i in validate(g)[0])


def test_shader_error_has_line(tmp_path):
    g = tiny({"o": {"op": "wgsl", "params": {"code": "let k = 1.0\nreturn vec4f(k);"}}})
    r = run("validate", {"graph": g}, str(tmp_path))
    err = [i for i in r["issues"] if i["code"] == "runtime"][0]
    assert err["node"] == "o" and err["line"] == 2


def test_render_feedback_and_signals(tmp_path):
    g = tiny({"l": {"op": "lfo", "params": {"freq": 2}},
              "s": {"op": "shape", "params": {"center": ["l.v * 0.3", 0]}},
              "fb": {"op": "feedback", "params": {"source": "o"}},
              "f": {"op": "level", "inputs": ["fb"], "params": {"gain": 0.8}},
              "o": {"op": "composite", "inputs": ["s", "f"], "params": {"mode": "max"}}})
    r = run("render", {"graph": g}, str(tmp_path))
    assert r["status"] == "ok", r
    assert os.path.exists(r["outputs"]["video"]) and r["checks"]["motion"]["pass"]


def test_deterministic(tmp_path):
    g = tiny({"o": {"op": "noise", "params": {"speed": 1}}})
    a = run("probe", {"graph": g, "frames": [5]}, str(tmp_path / "a"))
    b = run("probe", {"graph": g, "frames": [5]}, str(tmp_path / "b"))
    assert a["moments"][0]["nodes"]["o"]["mean"] == b["moments"][0]["nodes"]["o"]["mean"]


def test_patch_rename_rewires_expressions():
    g = tiny({"w": {"op": "wobble"}, "o": {"op": "noise", "params": {"amp": "w.v * 2"}}})
    g2, _ = patch(g, [{"rename": "w", "to": "wob"}, {"set": "o.scale", "value": 9}])
    assert g2["nodes"]["o"]["params"] == {"amp": "wob.v * 2", "scale": 9}


@pytest.mark.skipif(not os.path.exists(os.path.join(EX, "beat.wav")), reason="example audio missing")
def test_audio_example(tmp_path):
    r = run("render", {"graph": os.path.join(EX, "pulse.json"), "frames": 20, "res": [160, 90]}, str(tmp_path))
    assert r["status"] == "ok" and r["outputs"]["audio_muxed"]

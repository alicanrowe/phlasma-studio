"""phlasma.vfx: agent-first real-time node graphs (image + signal ops, WGSL on wgpu). See vfx/README.md."""
from .verbs import VERBS  # noqa: F401  (ops, validate, render, probe, patch, live)


def run(verb, job, out, device=None, **opts):
    """Python API: run = phlasma.vfx.run("render", {"graph": "g.json"}, out="runs/x") -> the run.json dict."""
    from phlasma.core.jobs import run_verb
    if device:
        opts["device"] = device
    return run_verb("vfx", verb, VERBS[verb], job, out, echo=False, **opts)

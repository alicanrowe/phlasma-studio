"""phlasma.cad: agent-first CAD for furniture and jewelry (exact B-rep on OpenCascade, flat plates for the laser,
SDF sculpting for cast pieces). `phlasma cad schema x.json --out o/` describes everything; see cad/README.md."""
from .verbs import VERBS  # noqa: F401
from .part import Model, Part, frame  # noqa: F401

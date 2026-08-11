"""EL-GS (Evidence-Lineage Gaussian Splatting) implementation package.

Implements the canonical semantics of research-wiki/operations/
elgs-v8-formal-spec.md (revision 4, commit c21de8b) with the LGS
substrate of research-wiki/operations/lgs-method.md.

The import surface is intentionally free of CUDA extensions and side
effects: every mathematical module is CPU-testable. GPU-only render
probes are injected behind the elgs.probe interface at runtime.
"""

ELGS_STATE_SCHEMA = "elgs-state-v1"
K_MAX = 4

__version__ = "0.1.0"

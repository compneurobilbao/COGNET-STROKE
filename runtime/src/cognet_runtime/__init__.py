"""Slim runtime of COGNET-STROKE (numpy, nibabel, nilearn, pandas; matplotlib for the figure).

The analysis environment of the repository (concept-map generation, null-map generation with
brainsmash, neuromaps, notebooks) is separate: see the top-level pyproject.toml. Everything that
is needed to turn a lesion and a disconnection map into a profile lives here.
"""

from .cdr import cdr_flag, compute_cdr, compute_overlap

__all__ = ["cdr_flag", "compute_cdr", "compute_overlap"]
__version__ = "1.0.0"

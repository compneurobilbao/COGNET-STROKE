"""Concept to Disconnection Representation (CDR) and lesion overlap.

The implementation moved to the slim runtime package ``cognet_runtime`` (``runtime/`` in this
repository), which is also what the container step uses. This module keeps the old import path
(``from src.cdr import compute_cdr``) working for the analysis workflows and notebooks.
"""

from cognet_runtime.cdr import cdr_flag, compute_cdr, compute_overlap

__all__ = ["cdr_flag", "compute_cdr", "compute_overlap"]

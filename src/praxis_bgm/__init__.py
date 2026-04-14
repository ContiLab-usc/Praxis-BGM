"""Public package exports for the installable Praxis-BGM layout.

This package exposes the updated damped global-z-prior implementation that was
previously organized in the flat ``Praxis`` package. The canonical import is:

``from praxis_bgm import Praxis_BGM``
"""

from .core import Praxis_BGM
from .prior_utils import build_gaussian_priors_from_source, prepare_source_target_datasets

__all__ = [
    "Praxis_BGM",
    "build_gaussian_priors_from_source",
    "prepare_source_target_datasets",
]

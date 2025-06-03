"""PMMoTo package initialization.

This module sets up the logger and imports all major subpackages,
making core functionality available at the top level.

Exports:
    - logger: The package-level logger.
    - initialize: Main initialization function for PMMoTo domains.
"""

from .core.logging import get_logger
from .core.pmmoto import initialize
from . import core
from . import analysis
from . import domain_generation
from . import filters
from . import io
from . import particles

# Initialize logger at package level
logger = get_logger()

__all__ = [
    "logger",
    "initialize",
    "core",
    "analysis",
    "domain_generation",
    "filters",
    "io",
    "particles",
]

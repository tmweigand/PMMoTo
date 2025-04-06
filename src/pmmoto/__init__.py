from .core.logging import get_logger

# Initialize logger at package level
logger = get_logger()

# Import other modules after logger is initialized
from . import core
from . import analysis
from . import domain_generation
from . import filters
from . import io
from . import particles
from .core.pmmoto import initialize

__all__ = ["logger", "initialize"]

"""Initialize the analysis subpackage for PMMoTo.

Provides statistical, Minkowski, averaging, and binning analysis utilities.
"""

from . import bins

# from . import stats
# from . import minkowski
# from . import average

__all__ = ["stats", "minkowski", "average", "bins"]

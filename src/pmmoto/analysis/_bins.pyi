"""interface to _bins.pyx"""

from typing import Any
import numpy as np
from numpy.typing import NDArray

def _count_locations(
    coordinates: NDArray[np.floating[Any]],
    dimension: int,
    num_bins: int,
    bin_width: float,
    min_bin_value: float,
) -> NDArray[np.integer[Any]]: ...
def _sum_masses(
    coordinates: NDArray[np.floating[Any]],
    masses: NDArray[np.number[Any]],
    dimension: int,
    num_bins: int,
    bin_width: float,
    min_bin_value: float,
) -> NDArray[np.floating[Any]]: ...

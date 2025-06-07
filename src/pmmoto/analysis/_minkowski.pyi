"""interface to _minkowski.pyc"""

from typing import Tuple, Sequence
import numpy as np
from numpy.typing import NDArray

def functionals(
    image: NDArray[np.bool_],
    voxels: Sequence[int],
    res: tuple[float, ...] = ...,
    norm: bool = ...,
    parallel: bool = ...,
) -> Tuple[float, float, float, float]: ...

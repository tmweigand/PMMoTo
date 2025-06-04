"""interface to _minkowski.pyc"""

from typing import Optional, Tuple, Sequence
import numpy as np
from numpy.typing import NDArray

def functionals(
    image: NDArray[np.bool_],
    voxels: Sequence[int],
    res: Optional[NDArray[np.float64]] = ...,
    norm: bool = ...,
    parallel: bool = ...,
) -> Tuple[float, float, float, float]: ...

"""interface to _data_read.pyx"""

import numpy as np
from numpy.typing import DTypeLike, NDArray

def read_lammps_atoms(
    filename: str,
    data_columns: dict[str, DTypeLike],
) -> tuple[dict[str, NDArray[np.generic]], NDArray[np.double], float]: ...

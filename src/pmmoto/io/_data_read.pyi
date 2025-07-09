"""interface to _data_read.pyx"""

import numpy as np
from numpy.typing import NDArray

def read_lammps_atoms(
    filename: str, type_map: None | dict[tuple[int, float], int] = None
) -> tuple[NDArray[np.double], NDArray[np.uint8], NDArray[np.double], float]: ...

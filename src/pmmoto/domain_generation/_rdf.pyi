"""interface to _rdf.pyx"""

from typing import TypeVar
from numpy.typing import NDArray
import numpy as np
from ..particles._particles import PyAtomList

T = TypeVar("T", bound=np.generic)

def _generate_rdf(
    probe_atoms: PyAtomList,
    atoms: PyAtomList,
    max_radius: float,
    bins: NDArray[T],
    bin_width: float,
) -> NDArray[T]: ...

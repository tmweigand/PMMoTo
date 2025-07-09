"""interface to _domain_generation.pyx"""

from typing import TypeVar
import numpy as np
from numpy.typing import NDArray

from ..particles._particles import PySphereList
from ..particles._particles import PyCylinderList
from ..particles._particles import AtomMap
from ..core.subdomain import Subdomain
from ..core.subdomain_padded import PaddedSubdomain
from ..core.subdomain_verlet import VerletSubdomain

T = TypeVar("T", bound=np.generic)

def gen_pm_shape(
    subdomain: Subdomain | PaddedSubdomain | VerletSubdomain,
    spheres: PySphereList | PyCylinderList,
    kd: bool,
) -> NDArray[np.uint8]: ...
def gen_pm_atom(
    subdomain: Subdomain | PaddedSubdomain | VerletSubdomain, atoms: AtomMap, kd: bool
) -> NDArray[np.uint8]: ...
def gen_inkbottle(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
    r_y: float,
    r_z: float,
) -> NDArray[np.uint8]: ...

"""interface to _particles.pyx"""

from typing import Any, TypeVar
import numpy as np
from numpy.typing import NDArray

from ..core.subdomain import Subdomain
from ..core.subdomain_padded import PaddedSubdomain
from ..core.subdomain_verlet import VerletSubdomain

T = TypeVar("T", bound=np.generic)

class AtomMap:
    def return_np_array(self, return_own: bool, return_label: bool) -> NDArray[T]: ...
    def size(
        self, subdomain: None | Subdomain | PaddedSubdomain | VerletSubdomain
    ) -> dict[int, int]: ...
    def get_own_count(
        self,
        subdomain: None | Subdomain | PaddedSubdomain | VerletSubdomain,
        global_size: bool,
    ) -> dict[int, int]: ...
    def add_periodic(
        self, subdomain: Subdomain | PaddedSubdomain | VerletSubdomain
    ) -> None: ...
    def set_own(
        self, subdomain: Subdomain | PaddedSubdomain | VerletSubdomain
    ) -> None: ...
    def trim_intersecting(
        self, subdomain: Subdomain | PaddedSubdomain | VerletSubdomain
    ) -> None: ...
    def trim_within(
        self, subdomain: Subdomain | PaddedSubdomain | VerletSubdomain
    ) -> None: ...
    def return_list(self, label: int) -> list[int]: ...

class PySphereList:
    def set_own(
        self, subdomain: Subdomain | PaddedSubdomain | VerletSubdomain
    ) -> None: ...
    def trim_intersecting(
        self, subdomain: Subdomain | PaddedSubdomain | VerletSubdomain
    ) -> None: ...
    def trim_within(
        self, subdomain: Subdomain | PaddedSubdomain | VerletSubdomain
    ) -> None: ...
    def add_periodic(
        self, subdomain: Subdomain | PaddedSubdomain | VerletSubdomain
    ) -> None: ...

def _initialize_atoms(
    atom_coordinates: NDArray[np.floating[Any]],
    atom_radii: NDArray[np.floating[Any]],
    atom_ids: NDArray[np.integer[Any]],
    atom_masses: None | NDArray[np.integer[Any]] = None,
    by_type: bool = False,
) -> AtomMap: ...
def _initialize_spheres(
    spheres: NDArray[np.floating[Any]],
    radii: NDArray[np.floating[Any]],
    masses: None | NDArray[np.integer[Any]] = None,
) -> PySphereList: ...

"""Initialize the domain_generation subpackage for PMMoTo.

Provides porous media, multiphase, and domain generation utilities.
"""

"""PMMoTo domain generation utilities.

This subpackage provides porous media, multiphase, and domain generation tools.
"""

from . import (
    multiphase,
    porousmedia,
    _domain_generation,
    domain_generation,
    rdf,
    lattice_packings,
)

from .domain_generation import (
    gen_random_binary_grid,
    gen_smoothed_random_binary_grid,
    gen_linear_img,
    gen_pm_spheres_domain,
    gen_pm_atom_domain,
    gen_pm_atom_file,
    gen_pm_inkbottle,
    gen_mp_constant,
    gen_mp_from_grid,
)

__all__ = [
    "multiphase",
    "porousmedia",
    "_domain_generation",
    "domain_generation",
    "rdf",
    "lattice_packings",
    "gen_random_binary_grid",
    "gen_smoothed_random_binary_grid",
    "gen_linear_img",
    "gen_pm_spheres_domain",
    "gen_pm_atom_domain",
    "gen_pm_atom_file",
    "gen_pm_inkbottle",
    "gen_mp_constant",
    "gen_mp_from_grid",
]

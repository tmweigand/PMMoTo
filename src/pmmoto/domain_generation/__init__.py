"""Initialize the domain_generation subpackage for PMMoTo.

Provides porous media, multiphase, and domain generation utilities.
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
    gen_img_random_binary,
    gen_img_smoothed_random_binary,
    gen_img_linear,
    gen_pm_spheres_domain,
    gen_pm_cylinders,
    gen_pm_atom_domain,
    gen_pm_atom_file,
    gen_pm_inkbottle,
    gen_mp_constant,
    deconstruct_img,
)

__all__ = [
    "multiphase",
    "porousmedia",
    "_domain_generation",
    "domain_generation",
    "rdf",
    "lattice_packings",
    "gen_img_random_binary",
    "gen_img_smoothed_random_binary",
    "gen_img_linear",
    "gen_pm_spheres_domain",
    "gen_pm_cylinders",
    "gen_pm_atom_domain",
    "gen_pm_atom_file",
    "gen_pm_inkbottle",
    "gen_mp_constant",
    "deconstruct_img",
]

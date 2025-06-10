"""rdf.py

Radial distribution function (RDF) utilities for PMMoTo.
Provides classes and functions for reading, generating, and binning RDF data.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, TypeVar
import numpy as np
from numpy.typing import NDArray
import warnings

from . import _rdf
from ..core import communication

if TYPE_CHECKING:
    from ..core.subdomain import Subdomain
    from ..core.subdomain_padded import PaddedSubdomain
    from ..core.subdomain_verlet import VerletSubdomain
    from ..particles._particles import AtomMap, PyAtomList
    from ..analysis.bins import Bins

T = TypeVar("T", bound=np.generic)

__all__ = ["bin_distances"]


class RDF:
    """Radial distribution function (RDF) class.

    RDF = g(r) where:
      r is the radial distance and
      g is the free energy.

    This class reads in LAMMPS generated output and generates
    a new interpolated function.
    """

    def __init__(
        self,
        name: str,
        atom_id: int,
        radii: NDArray[np.float64],
        rdf: NDArray[np.float64],
    ):
        """Initialize RDF object.

        Args:
            name (str): Atom or species name.
            atom_id (int): Atom ID.
            radii (np.ndarray): Array of radial distances.
            rdf (np.ndarray): Array of RDF values.

        """
        self.name = name
        self.atom_id = atom_id
        self.radii = radii
        self.rdf = rdf
        # self.rdf = self.rdf_from_counts(counts)

    def interpolate_rdf(self, radius: float) -> float:
        """Interpolate the RDF at a given radius.

        Args:
            radius (float): Radius at which to interpolate.

        Returns:
            float: Interpolated RDF value.

        """
        return float(np.interp(radius, self.radii, self.rdf))

    def potential_mean_force(
        self, k_b: float = 0.0083144621, temp: float = 300
    ) -> NDArray[np.float64]:
        """Compute the potential mean force (pmf).

        Args:
            k_b (float, optional): Boltzmann constant.
            temp (float, optional): Temperature.

        Returns:
            np.ndarray: PMF values.

        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return -k_b * temp * np.log(self.rdf, dtype=np.float64)


class BoundedRDF(RDF):
    """Bounded radial distribution function class.

    The interpolated function is restricted to:
        g(r)> 0 : r : g(r) = 1

    This class finds those bounds and introduces a function r(g).
    """

    def __init__(
        self,
        name: str,
        atom_id: int,
        radii: NDArray[np.float64],
        rdf: NDArray[np.float64],
        eps: float = 0,
    ):
        """Initialize Bounded_RDF object.

        Args:
            name (str): Atom or species name.
            atom_id (int): Atom ID.
            radii (np.ndarray): Array of radial distances.
            rdf (np.ndarray): Array of RDF values.
            eps (float, optional): Epsilon for determining bounds.

        """
        super().__init__(name, atom_id, radii, rdf)
        self.bounds = self.determine_bounds(rdf, eps)
        self.radii, self.rdf = self.get_bounded_RDF_data(radii, rdf, self.bounds)

    @classmethod
    def from_rdf(cls, rdf_instance: RDF, eps: float = 0) -> BoundedRDF:
        """Create a Bounded_RDF instance from an existing RDF instance.

        Args:
            rdf_instance (RDF): An instance of the RDF class.
            eps (float, optional): Epsilon value for determining bounds.

        Returns:
            Bounded_RDF: New bounded RDF instance.

        """
        return cls(
            name=rdf_instance.name,
            atom_id=rdf_instance.atom_id,
            radii=rdf_instance.radii,
            rdf=rdf_instance.rdf,
            eps=eps,
        )

    def determine_bounds(self, rdf: NDArray[np.float64], eps: float = 0) -> list[int]:
        """Get the r values of the bounded RDF such that g(r) > 0 : r : g(r) = 1.

        Args:
            radii (np.ndarray): Array of radial distances.
            rdf (np.ndarray): Array of RDF values.
            eps (float, optional): Epsilon for determining bounds.

        Returns:
            list: Indices for lower and upper bounds.

        """
        bounds = [0, len(rdf)]
        bounds[0] = self.find_min_radius(rdf, eps)
        g_max = np.max(rdf)
        if g_max < 1.0:
            bounds[1] = int(np.argmax(g_max))
        else:
            bounds[1] = self.find_max_radius(1.0)

        return bounds

    def find_min_radius(self, rdf: NDArray[np.float64], eps: float = 1e-3) -> int:
        """Find the smallest r value from the RDF data such that min r where g(r) < eps.

        Args:
            rdf (np.ndarray): Array of RDF values.
            eps (float, optional): Epsilon threshold.

        Returns:
            int: Index of minimum radius.

        """
        indices = np.where(rdf < eps)[0]

        if len(indices) == 0:
            return 0

        return int(indices[0])

    def find_max_radius(self, rdf_value: float) -> int:
        """Find the smallest r from the data so all g(r) values are non-zero after r.

        Args:
            rdf_value: rdf value

        Returns:
            int: Index of maximum radius.

        """
        find_r = rdf_value - self.rdf
        indices = np.where([find_r < 0])[1]
        if len(indices) == 0:
            return 0

        return int(indices[0])

    def get_bounded_RDF_data(
        self, radii: NDArray[np.float64], rdf: NDArray[np.float64], bounds: list[int]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Set the bounds of the radial distribution function.

        Args:
            radii (np.ndarray): Array of radial distances.
            rdf (np.ndarray): Array of RDF values.
            bounds (list): Indices for lower and upper bounds.

        Returns:
            tuple: (r_out, rdf_out) bounded arrays.

        """
        r_out = radii[bounds[0] : bounds[1]]
        rdf_out = rdf[bounds[0] : bounds[1]]

        return r_out, rdf_out

    def interpolate_radius_from_pmf(self, pmf_in: float) -> float:
        """Determine the radius given a potential mean force (pmf) value.

        Args:
            pmf_in (float): PMF value.

        Returns:
            float: Interpolated radius.

        """
        pmf = self.potential_mean_force()

        if pmf_in < min(pmf) or pmf_in > max(pmf):
            print("pmf_in is out of bounds. Interpolation will return boundary values.")

        sorted_pmf, sorted_radii = zip(*sorted(zip(pmf, self.radii)))

        return float(np.interp(pmf_in, sorted_pmf, sorted_radii))


def bin_distances(
    subdomain: Subdomain | PaddedSubdomain | VerletSubdomain,
    probe_atom_list: PyAtomList,
    atoms: AtomMap,
    bins: Bins,
) -> None:
    """Find atoms within a radius of probe atom and bin the distances.

    Args:
        subdomain: Subdomain object containing rank information.
        probe_atom_list: List of probe atoms to calculate RDF from.
        atoms: AtomMap containing target atoms.
        bins: RDFBins object containing binning information.

    Returns:
        None. Updates bins in-place with binned counts from all processes.

    """
    # Generate bins
    binned_distance: dict[int, NDArray[np.float64]] = {}
    for label, atom_list in atoms.atom_map.items():

        # Ensure kd_tree built
        atom_list.build_KDtree()

        binned_distance[label] = np.zeros_like(
            bins.bins[label].values, dtype=np.float64
        )

        binned_distance[label] = _rdf._generate_rdf(
            probe_atom_list,
            atom_list,
            atom_list.radius,
            binned_distance[label],
            bins.bins[label].width,
        )

    all_rdf = communication.all_gather(binned_distance)

    # Sum contributions from all processes
    for n_proc, proc_data in enumerate(all_rdf):
        if subdomain.rank == n_proc:
            continue

        for label in proc_data:
            binned_distance[label] = binned_distance[label] + proc_data[label]

    bins.update_bins(binned_distance)

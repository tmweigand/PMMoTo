"""rdf.py"""

import numpy as np
import warnings

from . import _rdf
from ..particles import _particles
from ..core import communication

__all__ = ["generate_rdf", "bin_distances"]


class RDF:
    """
    Radial distribution function class
    RDF = g(r) where:
      r is the radial distance and
      g is the free energy.

    An alternative is:
     G = -k_b*T ln g(r) / sum_{r=0}^{r_f} g(r) where:
      k_b is the Boltzmann constant
      T is the temperature
      r_f is pairwise distance cutoff

    This is a common output for MD simulations.
    See: https://docs.lammps.org/compute_rdf.html

    This class reads in LAMMPS generated output and generates
    a new interpolated function.
    """

    def __init__(self, name, atom_id, radii, rdf):
        self.name = name
        self.atom_id = atom_id
        self.radii = radii
        self.rdf = rdf
        # self.rdf = self.rdf_from_counts(counts)
        self.bounds = None

    def interpolate_rdf(self, radius):
        """
        Given a radius return the rdf
        """
        return np.interp(radius, self.radii, self.rdf)

    def potential_mean_force(self, k_b=0.0083144621, temp=300):
        """
        Potential mean force (pmf)
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return -k_b * temp * np.log(self.rdf)


class Bounded_RDF(RDF):
    """
    Bounded radial distribution function class
    where the intepolated function is restricted to:
        g(r)> 0 : r : g(r) = 1

    This class finds those bounds and introduces a function
     r(g)
    """

    def __init__(self, name, atom_id, radii, rdf, eps=0):
        super().__init__(name, atom_id, radii, rdf)
        self.bounds = self.determine_bounds(radii, rdf, eps)
        self.radii, self.rdf = self.get_bounded_RDF_data(radii, rdf, self.bounds)

    @classmethod
    def from_rdf(cls, rdf_instance: RDF, eps: float = 0) -> "Bounded_RDF":
        """
        Create a Bounded_RDF instance from an existing RDF instance.

        Args:
            rdf_instance: An instance of the RDF class
            eps: Epsilon value for determining bounds (default: 0)

        Returns:
            A new Bounded_RDF instance with the same data but bounded

        Example:
            >>> rdf = RDF("H2O", 1, radii, rdf_data)
            >>> bounded_rdf = Bounded_RDF.from_rdf(rdf)
        """
        return cls(
            name=rdf_instance.name,
            atom_id=rdf_instance.atom_id,
            radii=rdf_instance.radii,
            rdf=rdf_instance.rdf,
            eps=eps,
        )

    def determine_bounds(self, radii, rdf, eps=0):
        """
        Get the the r values of the bounded RDF such that:
            g(r) > 0 : r : g(r) = 1
        """
        bounds = [0, len(rdf)]
        bounds[0] = self.find_min_radius(rdf, eps)
        g_max = np.max(rdf)
        if g_max < 1.0:
            bounds[1] = np.argmax(g_max)
        else:
            bounds[1] = self.find_max_radius(1.0)

        return bounds

    def find_min_radius(self, rdf, eps=0):
        """
        Find the smallest r values from the RDF data such that:
             min r where g(r) > eps
        """
        r_loc = np.where(rdf < 1e-3)[0][-1]

        return r_loc

    def find_max_radius(self, rdf):
        """
        Find the smallest r value from the RDF data such that:
          all g(r) values are non-zero after r
        """
        find_r = rdf - self.rdf
        r_loc = np.where([find_r < 0])[1][0]

        return r_loc

    def get_bounded_RDF_data(self, radii, rdf, bounds):
        """
        Set the bounds of the radial distribution function
        """
        r_out = radii[bounds[0] : bounds[1]]
        rdf_out = rdf[bounds[0] : bounds[1]]

        return r_out, rdf_out

    def interpolate_radius_from_pmf(self, pmf_in):
        """
        Determine the radius given G.

        """
        pmf = self.potential_mean_force()

        if pmf_in < min(pmf) or pmf_in > max(pmf):
            print("pmf_in is out of bounds. Interpolation will return boundary values.")

        sorted_pmf, sorted_radii = zip(*sorted(zip(pmf, self.radii)))

        return np.interp(pmf_in, sorted_pmf, sorted_radii)


def generate_rdf(bins, binned_distances):
    """
    Generate an rdf from binned distances
    """
    rdf = {}
    for label, binned in binned_distances.items():
        _sum = np.sum(binned)
        if _sum == 0:
            rdf[label] = binned
        else:
            rdf[label] = binned / bins.shell_volumes[label] / np.sum(binned)

    return rdf


def bin_distances(subdomain, probe_atom_list, atoms, bins):
    """
    Finds the atoms that are within a radius of probe atom and bins the distances

    Args:
        subdomain: Subdomain object containing rank information
        probe_atom_list: List of probe atoms to calculate RDF from
        atoms: AtomMap containing target atoms
        bins: RDFBins object containing binning information

    Returns:
        RDFBins with updated bin counts from all processes
    """

    if not isinstance(probe_atom_list, _particles.PyAtomList):
        raise TypeError(
            f"Expected probe_atom_list to be of type PyAtomList, got {type(probe_atom_list)}"
        )

    # Generate bins
    binned_distance = {}
    for label, atom_list in atoms.atom_map.items():

        # Ensure kd_tree built
        atom_list.build_KDtree()

        binned_distance[label] = np.zeros_like(bins.bins[label].values)

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

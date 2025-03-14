"""rdf.py"""

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np

__all__ = ["generate_bins", "generate_rdf", "bin_distances"]

from . import _rdf
from . import particles
from . import _particles
from ..core import communication


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

    def __init__(self, name, atom_id, r, g):
        self.name = name
        self.atom_id = atom_id
        self.r_data = r
        self.g_data = g
        self.bounds = None

    def g(self, r):
        """
        Given a r-value return g
        """
        return np.interp(r, self.r_data, self.g_data)

    def get_G(self, k_b=8.31446261815324, temp=300):
        """
        Galculate G(r)
        """
        _sum = np.sum(self.g_data)
        return -k_b * temp * np.log(self.g_data) / _sum


class Bounded_RDF(RDF):
    """
    Bounded radial distribution function class
    where the intepolated function is restricted to:
        g(r)> 0 : r : g(r) = 1

    This class finds those bounds and introduces a function
     r(g)
    """

    def __init__(self, name, atom_id, r, g):
        super().__init__(name, atom_id, r, g)
        self.bounds = self.determine_bounds(r, g)
        self.r_data, self.g_data = self.get_bounded_RDF_data(r, g, self.bounds)

    def determine_bounds(self, r, g):
        """
        Get the the r values of the bounded RDF such that:
            g(r) > 0 : r : g(r) = 1
        """
        bounds = [0, len(g)]
        bounds[0] = self.find_min_r(g)
        bounds[1] = self.find_max_r(1.1)

        return bounds

    def find_min_r(self, rdf_g_data):
        """
        Find the smallest r values from the RDF data such that:
             min r where g(r) > 0
        """
        r_loc = np.where([rdf_g_data == 0])[1][-1]

        return r_loc

    def find_max_r(self, g):
        """
        Find the smallest r value from the RDF data such that:
          all g(r) values are non-zero after r
        """
        find_r = g - self.g_data
        r_loc = np.where([find_r < 0])[1][0]

        return r_loc

    def get_bounded_RDF_data(self, r, g, bounds):
        """
        Set the bounds of the radial distribution function
        """
        r_out = r[bounds[0] : bounds[1]]
        g_out = g[bounds[0] : bounds[1]]

        return r_out, g_out

    def r(self, g):
        """
        Given a g-value return r
        Note! This only works for bounded RDFs
        """
        return np.interp(g, self.g_data, self.r_data)


@dataclass
class RDFBins:
    """Container for RDF binning data"""

    rdf_bins: Dict[int, np.ndarray]
    bin_widths: Dict[int, float]
    bin_centers: Dict[int, np.ndarray]
    shell_volumes: Dict[int, np.ndarray]


def sphere_volume(radius):
    return (4.0 / 3.0) * np.pi * radius * radius * radius


def generate_bins(radii, num_bins):
    """
    Generate the bins, bin widths and shell volumes for RDF calculation

    Args:
        atoms: AtomMap object containing atom lists
        num_bins: Number of bins for the RDF histogram

    Returns:
        RDFBins: Dataclass containing binning information:
            - rdf_bins: Dictionary of zero-initialized bins for each atom type
            - bin_widths: Dictionary of bin widths for each atom type
            - bin_centers: Dictionary of bin center positions
            - shell_volumes: Dictionary of shell volumes for normalization
    """
    bins_data = RDFBins(rdf_bins={}, bin_widths={}, bin_centers={}, shell_volumes={})

    for label, radius in radii.items():
        # Initialize RDF bins

        bins_data.rdf_bins[label] = np.zeros(num_bins)

        # Calculate bin width based on radius
        bins_data.bin_widths[label] = radius / num_bins

        # Calculate bin centers
        bins_data.bin_centers[label] = np.linspace(
            bins_data.bin_widths[label] / 2,
            bins_data.bin_widths[label] / 2 + radius,
            num_bins,
        )

        # Calculate shell volumes
        bins_data.shell_volumes[label] = sphere_volume(
            bins_data.bin_centers[label] + bins_data.bin_widths[label] / 2
        ) - sphere_volume(
            bins_data.bin_centers[label] - bins_data.bin_widths[label] / 2
        )

    return bins_data


def generate_rdf(binned_distances, bins):
    """
    Generate an rdf from binned distances
    """
    rdf = {}
    for label, binned in binned_distances.items():
        rdf[label] = binned / bins.shell_volumes[label] / np.sum(binned)

    return rdf


def bin_distances(subdomain, probe_atom_list, atoms, bins):
    """
    Finds the atoms that are within a radius of probe atom
    """

    if not isinstance(probe_atom_list, _particles.PyAtomList):
        raise TypeError(
            f"Expected probe_atom_list to be of type PyAtomList, got {type(probe_atom_list)}"
        )

    # Generate bins
    for label, atom_list in atoms.atom_map.items():

        # Ensure kd_tree built
        atom_list.build_KDtree()

        bins.rdf_bins[label] = _rdf._generate_rdf(
            probe_atom_list,
            atom_list,
            atom_list.radius,
            bins.rdf_bins[label],
            bins.bin_widths[label],
        )

    all_rdf = communication.all_gather(bins.rdf_bins)

    for n_proc, proc_rdf in enumerate(all_rdf):
        if n_proc == subdomain.rank:
            continue
        for label in proc_rdf:
            bins.rdf_bins[label] = [
                a + b for a, b in zip(bins.rdf_bins[label], proc_rdf[label])
            ]

    g_r = {}
    for label in atoms.labels:
        g_r[label] = np.asarray(bins.rdf_bins[label])

    return g_r

"""rdf.py"""

import numpy as np

__all__ = ["g_rdf"]

from . import _rdf


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


def g_rdf(subdomain, probe_atom, radius, atom_list, num_bins):
    """
    Finds the atoms that are within a radius of probe atom
    """
    if len(probe_atom.shape) == 1:
        probe_atom = probe_atom[np.newaxis, :]

    # binned_distances = _rdf.generate_rdf(
    #     subdomain, atom_list, probe_atom, radius, num_bins
    # )

    atoms = _rdf.generate_rdf(subdomain, atom_list, probe_atom, radius, num_bins)

    # # If probe_atom in atom_list
    # binned_distances[0] = binned_distances[0] - probe_atom.shape[0]

    # # probe_radius = probe_atom[0, 3]
    # num_atoms = atom_list.shape[0]
    # num_probe_atoms = probe_atom.shape[0]
    # bin_width = radius / num_bins
    # bin_centers = np.linspace(bin_width / 2, bin_width / 2 + radius, num_bins)

    # shell_volumes = (
    #     (4 / 3)
    #     * np.pi
    #     * ((bin_centers + bin_width / 2) ** 3 - (bin_centers - bin_width / 2) ** 3)
    # )

    # # Normalize RDF
    # number_density = num_probe_atoms / subdomain.domain.volume

    # g_r = binned_distances / (number_density * shell_volumes * num_probe_atoms)

    return atoms


def sphere_volume(radius):
    return (4.0 / 3.0) * np.pi * radius * radius * radius


def local_rdf(subdomain, probe_atom, radius, atom_list, num_bins):
    """
    Finds the atoms that are within a radius of probe atom
    """
    if len(probe_atom.shape) == 1:
        probe_atom = probe_atom[np.newaxis, :]

    binned_distances = _rdf.generate_rdf(
        subdomain, atom_list, probe_atom, radius, num_bins
    )

    # If probe_atom in atom_list
    binned_distances[0] = binned_distances[0] - probe_atom.shape[0]

    # probe_radius = probe_atom[0, 3]
    num_atoms = atom_list.shape[0]
    num_probe_atoms = probe_atom.shape[0]
    bin_width = radius / num_bins
    bin_centers = np.linspace(bin_width / 2, bin_width / 2 + radius, num_bins)

    shell_volumes = sphere_volume(bin_centers + bin_width / 2) - sphere_volume(
        bin_centers - bin_width / 2
    )

    # Normalize RDF
    number_density = num_probe_atoms / subdomain.domain.volume
    g_r = np.asarray(binned_distances) / (np.sum(binned_distances))

    return g_r, bin_centers

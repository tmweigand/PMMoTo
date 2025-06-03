"""bins.py

Bin utilities for PMMoTo, including 1D and radial bins, and distributed bin counting.
"""

from typing import Dict, Optional
import numpy as np
from numpy.typing import NDArray
from . import _bins
from ..io import io_utils
from ..core import communication

__all__ = ["count_locations"]


def sphere_volume(radius: float) -> float:
    """Calculate the volume of a sphere given its radius.

    Args:
        radius (float): Sphere radius.

    Returns:
        float: Volume of the sphere.

    """
    return (4.0 / 3.0) * np.pi * radius * radius * radius


class Bin:
    """Generic class for a bin."""

    def __init__(
        self,
        start: float,
        end: float,
        num_bins: int,
        name: Optional[str] = None,
        values: Optional[np.ndarray] = None,
    ):
        """Initialize a Bin.

        Args:
            start (float): Start of the bin range.
            end (float): End of the bin range.
            num_bins (int): Number of bins.
            name (str, optional): Name of the bin.
            values (np.ndarray, optional): Initial bin values.

        """
        self.start = start
        self.end = end
        self.num_bins = num_bins

        if name is not None:
            self.name = name
        else:
            self.name = None

        if values is not None:
            assert len(values) == num_bins
            self.values = values
        else:
            self.values = np.zeros(num_bins)
        self.volume = None
        self.get_bin_width()
        self.get_bin_centers()

    def get_bin_width(self) -> None:
        """Calculate and set the bin width."""
        self.width = (self.end - self.start) / self.num_bins

    def get_bin_centers(self) -> None:
        """Calculate and set the bin centers."""
        self.centers = np.linspace(
            self.start + self.width / 2,
            self.end - self.width / 2,
            self.num_bins,
        )

    def set_values(self, values: NDArray) -> None:
        """Set the bin values.

        Args:
            values (np.ndarray): Values to set.

        """
        self.values = values

    def update_values(self, values: np.ndarray) -> None:
        """Increment the bin values with new values.

        Args:
            values (np.ndarray): Values to add.

        """
        self.values += values

    def calculate_volume(self, area: float = None, radial_volume: bool = False) -> None:
        """Calculate the bin volume.

        Args:
            area (float, optional): Area for non-radial bins.
            radial_volume (bool, optional): If True, use radial volume.

        """
        if radial_volume:
            self.volume = sphere_volume(self.centers + self.width / 2) - sphere_volume(
                self.centers - self.width / 2
            )
        else:
            if area:
                self.volume = self.width * area

    def generate_rdf(self) -> NDArray[float]:
        """Generate an RDF from bin counts.

        Returns:
            np.ndarray: RDF values.

        """
        self.calculate_volume(radial_volume=True)
        _sum = np.sum(self.values)
        if _sum == 0:
            rdf = np.zeros_like(self.values)
        else:
            rdf = self.values / (self.volume * _sum) * sphere_volume(self.end)

        return rdf

    def save_bin(self, subdomain: "Subdomain", folder: str) -> None:
        """Save the bin data to a file.

        Args:
            subdomain: Subdomain object.
            folder (str): Output folder.

        """
        if subdomain.rank != 0:
            return  # Only the root process saves the bins

        # Create output directory if it doesn't exist
        io_utils.check_file_path(folder)

        # Stack the data into columns
        data = np.column_stack((self.centers, self.values))

        # Save with header
        out_file = folder + f"bin_{self.name}.txt"
        np.savetxt(
            out_file,
            data,
            delimiter="\t",
        )


class Bins:
    """Container for managing multiple bins."""

    def __init__(self, starts, ends, num_bins, labels, names=None):
        """Initialize Bins.

        Args:
            starts (list): Start values for each bin.
            ends (list): End values for each bin.
            num_bins (list): Number of bins for each.
            labels (list): Labels for each bin.
            names (list, optional): Names for each bin.

        """
        # Check inputs
        assert len(starts) == len(ends) == len(num_bins) == len(labels)
        self.bins = self.initialize_bins(starts, ends, num_bins, labels, names)

    def initialize_bins(self, starts, ends, num_bins, labels, names=None):
        """Initialize the bins.

        Args:
            starts (list): Start values for each bin.
            ends (list): End values for each bin.
            num_bins (list): Number of bins for each.
            labels (list): Labels for each bin.
            names (list, optional): Names for each bin.

        Returns:
            dict: Dictionary of Bin objects.

        """
        if names is None:
            names = [str(label) for label in labels]

        bins = {}
        for start, end, num_bin, label, name in zip(
            starts, ends, num_bins, labels, names
        ):
            bins[label] = Bin(start, end, num_bin, name)

        return bins

    def update_bins(self, values: Dict[int, np.ndarray]):
        """Update the bin counts.

        Args:
            values (Dict[int, np.ndarray]): Counts to add.

        """
        for label, data in values.items():
            self.bins[label].values += data

    def save_bins(self, subdomain, folder):
        """Save all bins to files.

        Args:
            subdomain: Subdomain object.
            folder (str): Output folder.

        """
        if subdomain.rank != 0:
            return  # Only the root process saves the bins

        # Create output directory if it doesn't exist
        io_utils.check_file_path(folder)

        # Save data for each atom type
        for label, bin in self.bins.items():
            centers = bin.centers

            # Stack the data into columns
            data = np.column_stack((centers, bin.values))

            # Save with header
            out_file = folder + f"bins_{bin.name}.txt"
            header = f"Atom Type: {bin.name} \nAtom Label: {label}"
            np.savetxt(out_file, data, delimiter="\t", header=header)


def count_locations(coordinates, dimension, bin, subdomain=None):
    """Count the number of atoms in a bin.

    Args:
        coordinates (np.ndarray): Atom coordinates.
        dimension (int): Dimension to sum the bins.
        bin (Bin): Bin object. Results are stored in bin.values.
        subdomain (optional): Subdomain object for distributed sum.

    Note:
        Repeated calls to this function increment bin.values!

    """
    _counts = _bins._count_locations(
        coordinates, dimension, bin.num_bins, bin.width, bin.start
    )

    if subdomain is not None:
        _counts = _sum_process_bins(subdomain, _counts)

    bin.update_values(_counts)


def sum_masses(coordinates, dimension, bin, masses, subdomain=None):
    """Sum the masses of atoms in a bin.

    Args:
        coordinates (np.ndarray): Atom coordinates.
        dimension (int): Dimension to sum the bins.
        bin (Bin): Bin object.
        masses (np.ndarray): Atom masses.
        subdomain (optional): Subdomain object for distributed sum.

    """
    _masses = _bins._sum_masses(
        coordinates, masses, dimension, bin.num_bins, bin.width, bin.start
    )

    if subdomain is not None:
        _masses = _sum_process_bins(subdomain, _masses)

    bin.update_values(_masses)


def _sum_process_bins(subdomain, counts):
    """Sum the bins across all processes.

    Args:
        subdomain: Subdomain object.
        counts (np.ndarray): Array of counts.

    Returns:
        np.ndarray: Summed counts across all processes.

    """
    _all_counts = communication.all_gather(counts)

    # Sum contributions from all processes
    for n_proc, proc_data in enumerate(_all_counts):
        if subdomain.rank == n_proc:
            continue

        counts = counts + proc_data

    return counts

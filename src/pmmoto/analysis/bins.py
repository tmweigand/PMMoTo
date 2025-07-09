"""bins.py

Bin utilities for PMMoTo, including 1D and radial bins, and distributed bin counting.

This module provides classes and functions for binning data, such as atom locations,
and for computing radial distribution functions (RDFs) and distributed bin statistics.
"""

from __future__ import annotations
from typing import Sequence, Any, TYPE_CHECKING, TypeVar
import numpy as np
from numpy.typing import NDArray
from . import _bins
from ..io import io_utils
from ..core import communication

if TYPE_CHECKING:
    from pmmoto.core.subdomain import Subdomain

T = TypeVar("T", bound=np.generic)


__all__ = ["count_locations"]


class Bin:
    """Generic class for a bin.

    Represents a 1D or radial bin for counting or accumulating values.
    Provides methods for computing bin centers, widths, volumes, and for
    updating and saving bin data.
    """

    def __init__(
        self,
        start: float,
        end: float,
        num_bins: int,
        name: str | None = None,
        values: NDArray[np.number[Any]] | None = None,
    ):

        self.start = start
        self.end = end
        self.num_bins = num_bins

        if name is not None:
            self.name = name

        if values is not None:
            assert len(values) == num_bins
            self.values = values
        else:
            self.values = np.zeros(num_bins)

        self.width = self.get_bin_width()
        self.centers = self.get_bin_centers()
        self.volume = self.calculate_volume()

    def get_bin_width(self) -> float:
        """Calculate and return the bin width.

        Returns:
            float: The width of each bin.

        """
        width = (self.end - self.start) / self.num_bins
        return width

    def get_bin_centers(self) -> NDArray[np.float64]:
        """Calculate and return the bin centers.

        Returns:
            np.ndarray: Array of bin center positions.

        """
        return np.linspace(
            self.start + self.width / 2,
            self.end - self.width / 2,
            self.num_bins,
            dtype=np.float64,
        )

    def set_values(self, values: NDArray[np.number[Any]]) -> None:
        """Set the bin values.

        Args:
            values (np.ndarray): Values to set for the bins.

        """
        self.values = values

    def update_values(self, values: NDArray[np.number[Any]]) -> None:
        """Increment the bin values with new values.

        Args:
            values (np.ndarray): Values to add to the bins.

        """
        if self.values is not None:
            self.values = self.values + values

    def calculate_volume(
        self, area: float | None = None, radial_volume: bool = False
    ) -> NDArray[np.floating[Any]] | float:
        """Calculate and return the bin volume.

        Args:
            area (float, optional): Area for non-radial bins.
            radial_volume (bool, optional): If True, use radial volume.

        Returns:
            np.ndarray or float: Volume(s) of the bins.

        """
        if radial_volume:
            volume = radial_bin_volume(
                self.centers + self.width / 2
            ) - radial_bin_volume(self.centers - self.width / 2)
        elif area is not None:
            volume = self.width * area
        else:
            volume = np.ones_like(self.centers)  # fallback to 1s to avoid None

        return volume

    def generate_rdf(self) -> NDArray[np.floating[Any]]:
        """Generate a radial distribution function (RDF) from bin counts.

        Returns:
            np.ndarray: RDF values for each bin.

        """
        self.calculate_volume(radial_volume=True)
        _sum = np.sum(self.values)
        if _sum == 0:
            rdf = np.zeros_like(self.values)
        else:
            rdf = self.values / (self.volume * _sum) * radial_bin_volume(self.end)

        return rdf

    def save_bin(self, subdomain: Subdomain, folder: str) -> None:
        """Save the bin data to a file.

        Only the root process (rank 0) saves the bin data.

        Args:
            subdomain: Subdomain object.
            folder (str): Output folder path.

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
    """Container for managing multiple bins.

    This class manages a collection of Bin objects, allowing for batch
    initialization, updating, and saving of multiple bins.
    """

    def __init__(
        self,
        starts: Sequence[float],
        ends: Sequence[float],
        num_bins: Sequence[int],
        labels: Sequence[int],
        names: Sequence[str] | None = None,
    ) -> None:
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

    def initialize_bins(
        self,
        starts: Sequence[float],
        ends: Sequence[float],
        num_bins: Sequence[int],
        labels: Sequence[int],
        names: Sequence[str] | None = None,
    ) -> dict[int, Bin]:
        """Initialize the bins and return a dictionary of Bin objects.

        Args:
            starts (list): Start values for each bin.
            ends (list): End values for each bin.
            num_bins (list): Number of bins for each.
            labels (list): Labels for each bin.
            names (list, optional): Names for each bin.

        Returns:
            dict: Dictionary mapping label to Bin object.

        """
        if names is None:
            names = [str(label) for label in labels]

        bins = {}
        for start, end, num_bin, label, name in zip(
            starts, ends, num_bins, labels, names
        ):
            bins[label] = Bin(start, end, num_bin, name)

        return bins

    def update_bins(self, values: dict[int, NDArray[np.floating[Any]]]) -> None:
        """Update the bin counts for all bins.

        Args:
            values (Dict[int, np.ndarray]): Counts to add to each bin.

        """
        for label, data in values.items():
            self.bins[label].values = self.bins[label].values + data

    def save_bins(self, subdomain: Subdomain, folder: str) -> None:
        """Save all bins to files.

        Only the root process (rank 0) saves the bin data.

        Args:
            subdomain: Subdomain object.
            folder (str): Output folder path.

        """
        if subdomain.rank != 0:
            return  # Only the root process saves the bins

        # Create output directory if it doesn't exist
        io_utils.check_file_path(folder)

        # Save data for each atom type
        for label, _bin in self.bins.items():
            centers = _bin.centers

            # Stack the data into columns
            data = np.column_stack((centers, _bin.values))

            # Save with header
            out_file = folder + f"bins_{_bin.name}.txt"
            header = f"Atom Type: {_bin.name} \nAtom Label: {label}"
            np.savetxt(out_file, data, delimiter="\t", header=header)


def count_locations(
    coordinates: NDArray[np.floating[Any]],
    dimension: int,
    bin: Bin,
    parallel: bool = True,
) -> None:
    """Count the number of atoms or objects in each bin.

    This function increments the bin values by counting the number of
    coordinates that fall into each bin along the specified dimension.
    If a subdomain is provided, the counts are summed across all processes.

    Args:
        coordinates (np.ndarray): Atom or object coordinates.
        dimension (int): Dimension along which to bin.
        bin (Bin): Bin object. Results are stored in bin.values.
        parallel (bool): If the bins need to be summed in parallel

    Note:
        Repeated calls to this function increment bin.values!

    """
    _counts = _bins._count_locations(
        coordinates, dimension, bin.num_bins, bin.width, bin.start
    )

    if parallel:
        _counts = _sum_process_bins(_counts)

    bin.update_values(_counts)


def sum_masses(
    coordinates: NDArray[np.floating[Any]],
    dimension: int,
    bin: Bin,
    masses: NDArray[np.number[Any]],
    parallel: bool = True,
) -> None:
    """Sum the masses of atoms or objects in each bin.

    This function increments the bin values by summing the masses of
    coordinates that fall into each bin along the specified dimension.
    If a subdomain is provided, the sums are aggregated across all processes.

    Args:
        coordinates (np.ndarray): Atom or object coordinates.
        dimension (int): Dimension along which to bin.
        bin (Bin): Bin object.
        masses (np.ndarray): Masses of the atoms or objects.
        parallel (bool): If the bins need to be summed in parallel

    """
    _masses = _bins._sum_masses(
        coordinates, masses, dimension, bin.num_bins, bin.width, bin.start
    )

    if parallel:
        _masses = _sum_process_bins(_masses)

    bin.update_values(_masses)


def _sum_process_bins(
    counts: NDArray[T],
) -> NDArray[T]:
    """Sum the bins across all processes.

    This function gathers bin counts from all processes and sums them,
    returning the total counts for the calling process.

    Args:
        counts (np.ndarray): Array of counts.

    Returns:
        np.ndarray: Summed counts across all processes.

    """
    _all_counts = communication.all_gather(counts)

    out_count = np.zeros_like(counts)

    # Sum contributions from all processes
    for proc_data in _all_counts:
        np.add(out_count, proc_data, out=out_count)

    return out_count


def radial_bin_volume(
    radius: float | NDArray[np.floating[Any]],
) -> float | NDArray[np.floating[Any]]:
    """Calculate the volume of a sphere given its radius.

    This function is used for computing the volume of radial bins, which is
    important for normalization in radial distribution functions (RDFs).

    Args:
        radius (float or np.ndarray): Sphere radius or array of radii.

    Returns:
        float or np.ndarray: Volume(s) of the sphere(s).

    """
    return (4.0 / 3.0) * np.pi * radius * radius * radius

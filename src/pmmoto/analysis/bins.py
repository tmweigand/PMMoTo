"""bins.py"""

from typing import Dict, Optional
import numpy as np

from ..io import io_utils


def sphere_volume(radius):
    return (4.0 / 3.0) * np.pi * radius * radius * radius


class Bin:
    """
    Generic class for a bin.
    """

    def __init__(
        self,
        start: float,
        end: float,
        num_bins: int,
        name: Optional[str] = None,
        values: Optional[np.ndarray] = None,
    ):
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

    def get_bin_width(self):
        """
        Bin width
        """
        self.width = (self.end - self.start) / self.num_bins

    def get_bin_centers(self):
        """
        Bin centers
        """
        self.centers = np.linspace(
            self.width / 2,
            self.end,
            self.num_bins,
        )

    def calculate_volume(self, area=None, radial_volume=False):
        """
        Bin volume
        """

        if radial_volume:
            self.volume = sphere_volume(self.centers + self.width / 2) - sphere_volume(
                self.centers - self.width / 2
            )
        else:
            if area:
                self.volume = self.width * area

    def generate_rdf(self):
        """
        Generate an rdf bin counts
        """
        self.calculate_volume(radial_volume=True)
        _sum = np.sum(self.values)
        if _sum == 0:
            rdf = np.zeros_like(self.values)
        else:
            rdf = self.values / self.volume / _sum

        return rdf


class Bins:
    """
    Container for managing multiple bins.
    """

    def __init__(self, starts, ends, num_bins, labels, names=None):
        # Check inputs
        assert len(starts) == len(ends) == len(num_bins) == len(labels)
        self.bins = self.initialize_bins(starts, ends, num_bins, labels, names)

    def initialize_bins(self, starts, ends, num_bins, labels, names=None):
        """
        Initialize the bins
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
        """
        Update the bin counts

        Args:
            values (Dict[np.ndarray]): counts to add
        """

        for label, data in values.items():
            self.bins[label].values += data

    def save_bins(self, subdomain, folder):
        """
        Save the bins
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
            np.savetxt(
                out_file,
                data,
                delimiter="\t",
            )

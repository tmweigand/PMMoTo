"""porosimetry.py"""

import numpy as np

__all__ = ["get_sizes"]


def get_sizes(min_value, max_value, num_values, spacing="linear"):
    """
    Give list of pore sizes based on inputs provided
    """
    if min_value >= max_value:
        raise ValueError(
            f"Error: min_value {min_value} must be greater than max value {max_value}"
        )

    if num_values <= 0:
        raise ValueError(f"Error: num_values {num_values} must be greater than 0")

    if spacing == "linear":
        values = np.linspace(min_value, max_value, num_values)

    elif spacing == "log":
        values = np.logspace(min_value, max_value, num_values)

    else:
        raise ValueError(f"spacing {spacing} can only be 'linear' or 'log'")

    return values

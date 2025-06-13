"""Universal force field atom radii data for PMMoTo.

Provides a dictionary mapping element names or atomic numbers to
(atomic_number, radius) pairs, where radius is sigma/2 in Angstroms.
"""


def atom_universal_force_field() -> dict[str | int, tuple[int, float]]:
    """Return universal force field data for atom radii.

    Provides a dictionary mapping element names or atomic numbers to
    (atomic_number, sigma/2) pairs, sigma is in Angstroms
                             and divided by 2 to yield radius.

    Data source:
    https://github.com/SarkisovGitHub/PoreBlazer/blob/main/src/UFF.atoms

    Returns:
        dict: Mapping of element names or atomic numbers to (atomic_number, radius).

    """
    element_table: dict[str | int, tuple[int, float]] = {}

    # Adding elements with (atomic_number, sigma (Angstroms)) pairs
    element_table["H"] = (1, 2.571 / 2)
    element_table[1] = (1, 2.571 / 2)

    element_table["C"] = (6, 3.431 / 2)
    element_table[6] = (6, 3.431 / 2)

    element_table["N"] = (7, 3.261 / 2)
    element_table[7] = (7, 3.261 / 2)

    element_table["O"] = (8, 3.118 / 2)
    element_table[8] = (8, 3.118 / 2)

    element_table["Cl"] = (17, 3.516 / 2)
    element_table[17] = (17, 3.516 / 2)

    return element_table

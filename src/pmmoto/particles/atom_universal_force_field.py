def atom_universal_force_field():
    """Universal force field data given as element name or atomic number
    and yields atomic number and sigma (Angstroms). To use with pmmoto,
    must divide by 2 so a radius.

    Data taken from:
        https://github.com/SarkisovGitHub/PoreBlazer/blob/main/src/UFF.atoms

    Will add more soon

    """
    element_table = {}

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

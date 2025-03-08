"""test_inkbottle.py"""

from mpi4py import MPI
import numpy as np
import pytest
import pmmoto


capillary_pressure = [
    1.58965,
    1.5943,
    1.60194,
    1.61322,
    1.62893,
    1.65002,
    1.67755,
    1.7127,
    1.75678,
    1.81122,
    1.87764,
    1.95783,
    2.05388,
    2.16814,
    2.30332,
    2.4625,
    2.64914,
    2.86704,
    3.12024,
    3.41274,
    3.74806,
    4.12854,
    4.55421,
    5.02123,
    5.52008,
    6.03352,
    6.53538,
    6.9909,
    7.36005,
    7.60403,
    7.69393,
    7.69409,
]


w_saturation_expected = [
    0.914608928,
    0.914608928,
    0.914608928,
    0.903539715,
    0.903539715,
    0.895876414,
    0.885658679,
    0.877508819,
    0.870696995,
    0.865466488,
    0.86047926,
    0.856343511,
    0.853545797,
    0.840651989,
    0.838462474,
    0.837124437,
    0.830190974,
    0.829461136,
    0.827514901,
    0.826055224,
    0.822649313,
    0.821919475,
    0.820824717,
    0.818513563,
    0.817418805,
    0.816810607,
    0.816810607,
    0.816445688,
    0.81571585,
    0.81571585,
    0.81571585,
    0.81571585,
]


def initialize_ink_bottle(parallel=False):

    if parallel:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        subdomains = (8, 1, 1)
    else:
        subdomains = (1, 1, 1)
        rank = 0

    voxels = (140, 30, 30)
    reservoir_voxels = 10

    box = ((0.0, 14.0), (-1.5, 1.5), (-1.5, 1.5))

    inlet = ((0, 1), (0, 0), (0, 0))
    outlet = ((1, 0), (0, 0), (0, 0))

    sd = pmmoto.initialize(
        voxels,
        rank=rank,
        box=box,
        subdomains=subdomains,
        inlet=inlet,
        outlet=outlet,
        reservoir_voxels=reservoir_voxels,
    )

    pm = pmmoto.domain_generation.gen_pm_inkbottle(sd)
    mp = pmmoto.domain_generation.gen_mp_constant(pm, 2)

    return mp


def test_serial_drainage():

    mp = initialize_ink_bottle()

    w_saturation = pmmoto.filters.equilibrium_distribution.drainage(
        mp, capillary_pressure
    )

    np.testing.assert_array_almost_equal(w_saturation, w_saturation_expected)


@pytest.mark.mpi(min_size=8)
def test_parallel_drainage():

    mp = initialize_ink_bottle(True)

    w_saturation = pmmoto.filters.equilibrium_distribution.drainage(
        mp, capillary_pressure
    )

    np.testing.assert_array_almost_equal(w_saturation, w_saturation_expected)

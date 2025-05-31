import pmmoto

import profiling_utils


def initialize_ink_bottle():
    """Initialize ink bottle
    """
    # voxels = (560, 120, 120)  ##res = 40
    # reservoir_voxels = 40

    voxels = (1120, 240, 240)  ##res = 80
    reservoir_voxels = 80

    box = ((0.0, 14.0), (-1.5, 1.5), (-1.5, 1.5))

    inlet = ((0, 1), (0, 0), (0, 0))
    outlet = ((1, 0), (0, 0), (0, 0))

    sd = pmmoto.initialize(
        voxels,
        box=box,
        inlet=inlet,
        outlet=outlet,
        reservoir_voxels=reservoir_voxels,
    )

    pm = pmmoto.domain_generation.gen_pm_inkbottle(sd)
    mp = pmmoto.domain_generation.gen_mp_constant(pm, 2)

    return sd, pm, mp


@profiling_utils.profile("profiling/drainage.prof")
def test_drainage():
    """Profiling for connected components.
    To run:
        python profiling/drainage_profiling.py
    Note: Cannot be used on python 12!!!!
    """
    sd, pm, mp = initialize_ink_bottle()

    mp_img, w_saturation = pmmoto.filters.equilibrium_distribution.calcDrainage(
        7.41274, mp
    )


if __name__ == "__main__":
    test_drainage()

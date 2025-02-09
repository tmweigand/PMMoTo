import profiling_utils
import pmmoto


@profiling_utils.profile("profiling/edt_periodic.prof")
def test_edt_periodic_profile():
    """
    Profiling for edt.
    To run:
        python profiling/edt_profiling.py
    Note: Cannot be used on python 12!!!!
    """

    voxels = (600, 600, 600)
    prob_zero = 0.5
    seed = 1
    img = pmmoto.domain_generation.gen_random_binary_grid(voxels, prob_zero, seed)
    _edt = pmmoto.filters.distance.edt3d(img, periodic=[True, True, True])


@profiling_utils.profile("profiling/edt_non_periodic.prof")
def test_edt_profile():
    """
    Profiling for edt.
    To run:
        python profiling/edt_profiling.py
    Note: Cannot be used on python 12!!!!
    """

    voxels = (600, 600, 600)
    prob_zero = 0.5
    seed = 1
    img = pmmoto.domain_generation.gen_random_binary_grid(voxels, prob_zero, seed)
    _edt = pmmoto.filters.distance.edt3d(img, periodic=[False, False, False])


if __name__ == "__main__":
    test_edt_profile()
    test_edt_periodic_profile()

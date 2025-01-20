import cProfile
import pmmoto
from functools import wraps


def profile(filename=None):
    """
    A decorator to profile a function using cProfile.
    Saves the profile results to a file if a filename is provided.
    """

    def prof_decorator(f):
        @wraps(f)
        def wrap_f(*args, **kwargs):
            pr = cProfile.Profile()
            pr.enable()
            result = f(*args, **kwargs)
            pr.disable()

            if filename:
                pr.dump_stats(filename)
            else:
                pr.print_stats()

            return result

        return wrap_f

    return prof_decorator


@profile("profiling/edt_periodic.prof")
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


@profile("profiling/edt_non_periodic.prof")
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

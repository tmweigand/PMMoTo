import profiling_utils
import pmmoto


def setup():
    """
    Setup for for benchmarking morphological operators
    """
    voxels = (600, 600, 600)
    prob_zero = 0.1
    seed = 1
    sd = pmmoto.initialize(voxels)
    img = pmmoto.domain_generation.gen_random_binary_grid(voxels, prob_zero, seed)
    return sd, img


@profiling_utils.profile("profiling/morph_addition_fft_small_r.prof")
def test_morp_addition_fft_small_r():
    """
    Profiling for edt.
    To run:
        python profiling/edt_profiling.py
    Note: Cannot be used on python 12!!!!
    """
    radius = 0.004
    fft = True
    sd, img = setup()
    morp_addition_fft = pmmoto.filters.morphological_operators.addition(
        subdomain=sd,
        img=img,
        radius=radius,
        fft=fft,
    )


@profiling_utils.profile("profiling/morph_addition_fft_large_r.prof")
def test_morp_addition_fft_large_r():
    """ """
    radius = 0.1
    fft = True
    sd, img = setup()
    morp_addition_fft = pmmoto.filters.morphological_operators.addition(
        subdomain=sd,
        img=img,
        radius=radius,
        fft=fft,
    )


@profiling_utils.profile("profiling/morph_addition_edt_small_r.prof")
def test_morp_addition_edt_small_r():
    """ """
    radius = 0.004
    fft = False
    sd, img = setup()
    morp_addition_edt = pmmoto.filters.morphological_operators.addition(
        subdomain=sd,
        img=img,
        radius=radius,
        fft=fft,
    )


@profiling_utils.profile("profiling/morph_addition_edt_large_r.prof")
def test_morp_addition_edt_large_r():
    """ """
    radius = 0.1
    fft = False
    sd, img = setup()
    morp_addition_edt = pmmoto.filters.morphological_operators.addition(
        subdomain=sd,
        img=img,
        radius=radius,
        fft=fft,
    )


if __name__ == "__main__":
    test_morp_addition_fft_small_r()
    test_morp_addition_fft_large_r()
    test_morp_addition_edt_small_r()
    test_morp_addition_edt_large_r()

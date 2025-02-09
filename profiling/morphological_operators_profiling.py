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
def test_morp_addition_fft_small_r(sd, img, radius):
    """
    Wrapper to addition
    """

    morp_addition_fft = pmmoto.filters.morphological_operators.addition(
        subdomain=sd,
        img=img,
        radius=radius,
        fft=True,
    )


@profiling_utils.profile("profiling/morph_addition_fft_large_r.prof")
def test_morp_addition_fft_large_r(sd, img, radius):
    """
    Wrapper to addition
    """
    morp_addition_fft = pmmoto.filters.morphological_operators.addition(
        subdomain=sd,
        img=img,
        radius=radius,
        fft=True,
    )


@profiling_utils.profile("profiling/morph_addition_edt_small_r.prof")
def test_morp_addition_edt_small_r(sd, img, radius):
    """
    Wrapper to addition
    """
    morp_addition_edt = pmmoto.filters.morphological_operators.addition(
        subdomain=sd,
        img=img,
        radius=radius,
        fft=False,
    )


@profiling_utils.profile("profiling/morph_addition_edt_large_r.prof")
def test_morp_addition_edt_large_r(sd, img, radius):
    """
    Wrapper to addition
    """
    morp_addition_edt = pmmoto.filters.morphological_operators.addition(
        subdomain=sd,
        img=img,
        radius=radius,
        fft=False,
    )


if __name__ == "__main__":
    small_radius = 0.004
    large_radius = 0.1
    sd, img = setup()
    test_morp_addition_fft_small_r(sd, img, small_radius)
    test_morp_addition_fft_large_r(sd, img, large_radius)
    test_morp_addition_edt_small_r(sd, img, small_radius)
    test_morp_addition_edt_large_r(sd, img, large_radius)

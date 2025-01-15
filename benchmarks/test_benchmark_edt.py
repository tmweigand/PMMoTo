from pmmoto import domain_generation
from pmmoto import filters
import edt


# def pmmoto_edt(img):
#     return filters.distance.edt3d(img)


def test_pmmoto_edt(benchmark):
    """ """
    voxels = (300, 300, 300)
    prob_zero = 0.1
    seed = 1
    img = domain_generation.gen_random_binary_grid(voxels, prob_zero, seed)
    edt_pmmoto = benchmark(filters.distance.edt3d, img, periodic=[False, False, False])


def test_pmmoto_periodic_edt(benchmark):
    """ """
    voxels = (300, 300, 300)
    prob_zero = 0.1
    seed = 1
    img = domain_generation.gen_random_binary_grid(voxels, prob_zero, seed)
    edt_pmmoto = benchmark(filters.distance.edt3d, img, periodic=[True, True, True])


def test_edt(benchmark):
    """ """
    voxels = (300, 300, 300)
    prob_zero = 0.1
    seed = 1
    img = domain_generation.gen_random_binary_grid(voxels, prob_zero, seed)
    img_edt = benchmark(edt.edt, img)

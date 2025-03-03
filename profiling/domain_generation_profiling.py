import profiling_utils
import pmmoto
import numpy as np
import time


@profiling_utils.profile("profiling/domain_generation_sphere_pack.prof")
def test_domain_generation_sphere_pack():
    """
    Profiling for domain generation.
    Note: Cannot be used on python 12!!!!
    """
    num_spheres = 500
    spheres = np.random.rand(num_spheres, 4)
    spheres[:, 3] = 0.0

    voxels = (300, 300, 300)
    sd = pmmoto.initialize(voxels, verlet_domains=[1, 1, 1])

    start_time = time.perf_counter()
    img = pmmoto.domain_generation.gen_pm_spheres_domain(sd, spheres)
    end_time = time.perf_counter()
    runtime = end_time - start_time
    print(f"Execution time: {runtime:.6f} seconds")


def test_domain_generation_sphere_pack_verlet():
    """
    Profiling for domain generation.
    Note: Cannot be used on python 12!!!!
    """
    num_spheres = 50000
    spheres = np.random.rand(num_spheres, 4)
    spheres[:, 3] = spheres[:, 3] * 0.0001

    spheres = spheres[np.lexsort((spheres[:, 0], spheres[:, 1], spheres[:, 2]))]

    voxels = (500, 500, 500)

    for n in range(10, 45):
        sd = pmmoto.initialize(voxels, verlet_domains=[n, n, n])

        start_time = time.perf_counter()
        img = pmmoto.domain_generation.gen_pm_spheres_domain(sd, spheres)
        end_time = time.perf_counter()
        runtime = end_time - start_time
        print(f"Execution time: {runtime:.6f} seconds for {n} Verlet domain")


def test_domain_generation_sphere_pack_kd():
    """
    Profile the generation of the Verlet lists
    """
    num_spheres = 50000
    spheres = np.random.rand(num_spheres, 4)
    spheres[:, 3] = spheres[:, 3] * 0.0001
    voxels = (300, 300, 300)

    for n in range(1, 25):
        sd = pmmoto.initialize(voxels, verlet_domains=[n, n, n])

        start_time = time.perf_counter()
        pmmoto.domain_generation._domain_generation.gen_pm_sphere(sd, spheres, kd=True)
        end_time = time.perf_counter()
        runtime = end_time - start_time

        print(
            f"Execution time for kd: {runtime:.6f} seconds with {n,n,n} Verlet domains"
        )

    for n in range(20, 25):
        sd = pmmoto.initialize(voxels, verlet_domains=[n, n, n])

        start_time = time.perf_counter()
        img = pmmoto.domain_generation.gen_pm_spheres_domain(sd, spheres)
        end_time = time.perf_counter()
        runtime = end_time - start_time
        print(f"Execution time: {runtime:.6f} seconds for {n,n,n} Verlet domains")


if __name__ == "__main__":
    # test_domain_generation_sphere_pack()
    # test_domain_generation_sphere_pack_verlet()
    test_domain_generation_sphere_pack_kd()

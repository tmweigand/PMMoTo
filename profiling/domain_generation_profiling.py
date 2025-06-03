"""Profiling script for domain generation in PMMoTo.

Profiles the performance of sphere pack domain generation with KD-tree and Verlet lists.
"""

import profiling_utils
import pmmoto
import numpy as np
import time


@profiling_utils.profile("profiling/domain_generation_sphere_pack.prof")
def test_domain_generation_sphere_pack():
    """Profiling for domain generation.

    Note: Cannot be used on python 12!!!!
    """
    num_spheres = 500
    spheres = np.random.rand(num_spheres, 4)
    spheres[:, 3] = 0.0

    voxels = (300, 300, 300)
    sd = pmmoto.initialize(voxels, verlet_domains=[1, 1, 1])

    start_time = time.perf_counter()
    _ = pmmoto.domain_generation.gen_pm_spheres_domain(sd, spheres)
    end_time = time.perf_counter()
    runtime = end_time - start_time
    print(f"Execution time: {runtime:.6f} seconds")


def test_domain_generation_sphere_pack_verlet():
    """Profiling for domain generation.

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
        _ = pmmoto.domain_generation.gen_pm_spheres_domain(sd, spheres)
        end_time = time.perf_counter()
        runtime = end_time - start_time
        print(f"Execution time: {runtime:.6f} seconds for {n} Verlet domain")


def test_domain_generation_sphere_pack_kd():
    """Profile the generation of the Verlet lists."""
    num_spheres = 50000
    spheres = np.random.rand(num_spheres, 4)
    spheres[:, 3] = spheres[:, 3] * 0.0001

    sd = pmmoto.initialize(voxels=(300, 300, 300))

    start_time = time.perf_counter()
    pmmoto.domain_generation.gen_pm_spheres_domain(sd, spheres, kd=True)

    print(f"Execution time for kd: {(time.perf_counter() - start_time):.6f} seconds")


if __name__ == "__main__":
    # test_domain_generation_sphere_pack()
    # test_domain_generation_sphere_pack_verlet()
    test_domain_generation_sphere_pack_kd()

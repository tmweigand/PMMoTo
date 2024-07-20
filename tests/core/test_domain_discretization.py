"""test_domain_discretization.py"""
import pmmoto

def test_discretized_domain(domain,
                            decomposed_domain,
                            domain_discretization
                            ):
    """
    Test for checking initialization of domain values
    """

    discretized_domain = pmmoto.core.DiscretizedDomain(
        voxels = domain_discretization['voxels'],
        size_domain = domain['size_domain'],
        boundaries = domain['boundaries'],
        inlet = domain['inlet'],
        outlet = domain['outlet']
    )

    assert all(discretized_domain.voxel == 1)
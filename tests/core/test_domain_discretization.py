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
        num_voxels = domain_discretization['num_voxels'],
        box = domain['box'],
        boundaries = domain['boundaries'],
        inlet = domain['inlet'],
        outlet = domain['outlet']
    )

    assert all(discretized_domain.num_voxels) == 1
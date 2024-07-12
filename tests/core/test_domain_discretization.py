"""test_domain_discretization.py"""
import pmmoto

def test_discretized_domain(domain,
                            decomposed_domain,
                            domain_discretization
                            ):
    """
    Test for checking initialization of domain values
    """
    domain = pmmoto.core.Domain(
        domain['size_domain'],
        domain['boundaries'],
        domain['inlet'],
        domain['outlet']
    )

    discretized_domain = pmmoto.core.DiscretizedDomain(
        domain,
        decomposed_domain['subdomain_map'],
        domain_discretization['nodes']
    )

    assert all(discretized_domain.voxel == 1)
    assert all(discretized_domain.sd_nodes == 50)
    assert all(discretized_domain.rem_nodes == 0)
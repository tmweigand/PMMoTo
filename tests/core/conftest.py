import pytest
import pickle
import numpy as np


@pytest.fixture
def domain():
    """
    Domain data to pass into tests
    """
    data = {
        "box": ((0, 100), (0, 100), (0, 100)),
        "boundaries": ((0, 0), (1, 1), (2, 2)),
        "inlet": ((1, 0), (0, 0), (0, 0)),
        "outlet": ((0, 1), (0, 0), (0, 0)),
        "length": np.array([100, 100, 100]),
    }

    return data


@pytest.fixture
def domain_discretization():
    """
    Domain discretization data to pass into tests.
    """
    data = {"voxels": (10, 10, 10)}

    return data


@pytest.fixture
def domain_discretization_true():
    """
    Domain discretization results to pass into tests.
    """
    with open("tests/core/test_output/test_discretized_domain.pkl", "rb") as file:
        data = pickle.load(file)

    return data


@pytest.fixture
def domain_decomposed():
    """
    Domain decomposition data to pass into tests.
    """
    data = {"subdomain_map": (3, 3, 3)}

    return data


@pytest.fixture
def domain_decomposed_true():
    """
    Domain decomposition results to pass into tests.
    """
    with open("tests/core/test_output/test_decomposed_domain.pkl", "rb") as file:
        data = pickle.load(file)

    return data


@pytest.fixture
def subdomains():
    """
    Just changing the name of domain_decomposed_true
    """
    with open("tests/core/test_output/test_decomposed_domain.pkl", "rb") as file:
        data = pickle.load(file)

    return data


@pytest.fixture
def subdomains_padded_true():
    """
    PAdded subdomain results
    """
    with open("tests/core/test_output/test_subdomain_padded.pkl", "rb") as file:
        data = pickle.load(file)

    return data

import pytest
from numpy import isclose, linspace, sqrt

from xs2arr import Model
from xs2arr.eedf import Maxwellian
from xs2arr.rate import compute_rate

EXAMPLE_LXCAT_FILE = "tests/example_lxcat.txt"


def test_model():
    """Test the robustness of the model class from an example LXCat data."""

    model = Model(EXAMPLE_LXCAT_FILE)

    results = model.arrhenius()  # noqa: F841


@pytest.mark.parametrize("T", [1.0, 2.0, 3.0, 4.0, 5.0])
def test_rate_integral(T: float):
    """Test the rate integral calculation for an analytic example."""

    energies = linspace(start=0.1, stop=100.0, num=1000, dtype=float)

    # This gives rate integral as purely integral of the EEDF.
    xs = 1.0 / sqrt(energies)

    EEDF = Maxwellian

    rate = compute_rate(energies, xs, EEDF(T).pdf(energies))

    # EEDF is normalised, therefore rate should be close to 1.
    assert isclose(rate, 1.0, atol=1e-1)

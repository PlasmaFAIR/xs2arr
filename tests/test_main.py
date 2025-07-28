import pytest
from numpy import isclose, linspace, sqrt

from xs2arr import Model
from xs2arr.eedf import Druyvesteyn, Maxwellian
from xs2arr.rate import compute_rate_simpson, compute_rate_trapezoid

EXAMPLE_LXCAT_FILE = "tests/example_lxcat.txt"


def test_model():
    """Test the robustness of the model class from an example LXCat data."""

    model = Model(EXAMPLE_LXCAT_FILE)

    results = model.fit()  # noqa: F841


@pytest.mark.parametrize("T", [0.5, 1.0, 3.0, 5.0, 10.0])
def test_rate_integral_maxwellian(T: float):
    """Test the rate integral calculation with the Maxwellian EEDF for an analytic example."""

    energies = linspace(start=1e-6, stop=100.0, num=10000, dtype=float)

    eedf = Maxwellian(T)

    # This gives rate integral as purely integral of the EEDF.
    xs = 1.0 / sqrt(energies)

    pdf = eedf.pdf(energies)

    rate = compute_rate_trapezoid(energies, xs, pdf)

    # EEDF is normalised, therefore result should be close to 1.
    assert isclose(rate, 1.0, atol=1e-3)

    rate = compute_rate_simpson(energies, xs, pdf)

    # EEDF is normalised, therefore result should be close to 1.
    assert isclose(rate, 1.0, atol=1e-3)


@pytest.mark.parametrize("T", [0.5, 1.0, 3.0, 5.0, 10.0])
def test_rate_integral_druyvesteyn(T: float):
    """Test the rate integral calculation with the Druyvesteyn EEDF for an analytic example."""

    energies = linspace(start=1e-6, stop=100.0, num=10000, dtype=float)

    eedf = Druyvesteyn(T)

    # This gives rate integral equal to 1.
    xs = 2.0 * sqrt(1.5 * T) * eedf.beta2 / eedf.beta1

    pdf = eedf.pdf(energies)

    rate = compute_rate_trapezoid(energies, xs, pdf)

    assert isclose(rate, 1.0, atol=1e-3)

    rate = compute_rate_simpson(energies, xs, pdf)

    assert isclose(rate, 1.0, atol=1e-3)

import re

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from lmfit.model import ModelResult

from xs2arr import Model
from xs2arr.cross_section import interpolate_xs
from xs2arr.eedf import EEDF, Druyvesteyn, Maxwellian
from xs2arr.io import parse_lxcat_data
from xs2arr.rate import compute_rate_simpson, compute_rate_trapezoid
from xs2arr.utils import arrhenius, arrhenius_log

EXAMPLE_LXCAT_FILE_FULL = "tests/example_lxcat.txt"
EXAMPLE_LXCAT_FILE_SHORT = "tests/example_lxcat_short.txt"


@given(
    eedf_type=st.sampled_from(("maxwellian", "druyvesteyn")),
    integrator=st.sampled_from(("simpson", "trapezoid")),
)
def test_model_init(eedf_type, integrator):
    """Test the model class from a variety of inputs."""

    eedf_grid = np.linspace(start=0.0, stop=100.0, num=10000, dtype=float)

    Model(
        EXAMPLE_LXCAT_FILE_SHORT,
        eedf_type=eedf_type,
        eedf_grid=eedf_grid,
        integrator=integrator,
    )


def test_model_init_errors():
    """Test the model class for errors."""

    with pytest.raises(TypeError, match="lxcat_file must be of type str, found "):
        Model(3)

    with pytest.raises(ValueError, match="lxcat_file cannot be an empty string"):
        Model("")

    with pytest.raises(TypeError, match="eedf_type must be of type str"):
        Model(EXAMPLE_LXCAT_FILE_SHORT, eedf_type=3)

    with pytest.raises(
        ValueError, match="eedf_type must be 'maxwellian' or 'druyvesteyn'"
    ):
        Model(EXAMPLE_LXCAT_FILE_SHORT, eedf_type="somethingelse")

    with pytest.raises(ValueError, match="eedf_grid must be 1-dimensional"):
        Model(EXAMPLE_LXCAT_FILE_SHORT, eedf_grid=[[1, 2], [3, 4], [5, 6]])

    with pytest.raises(ValueError, match=re.escape("len(eedf_grid) must be >= 2")):
        Model(EXAMPLE_LXCAT_FILE_SHORT, eedf_grid=[1])

    with pytest.raises(ValueError, match=r"All values in eedf_grid must be >= 0.0"):
        Model(EXAMPLE_LXCAT_FILE_SHORT, eedf_grid=[-1, 2, 8])

    with pytest.raises(TypeError, match="integrator must be of type str"):
        Model(EXAMPLE_LXCAT_FILE_SHORT, integrator=3)

    with pytest.raises(ValueError, match="integrator must be 'simpson' or 'trapezoid'"):
        Model(EXAMPLE_LXCAT_FILE_SHORT, integrator="somethingelse")


def test_model_fit(snapshot):
    """Test the robustness of the model fit method for an example LXCat data."""

    model = Model(EXAMPLE_LXCAT_FILE_FULL)

    model.fit()

    assert all(isinstance(result, ModelResult) for result in model.fitting_results)

    abc = np.array(model.get_abc(), dtype=float)

    assert np.round(abc, 6).tolist() == snapshot


def test_model_fit_errors():
    """Test the model fit method for errors."""
    model = Model(EXAMPLE_LXCAT_FILE_SHORT)

    with pytest.raises(
        ValueError, match="Only one of T_grid or mean_E_grid must be provided"
    ):
        model.fit(T_grid=[1, 2, 3], mean_E_grid=[1, 2, 3])

    with pytest.raises(ValueError, match="T_grid must be 1-dimensional"):
        model.fit(T_grid=[[1, 2], [3, 4], [5, 6]])

    with pytest.raises(ValueError, match=re.escape("len(T_grid) must be >= 2")):
        model.fit(T_grid=[1])

    with pytest.raises(ValueError, match=r"All values in T_grid must be >= 0.0"):
        model.fit(T_grid=[-1, 2, 8])

    with pytest.raises(TypeError, match="logarithmic must be of type bool"):
        model.fit(logarithmic=3)


def test_interpolation(snapshot):
    """Test the interpolation of the EEDF for an example LXCat data."""

    model = Model(EXAMPLE_LXCAT_FILE_SHORT)

    cross_section_info = model.cross_section_set.cross_sections[0]

    xs_energy = np.asarray(cross_section_info.data["energy"], dtype=float)
    xs = np.asarray(cross_section_info.data["cross section"], dtype=float)

    xs_interp = interpolate_xs(model.eedf_grid, xs_energy, xs)

    assert np.round(xs_interp, 6).tolist() == snapshot


def test_interpolation_errors():
    """Test the interpolation of the EEDF for errors."""

    with pytest.raises(ValueError, match=r"All values in xs must be >= 0.0"):
        interpolate_xs([1.0, 2.0], [0.0, 1.0], [0.0, -1.0])


def test_eedf(snapshot):
    """Test the EEDF creation and generation of PDF."""

    g = 1.5
    T = 5.0

    eedf = EEDF(g=g, Te=T)

    energies = np.linspace(start=1e-6, stop=100.0, num=10000, dtype=float)

    pdf = eedf.pdf(energies)

    assert np.round(pdf, 6).tolist() == snapshot
    assert np.isclose(eedf.g, g)
    assert np.isclose(eedf.Te, T)
    assert np.isclose(eedf.mean_E, 3.0 * T / 2.0)


def test_eedf_errors():
    """Test the EEDF creation for errors."""

    with pytest.raises(ValueError, match="g does not satisfy the condition"):
        EEDF(g=0.5, Te=5.0)

    with pytest.raises(ValueError, match="g does not satisfy the condition"):
        EEDF(g=2.5, Te=5.0)

    with pytest.raises(ValueError, match="Either Te or mean_E must be provided"):
        EEDF(g=1.5)

    with pytest.raises(ValueError, match="Only one of Te or mean_E must be provided"):
        EEDF(g=1.5, Te=5.0, mean_E=5.0)

    with pytest.raises(ValueError, match="mean_E does not satisfy the condition"):
        EEDF(g=1.5, mean_E=0.0)

    with pytest.raises(ValueError, match="Te does not satisfy the condition"):
        EEDF(g=1.5, Te=0.0)


@pytest.mark.parametrize("T", [0.5, 1.0, 3.0, 5.0, 10.0])
def test_rate_integral_maxwellian(T: float):
    """Test the rate integral calculation with the Maxwellian EEDF for an analytic example."""

    energies = np.linspace(start=1e-6, stop=100.0, num=10000, dtype=float)

    eedf = Maxwellian(T)

    # This gives rate integral as purely integral of the EEDF.
    xs = 1.0 / np.sqrt(energies)

    pdf = eedf.pdf(energies)

    rate = compute_rate_trapezoid(energies, xs, pdf)

    # EEDF is normalised, therefore result should be close to 1.
    assert np.isclose(rate, 1.0, atol=1e-3)

    rate = compute_rate_simpson(energies, xs, pdf)

    # EEDF is normalised, therefore result should be close to 1.
    assert np.isclose(rate, 1.0, atol=1e-3)


@pytest.mark.parametrize("T", [0.5, 1.0, 3.0, 5.0, 10.0])
def test_rate_integral_druyvesteyn(T: float):
    """Test the rate integral calculation with the Druyvesteyn EEDF for an analytic example."""

    energies = np.linspace(start=1e-6, stop=100.0, num=10000, dtype=float)

    eedf = Druyvesteyn(T)

    # This gives rate integral equal to 1.
    xs = 2.0 * np.sqrt(1.5 * T) * eedf.beta2 / eedf.beta1

    pdf = eedf.pdf(energies)

    rate = compute_rate_trapezoid(energies, xs, pdf)

    assert np.isclose(rate, 1.0, atol=1e-3)

    rate = compute_rate_simpson(energies, xs, pdf)

    assert np.isclose(rate, 1.0, atol=1e-3)


def test_io_errors():
    """Test the IO functions for errors."""

    with pytest.raises(TypeError, match="filename must be of type str"):
        parse_lxcat_data(3)

    with pytest.raises(ValueError, match="filename cannot be an empty string"):
        parse_lxcat_data("")


def test_arrhenius(snapshot):
    """Test the Arrhenius equation."""

    a = 1e-15
    b = 0.5
    c = -10.0

    T_grid = np.linspace(start=0.001, stop=6.0, num=1000, dtype=float)

    arr = arrhenius(T_grid, a, b, c)

    assert np.round(arr, 6).tolist() == snapshot


def test_arrhenius_log(snapshot):
    """Test the log of the Arrhenius equation."""

    log10_a = -15.0
    b = 0.5
    c = -10.0

    T_grid = np.linspace(start=0.001, stop=6.0, num=1000, dtype=float)

    arr_log = arrhenius_log(T_grid, log10_a, b, c)

    assert np.round(arr_log, 6).tolist() == snapshot

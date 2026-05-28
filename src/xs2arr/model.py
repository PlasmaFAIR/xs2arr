from collections.abc import Callable
from pathlib import Path

import numpy as np
from lmfit import Model as FittingModel
from lmfit import Parameters, create_params
from lxcat_data_parser import CrossSectionSet
from numpy.typing import ArrayLike
from scipy import constants

from xs2arr.cross_section import interpolate_xs
from xs2arr.eedf import Druyvesteyn, Maxwellian
from xs2arr.io import (
    _clear_file,
    _write_header,
    _write_reaction,
    _write_reaction_footer,
    _write_reaction_header,
    parse_lxcat_data,
)
from xs2arr.rate import compute_rate_simpson, compute_rate_trapezoid
from xs2arr.utils import arrhenius, arrhenius_log


class Model:
    def __init__(
        self,
        lxcat_file: str,
        *,
        eedf_type: str = "maxwellian",
        eedf_grid: np.ndarray | None = None,
        integrator: str = "simpson",
    ) -> None:
        self.cross_section_set = _validate_and_prepare_lxcat_file(lxcat_file)
        self.eedf_cls = _validate_and_prepare_eedf(eedf_type)
        self.eedf_grid = _validate_and_prepare_eedf_grid(eedf_grid)
        self.rate_computer = _validate_and_prepare_integrator(integrator)
        self.fitting_results = None

    def fit(
        self,
        T_grid: ArrayLike | None = None,
        *,
        mean_E_grid: ArrayLike | None = None,
        logarithmic: bool = True,
    ) -> None:
        """
        Fits the Arrhenius equation to rate coefficients a b, and c, calculated from cross sections. Stores results in
        model.

        Parameters
        ----------
        T_grid : optional
            Temperature grid for rate calculations. If None and mean_E_grid is None,
            a default grid of 1000 points between 0.001 and 6 eV will be used.
        mean_E_grid : optional
            Mean energy grid to define the temperature grid. If provided, T_grid will be
            calculated as 2/3 of mean_E_grid.
        logarithmic : optional
            Whether to perform the fitting in logarithmic space. Default is True.

        Raises
        ------
        TypeError
            If the method is called on the class instead of an instance
        ValueError
            If both T_grid and mean_E_grid are provided
        """

        # Only allow a user to call arrhenius from an instantiated object of the class.
        if isinstance(self, type):
            raise TypeError(
                "This method is not intended to be called directly - instantiate an object of the class first"
            )

        T_grid = _validate_and_prepare_T_grid(T_grid, mean_E_grid)

        if not isinstance(logarithmic, bool):
            raise TypeError("logarithmic must be of type bool")

        regressor, params = _create_fitting_model(logarithmic)

        results = []

        for cross_section_info in self.cross_section_set.cross_sections:
            xs_energy = np.asarray(cross_section_info.data["energy"], dtype=float)
            xs = np.asarray(cross_section_info.data["cross section"], dtype=float)

            # Interpolate the cross section onto the EEDF grid.
            xs_interp = interpolate_xs(self.eedf_grid, xs_energy, xs)

            # Perform the rate integrals.
            rates = [
                self.rate_computer(
                    self.eedf_grid, xs_interp, self.eedf_cls(T).pdf(self.eedf_grid)
                )
                for T in T_grid
            ]

            rates = np.asarray(rates, dtype=float)

            # Convert rates to appropriate units.
            rates *= np.sqrt(
                2.0 * constants.elementary_charge / constants.electron_mass
            )

            T_grid, rates = _remove_bad_data(T_grid, rates, logarithmic)

            rates = np.log10(rates) if logarithmic else rates

            fitting = regressor.fit(rates, params, T=T_grid)

            results.append(fitting)

        self.fitting_results = results

    def get_abc(self) -> list[tuple[float, float, float]]:
        """
        Extracts the fitted Arrhenius parameters (a, b, c) for each cross section in the set.

        Raises
        ------
        ValueError
            If no fitting results are available (fit() hasn't been called yet).
        """

        if self.fitting_results is None:
            raise ValueError("No fitting results to extract")

        return [
            _get_true_abc(fitting_results) for fitting_results in self.fitting_results
        ]

    def write_results(self, output_file: str, *, append_to_file: bool = False) -> None:
        """
        Write results determined by the Arrhenius fitting method to a file.

        Parameters
        ----------
        output_file : str
            Output file to write formatted results to.
        append_to_file: bool
            Whether to append the results to an existing file. If False, the file will be overwritten.
        """

        if self.fitting_results is None:
            raise ValueError("No fitting results to write")

        if not isinstance(output_file, str):
            raise TypeError("Output file must be of type str")
        if not output_file.strip():
            raise ValueError("Output file cannot be an empty string")

        output_file = Path(output_file)

        if not append_to_file:
            _clear_file(output_file)

        _write_header(output_file)

        _write_reaction_header(
            output_file, reaction_type="default"
        )  # Only have "default" reactions for now.

        abc = self.get_abc()

        for cross_section_info, (a, b, c) in zip(
            self.cross_section_set.cross_sections, abc
        ):
            reaction = cross_section_info.info.get("PROCESS", None)

            if reaction is None:
                print(f"Skipping {cross_section_info} as no process type specified.")
                continue

            # Remove the reaction type if it exists.
            reaction = reaction.replace(", Attachment", "")
            reaction = reaction.replace(", Elastic", "")
            reaction = reaction.replace(", Excitation", "")
            reaction = reaction.replace(", Ionization", "")

            energy_str = cross_section_info.info.get("PARAM.", None)

            if energy_str is None:
                print(f"Skipping {cross_section_info} as no energy specified.")
                continue

            try:
                energy_str = energy_str[energy_str.index("=") + 1 :].strip()
            except ValueError:
                print(
                    f"Skipping {cross_section_info} as energy value improperly specified."
                )
                continue

            try:
                energy = float(energy_str[: energy_str.index("eV")].strip())
            except ValueError:
                print(
                    f"Skipping {cross_section_info} as energy value improperly specified."
                )
                continue

            _write_reaction(output_file, reaction, energy, a, b, c)

        _write_reaction_footer(output_file, reaction_type="default")


def _validate_and_prepare_lxcat_file(lxcat_file: str) -> CrossSectionSet:
    """
    Validates and prepares the LXCAT data file for cross section analysis. Returns the LXCat data parsed as a set of
    cross sections.

    Parameters
    ----------
    lxcat_file
        Path to the LXCAT data file containing cross section information.

    Raises
    ------
    ValueError
        If lxcat_file is an empty string.
    TypeError
        If lxcat_file is not a string.
    """

    if not isinstance(lxcat_file, str):
        raise TypeError(f"lxcat_file must be of type str, found {type(lxcat_file)}")
    if not lxcat_file.strip():
        raise ValueError("lxcat_file cannot be an empty string")

    return parse_lxcat_data(lxcat_file)


def _validate_and_prepare_eedf(eedf_type: str) -> type[Maxwellian | Druyvesteyn]:
    """
    Validates and prepares the electron energy distribution function (EEDF) type. Returns the EEDF class.

    Parameters
    ----------
    eedf_type
        Type of EEDF to use. Must be either 'maxwellian' or 'druyvesteyn'.

    Raises
    ------
    TypeError
        If eedf_type is not a string.
    ValueError
        If eedf_type is not 'maxwellian' or 'druyvesteyn'.
    """

    if not isinstance(eedf_type, str):
        raise TypeError("eedf_type must be of type str")
    if eedf_type not in ("maxwellian", "druyvesteyn"):
        raise ValueError("eedf_type must be 'maxwellian' or 'druyvesteyn'")

    return Maxwellian if eedf_type == "maxwellian" else Druyvesteyn


def _validate_and_prepare_eedf_grid(eedf_grid: ArrayLike | None) -> np.ndarray:
    """
    Validates and prepares the energy grid for the electron energy distribution function (EEDF). Returns the validated
    EEDF grid.

    Parameters
    ----------
    eedf_grid
        Grid of energy values for EEDF calculation. If None, a default grid of 10000 points between 0 and 100 will
        be used.

    Raises
    ------
    ValueError
        If eedf_grid is not 1-dimensional, has less than 2 points, or contains negative values.
    """
    if eedf_grid is None:
        eedf_grid = np.linspace(start=0.0, stop=100.0, num=10000, dtype=float)
    eedf_grid = np.asarray(eedf_grid, dtype=float)
    if eedf_grid.ndim != 1:
        raise ValueError("eedf_grid must be 1-dimensional")
    if len(eedf_grid) < 2:
        raise ValueError("len(eedf_grid) must be >= 2")
    if np.any(eedf_grid < 0.0):
        raise ValueError("All values in eedf_grid must be >= 0.0")

    return eedf_grid


def _validate_and_prepare_integrator(
    integrator: str,
) -> Callable[[np.array, np.array, np.array], float]:
    """
    Validates and prepares the integration method for rate computation. Returns the rate computation function.

    Parameters
    ----------
    integrator
        The integration method to use. Must be either 'simpson' or 'trapezoid'.

    Raises
    ------
    TypeError
        If integrator is not a string.
    ValueError
        If integrator is not 'simpson' or 'trapezoid'.
    """

    if not isinstance(integrator, str):
        raise TypeError("integrator must be of type str")
    if integrator not in ("trapezoid", "simpson"):
        raise ValueError("integrator must be 'simpson' or 'trapezoid'")

    return compute_rate_simpson if integrator == "simpson" else compute_rate_trapezoid


def _validate_and_prepare_T_grid(
    T_grid: ArrayLike | None, mean_E_grid: ArrayLike | None
) -> np.ndarray:
    """
    Validates and prepares the temperature grid for rate calculations. Returns the validated temperature grid.

    Parameters
    ----------
    T_grid
        Temperature grid for rate calculations. If None and mean_E_grid is None,
        a default grid of 1000 points between 0.001 and 6 eV will be used.
    mean_E_grid
        Mean energy grid to define the temperature grid. If provided, T_grid will be
        calculated as 2/3 of mean_E_grid.

    Raises
    ------
    ValueError
        If both T_grid and mean_E_grid are provided,
        If provided grid is not 1-dimensional,
        If provided grid has less than 2 points,
        If provided grid contains negative values.
    """
    if T_grid is not None and mean_E_grid is not None:
        raise ValueError("Only one of T_grid or mean_E_grid must be provided")
    if T_grid is None and mean_E_grid is None:
        T_grid = np.linspace(start=0.001, stop=6.0, num=1000, dtype=float)
    if mean_E_grid is not None:
        T_grid = 2.0 * np.asarray(mean_E_grid, dtype=float) / 3.0
    T_grid = np.asarray(T_grid, dtype=float)
    if T_grid.ndim != 1:
        raise ValueError("T_grid must be 1-dimensional")
    if len(T_grid) < 2:
        raise ValueError("len(T_grid) must be >= 2")
    if np.any(T_grid < 0.0):
        raise ValueError("All values in T_grid must be >= 0.0")

    return T_grid


def _create_fitting_model(logarithmic: bool) -> tuple[FittingModel, Parameters]:
    """
    Creates a fitting model and its parameters for Arrhenius equation fitting. Returns the model and parameters.

    Parameters
    ----------
    logarithmic
        Whether to use logarithmic form of the Arrhenius equation for fitting.

    Notes
    -----
        Default parameter values are:

        - ``log10_a = -14.0`` (or ``a = 1e-14`` for non-logarithmic)
        - ``b = 0.1``
        - ``c = -10.0`` (with maximum value of 0)
    """
    if logarithmic:
        params = create_params(
            log10_a={"value": -14.0}, b=0.1, c={"value": -10.0, "max": 0.0}
        )
    else:
        params = create_params(
            a={"value": 1e-14}, b=0.1, c={"value": -10.0, "max": 0.0}
        )

    regressor = FittingModel(
        arrhenius_log if logarithmic else arrhenius, independent_vars=["T"]
    )

    return regressor, params


def _remove_bad_data(
    T_grid: np.ndarray, rates: np.ndarray, logarithmic: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Removes data points with zero rates when performing logarithmic fitting. Returns the filtered temperature grid and
    rates array.

    Parameters
    ----------
    T_grid
        Array of temperature values.
    rates
        Array of rate values corresponding to the temperature grid.
    logarithmic : optional
        Whether logarithmic fitting is being performed. Default is True.

    Notes
    -----
        If logarithmic is False then the original arrays are unmodified.
    """

    if not logarithmic:
        return T_grid, rates

    # Remove entries that have rate exactly as 0 if doing logarithmic fitting.
    mask = rates > 0.0

    return T_grid[mask], rates[mask]


def _get_true_abc(fitting: FittingModel) -> tuple[float, float, float]:
    """
    Extracts the true a, b, and c parameters from the fitting model. Returns a tuple containing the a, b, and c where a
    has been converted from ``log10`` if a logarithmic fitting was performed.

    Parameters
    ----------
    fitting
        The fitted model containing the Arrhenius parameters.
    """

    if not ("log10_a" in fitting.params or "a" in fitting.params):
        raise ValueError("Arrhenius parameter for 'a' not found in the model")

    if "log10_a" in fitting.params and "a" in fitting.params:
        raise ValueError(
            "Both 'log10_a' and 'a' parameters found in the model. Only one should be used"
        )

    if "b" not in fitting.params:
        raise ValueError("Arrhenius parameter for 'b' not found in the model")

    if "c" not in fitting.params:
        raise ValueError("Arrhenius parameter for 'c' not found in the model")

    logarithmic = "log10_a" in fitting.params

    a = (
        (10.0 ** fitting.params["log10_a"].value)
        if logarithmic
        else fitting.params["a"].value
    )
    b = fitting.params["b"].value
    c = fitting.params["c"].value

    return a, b, c

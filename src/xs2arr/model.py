import numpy as np
from lmfit import Model as FittingModel
from lmfit import Parameters, create_params

from xs2arr.cross_section import interpolate_xs
from xs2arr.eedf import Druyvesteyn, Maxwellian
from xs2arr.io import parse_lxcat_data
from xs2arr.rate import compute_rate
from xs2arr.utils import arrhenius, arrhenius_log, m_e, q


class Model:
    def __init__(
        self,
        lxcat_file: str | None = None,
        eedf_type: str = "maxwellian",
        eedf_grid: np.ndarray | None = None,
    ):
        if lxcat_file is None:
            raise ValueError("No lxcat file provided")
        if not isinstance(lxcat_file, str):
            raise TypeError("lxcat_file must be of type str")
        if not lxcat_file:
            raise ValueError("lxcat_file cannot be an empty string")

        if not isinstance(eedf_type, str):
            raise TypeError("eedf_type must be of type str")
        if eedf_type not in ("maxwellian", "druyvesteyn"):
            raise ValueError("eedf_type must be 'maxwellian' or 'druyvesteyn'")

        if eedf_grid is None:
            eedf_grid = np.linspace(start=0.0, stop=100.0, num=10000, dtype=float)
        if isinstance(eedf_grid, list | tuple):
            eedf_grid = np.array(eedf_grid, dtype=float)
        if not isinstance(eedf_grid, np.ndarray):
            raise TypeError("eedf_grid must be of type ndarray")
        if eedf_grid.ndim != 1:
            raise ValueError("eedf_grid must be 1-dimensional")
        if len(eedf_grid) < 2:
            raise ValueError("len(eedf_grid) must be >= 2")
        if np.any(eedf_grid < 0.0):
            raise ValueError("All values in eedf_grid must be >= 0.0")

        self.cross_section_set = parse_lxcat_data(lxcat_file)
        self.eedf_cls = Maxwellian if eedf_type == "maxwellian" else Druyvesteyn
        self.eedf_grid = eedf_grid

    def fit(
        self,
        T_grid: np.ndarray | None = None,
        mean_E_grid: np.ndarray | None = None,
        logarithmic: bool = True,
    ) -> list[tuple[FittingModel, float, float, float]]:
        # Only allow a user to call arrhenius from an instantiated object of the class.
        if isinstance(self, type):
            raise TypeError(
                "This method is not intended to be called directly - instantiate an object of the class first"
            )

        if T_grid is not None and mean_E_grid is not None:
            raise ValueError("Only one of T_grid or mean_E_grid must be provided")
        if T_grid is None and mean_E_grid is None:
            T_grid = np.linspace(start=0.001, stop=6.0, num=1000, dtype=float)
        if mean_E_grid is not None:
            T_grid = 2.0 * mean_E_grid / 3.0
        if isinstance(T_grid, list | tuple):
            T_grid = np.array(T_grid, dtype=float)
        if not isinstance(T_grid, np.ndarray):
            raise TypeError("T_grid must be of type ndarray")
        if T_grid.ndim != 1:
            raise ValueError("T_grid must be 1-dimensional")
        if len(T_grid) < 2:
            raise ValueError("len(T_grid) must be >= 2")
        if np.any(T_grid < 0.0):
            raise ValueError("All values in T_grid must be >= 0.0")

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
                compute_rate(
                    self.eedf_grid, xs_interp, self.eedf_cls(T).pdf(self.eedf_grid)
                )
                for T in T_grid
            ]

            rates = np.asarray(rates, dtype=float)

            # Convert rates to appropriate units.
            rates *= np.sqrt(2.0 * q / m_e)

            T_grid, rates = _remove_bad_data(T_grid, rates, logarithmic)

            rates = np.log10(rates) if logarithmic else rates

            fitting = regressor.fit(rates, params, T=T_grid)

            a_true, b_true, c_true = _get_true_abc(fitting, logarithmic)

            results.append((fitting, a_true, b_true, c_true))

        return results


def _create_fitting_model(logarithmic: bool) -> tuple[FittingModel, Parameters]:
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
    if not logarithmic:
        return T_grid, rates

    # Remove entries that have rate exactly as 0 if doing logarithmic fitting.
    mask = rates > 0.0

    return T_grid[mask], rates[mask]


def _get_true_abc(
    fitting: FittingModel, logarithmic: bool = True
) -> tuple[float, float, float]:
    a = (
        (10.0 ** fitting.params["log10_a"].value)
        if logarithmic
        else fitting.params["a"].value
    )
    b = fitting.params["b"].value
    c = fitting.params["c"].value

    return a, b, c

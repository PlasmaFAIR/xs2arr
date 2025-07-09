from lmfit import Model as FittingModel
from lmfit import Parameters
from numpy import array, asarray, linspace, ndarray, sqrt

from xs2arr.cross_section import interpolate_xs
from xs2arr.eedf import Druyvesteyn, Maxwellian
from xs2arr.io import parse_lxcat_data
from xs2arr.rate import compute_rate
from xs2arr.utils import arrhenius, m_e, q


class Model:
    def __init__(
        self,
        lxcat_file: str | None = None,
        eedf_type: str = "maxwellian",
        eedf_grid: ndarray | None = None,
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
            eedf_grid = linspace(start=0.0, stop=100.0, num=10000, dtype=float)
        if isinstance(eedf_grid, list | tuple):
            eedf_grid = array(eedf_grid, dtype=float)
        if not isinstance(eedf_grid, ndarray):
            raise TypeError("eedf_grid must be of type ndarray")
        if eedf_grid.ndim != 1:
            raise ValueError("eedf_grid must be 1-dimensional")
        if len(eedf_grid) < 2:
            raise ValueError("len(eedf_grid) must be >= 2")
        if any(eedf_grid < 0.0):
            raise ValueError("All values in eedf_grid must be >= 0.0")

        self.cross_section_set = parse_lxcat_data(lxcat_file)
        self.eedf_cls = Maxwellian if eedf_type == "maxwellian" else Druyvesteyn
        self.eedf_grid = eedf_grid

    def arrhenius(
        self, T_grid: ndarray | None = None, mean_E_grid: ndarray | None = None
    ):
        # Only allow a user to call arrhenius from an instantiated object of the class.
        if isinstance(self, type):
            raise TypeError(
                "This method is not intended to be called directly - instantiate an object of the class first"
            )

        if T_grid is not None and mean_E_grid is not None:
            raise ValueError("Only one of T_grid or mean_E_grid must be provided")
        if T_grid is None and mean_E_grid is None:
            T_grid = linspace(start=0.001, stop=6.0, num=1000, dtype=float)
        if mean_E_grid is not None:
            T_grid = 2.0 * mean_E_grid / 3.0
        if isinstance(T_grid, list | tuple):
            T_grid = array(T_grid, dtype=float)
        if not isinstance(T_grid, ndarray):
            raise TypeError("T_grid must be of type ndarray")
        if T_grid.ndim != 1:
            raise ValueError("T_grid must be 1-dimensional")
        if len(T_grid) < 2:
            raise ValueError("len(T_grid) must be >= 2")
        if any(T_grid < 0.0):
            raise ValueError("All values in T_grid must be >= 0.0")

        # For the Arrhenius equation fittings.
        params = Parameters()
        params.add("a", value=1e-14)
        params.add("b", value=0.1)
        params.add("c", value=-10.0, max=0.0)

        results = []

        for cross_section_info in self.cross_section_set.cross_sections:
            xs_energy = asarray(cross_section_info.data["energy"], dtype=float)
            xs = asarray(cross_section_info.data["cross section"], dtype=float)

            # Interpolate the cross section onto the EEDF grid.
            xs_interp = interpolate_xs(self.eedf_grid, xs_energy, xs)

            # Perform the rate integrals.
            rates = [
                compute_rate(
                    self.eedf_grid, xs_interp, self.eedf_cls(T).pdf(self.eedf_grid)
                )
                for T in T_grid
            ]

            rates = asarray(rates, dtype=float)

            # Convert rates to appropriate units.
            rates *= sqrt(2.0 * q / m_e)

            # And compute the a, b and c Arrhenius coefficients.
            regressor = FittingModel(arrhenius, independent_vars=["T"])

            results.append(regressor.fit(rates, params, T=T_grid))

        return results

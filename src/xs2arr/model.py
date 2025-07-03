from numpy import arange, array, ndarray

from xs2arr.eedf import Druyvesteyn, Maxwellian
from xs2arr.io import parse_lxcat_data


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
            eedf_grid = arange(start=0.0, stop=100.0, step=0.01, dtype=float)
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

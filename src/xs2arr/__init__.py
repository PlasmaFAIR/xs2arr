from importlib.metadata import (
    PackageNotFoundError as _PackageNotFoundError,
)
from importlib.metadata import (
    version as _version,
)

from xs2arr.cross_section import interpolate_xs
from xs2arr.eedf import EEDF, Druyvesteyn, Maxwellian
from xs2arr.io import parse_lxcat_data
from xs2arr.model import Model
from xs2arr.rate import compute_rate_simpson, compute_rate_trapezoid
from xs2arr.utils import arrhenius, arrhenius_log, constants

try:
    __version__ = _version(__name__)
except _PackageNotFoundError:
    __version__ = "0.0.0"


__all__ = [
    "EEDF",
    "Druyvesteyn",
    "Maxwellian",
    "Model",
    "__version__",
    "arrhenius",
    "arrhenius_log",
    "compute_rate_simpson",
    "compute_rate_trapezoid",
    "constants",
    "interpolate_xs",
    "parse_lxcat_data",
]

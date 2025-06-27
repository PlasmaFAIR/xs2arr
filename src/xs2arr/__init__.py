from xs2arr.main import run
from xs2arr.model import Model
from xs2arr.constants import EXAMPLE_LXCAT_FILE

from importlib.metadata import version, PackageNotFoundError


try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # Package hasn't been installed
    pass

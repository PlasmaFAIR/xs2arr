from xs2arr.model import Model

from importlib.metadata import (
    version as _version,
    PackageNotFoundError as _PackageNotFoundError,
)

__all__ = ["Model"]


try:
    __version__ = _version(__name__)
except _PackageNotFoundError:
    # Package hasn't been installed
    pass

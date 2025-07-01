from contextlib import suppress
from importlib.metadata import (
    PackageNotFoundError as _PackageNotFoundError,
)
from importlib.metadata import (
    version as _version,
)

from xs2arr.model import Model

__all__ = ["Model"]


with suppress(_PackageNotFoundError):
    __version__ = _version(__name__)

from xs2arr.main import main

from importlib.metadata import version, PackageNotFoundError


try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # Package hasn't been installed
    pass

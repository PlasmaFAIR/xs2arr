import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import simpson


def compute_rate_trapezoid(
    energies: ArrayLike, cross_section: ArrayLike, eedf: ArrayLike
) -> float:
    """
    Returns the computed rection rate using trapezoidal integration.

    Parameters
    ----------
    energies
        Energy values.
    cross_section
        Cross section values corresponding to energies.
    eedf
        Electron energy distribution function values.
    """
    energies = np.asarray(energies, dtype=float)
    cross_section = np.asarray(cross_section, dtype=float)
    eedf = np.asarray(eedf, dtype=float)

    return np.trapezoid(np.sqrt(energies) * cross_section * eedf, x=energies)


def compute_rate_simpson(
    energies: ArrayLike, cross_section: ArrayLike, eedf: ArrayLike
) -> float:
    """
    Returns the computed rection rate using Simpson's integration.

    Parameters
    ----------
    energies
        Energy values.
    cross_section
        Cross section values corresponding to energies.
    eedf
        Electron energy distribution function values.
    """
    energies = np.asarray(energies, dtype=float)
    cross_section = np.asarray(cross_section, dtype=float)
    eedf = np.asarray(eedf, dtype=float)

    return simpson(np.sqrt(energies) * cross_section * eedf, x=energies)
